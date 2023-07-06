import math
from typing import Dict, List

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from cvpods.layers import ShapeSpec, NaiveGroupNorm, DFConv2d, NaiveSyncBatchNorm, DeformConv, ModulatedDeformConv
from cvpods.structures import ImageList, Instances, Boxes, pairwise_iou_tensor
from cvpods.modeling.postprocessing import detector_postprocess, sem_seg_postprocess
from cvpods.structures.masks import polygons_to_bitmask
from cvpods.layers.conv_with_kaiming_uniform import conv_with_kaiming_uniform
from cvpods.modeling.losses import sigmoid_focal_loss_jit
from cvpods.layers import cat, Conv2d
from cvpods.modeling.poolers import ROIPooler
from cvpods.layers.dynamic_conv import DynamicBottleneck, get_module_running_cost, DynamicScale
from cvpods.utils import get_event_storage

INF = 100000000

def build_blender(cfg):
    return Blender(cfg)


def build_basis_module(cfg, input_shape):
    return ProtoNet(cfg, input_shape)


class DynamicTemporalAlign(nn.Module):
    """
        Implementation of Dynamic Temporal Align Heads
    """

    def __init__(self, cfg, input_shape):
        super(DynamicTemporalAlign, self).__init__()

        self.align_features = cfg.MODEL.BLENDMASK.TEMPORAL_ALIGN.IN_FEATURES                    # p3, p4, p5, p6, p7
        self.num_conv_per_layer = cfg.MODEL.BLENDMASK.TEMPORAL_ALIGN.NUM_CONVS                  # 2
        self.align_channels = cfg.MODEL.BLENDMASK.TEMPORAL_ALIGN.ALIGN_CHANNELS                 # 256
        self.dcnv2 = cfg.MODEL.BLENDMASK.TEMPORAL_ALIGN.USE_DCN_v2
        self.dcn_on = cfg.MODEL.BLENDMASK.TEMPORAL_ALIGN.USE_DCN
        self.num_offset_gen_conv = cfg.MODEL.BLENDMASK.TEMPORAL_ALIGN.NUM_OFFSET_GEN_CONV
        self.share_convs = cfg.MODEL.BLENDMASK.TEMPORAL_ALIGN.SHARE_CONVS
        self.num_layers = len(self.align_features)
        self.num_conv_routing_list = cfg.MODEL.BLENDMASK.TEMPORAL_ALIGN.NUM_CONV_ROUTING_LIST
        norm = None if cfg.MODEL.BLENDMASK.TEMPORAL_ALIGN.NORM == 'none' else cfg.MODEL.BLENDMASK.TEMPORAL_ALIGN.NORM

        input_shapes = [input_shape[f] for f in self.align_features]
        in_channels = [s.channels for s in input_shapes]
        assert len(set(in_channels)) == 1, "Each input level must have the same channel!"
        in_channels = in_channels[0]
        self.in_channels = in_channels

        # default by dynamic network
        gate_activation = "GeReTanH"
        gate_activation_kargs = dict(tau=1.5)

        if self.dcn_on:
            self.fuse1 = DFConv2d(
                in_channels=self.in_channels * 2,
                out_channels=self.in_channels,
                kernel_size=3,
                stride=1,
                padding=1)
        else:
            self.fuse1 = nn.Conv2d(
                in_channels=self.in_channels * 2,
                out_channels=self.in_channels,
                kernel_size=3,
                stride=1,
                padding=1)
        offset_gen = []
        for i in range(self.num_offset_gen_conv):
            if i == 0:
                in_planes = 2 * self.in_channels
            else:
                in_planes = self.align_channels
            if i == self.num_offset_gen_conv - 1:
                if self.dcnv2:
                    out_planes = 27 * 2
                else:
                    out_planes = 18 * 2
            else:
                out_planes = self.align_channels
            offset_gen.append(nn.Conv2d(in_channels=in_planes,
                                        out_channels=out_planes,
                                        kernel_size=3,
                                        padding=1,
                                        stride=1,
                                        bias=False))
            if i != self.num_offset_gen_conv - 1:
                if norm == "GN":
                    offset_gen.append(nn.GroupNorm(32, out_planes))
                elif norm == "NaiveGN":
                    offset_gen.append(NaiveGroupNorm(32, out_planes))
                elif norm == "BN":
                    if self.share_convs:
                        offset_gen.append(ModuleListDial([
                            nn.BatchNorm2d(out_planes) for _ in range(self.num_layers)
                        ]))
                    else:
                        offset_gen.append(nn.BatchNorm2d(out_planes))
                elif norm == "SyncBN":
                    if self.share_convs:
                        offset_gen.append(ModuleListDial([
                            NaiveSyncBatchNorm(out_planes) for _ in range(self.num_layers)
                        ]))
                    else:
                        offset_gen.append(NaiveSyncBatchNorm(out_planes))
                offset_gen.append(nn.ReLU())
        self.add_module('offset_gen', nn.Sequential(*offset_gen))

        if self.dcnv2:
            fuse_conv = ModulatedDeformConv
        else:
            fuse_conv = DeformConv
        self.fuse2_1 = fuse_conv(in_channels=self.in_channels,
                                 out_channels=self.in_channels,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1,
                                 bias=False)
        self.fuse2_2 = fuse_conv(in_channels=self.in_channels,
                                 out_channels=self.in_channels,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1,
                                 bias=False)

        align_tower = []
        if self.dcn_on:
            conv_func = DFConv2d
        else:
            conv_func = nn.Conv2d
        align_tower.append(
            conv_func(
            self.in_channels * 2,
            self.in_channels,
            kernel_size=1,
            stride=1,
            padding=1)
        )
        for i in range(self.num_conv_per_layer):
            align_tower.append(DynamicBottleneck(
                self.in_channels,
                self.in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                norm="GN",
                num_groups=1,
                gate_activation=gate_activation,
                gate_activation_kargs=gate_activation_kargs
            ))
        self.add_module('align_tower', nn.Sequential(*align_tower))

        self.gate = nn.Conv2d(2 * in_channels, 1, kernel_size=1, stride=1, bias=True)

        for l in self.offset_gen.modules():
            if isinstance(l, nn.Conv2d):
                nn.init.normal_(l.weight, std=0.01)
                if l.bias is not None:
                    nn.init.constant_(l.bias, 0)

        for l in self.align_tower.modules():
            if isinstance(l, nn.Conv2d):
                nn.init.normal_(l.weight, std=0.01)
                if l.bias is not None:
                    nn.init.constant_(l.bias, 0)
        nn.init.constant_(self.gate.weight, 0)
        nn.init.constant_(self.gate.bias, 1)

        self.fuse_subnets = []
        assert len(self.num_conv_routing_list) == len(self.align_features)

        for i, num_convs in enumerate(self.num_conv_routing_list):
            fuse_subnet = []
            for _ in range(num_convs):
                fuse_subnet_conv = DynamicBottleneck(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm="GN",
                    num_groups=1,
                    gate_activation=gate_activation,
                    gate_activation_kargs=gate_activation_kargs
                )
                fuse_subnet.append(
                    DynamicScale(
                        in_channels,
                        in_channels,
                        num_convs=1,
                        kernel_size=3,
                        padding=1,
                        stride=1,
                        num_groups=1,
                        num_adjacent_scales=2,
                        resize_method="bilinear",
                        depth_module=fuse_subnet_conv,
                        gate_activation=gate_activation,
                        gate_activation_kargs=gate_activation_kargs
                    )
                )
            self.fuse_subnets.append(nn.Sequential(*fuse_subnet))

        self.fuse_subnets = nn.ModuleList(self.fuse_subnets)

        for l in self.fuse_subnets.modules():
            if isinstance(l, nn.Conv2d):
                nn.init.normal_(l.weight, std=0.01)
                if l.bias is not None:
                    nn.init.constant_(l.bias, 0)

    def forward(self, reference_features, query_features):
        reference_feats_list = [reference_features[f] for f in self.align_features]
        query_feats_list = [query_features[f] for f in self.align_features]
        assert len(reference_feats_list) == len(query_feats_list), "reference features must equal to query features"

        aligned_query = []
        aligned_query.append(None)

        for i in range(len(query_feats_list)):

            reference_feat = reference_feats_list[i]
            query_feat = query_feats_list[i]
            aligned_feat = self.align_feature_func(reference_feat, query_feat)  # first to align the feature
            query_feats_list[i] = aligned_feat

            if i < 3:
                query_feats_list[i:] = self.fuse_subnets[i](query_feats_list[i:])
            else:
                query_feats_list[-3:] = self.fuse_subnets[i](query_feats_list[-3:])

            aligned_query.append(query_feats_list[i])

        for i, key in enumerate(query_features.keys()):
            if i == 0:
                continue
            else:
                query_features[key] = aligned_query[i]

        return query_features

    def align_feature_func(self, reference_feat, query_feat):
        """
            share the aligned tower
        """
        concat_feat = cat([reference_feat, query_feat], dim=1)  # N, 2*C, H, W
        offset = self.offset_gen(concat_feat)
        offset_channels = offset.shape[1]
        offset1 = offset[:, :offset_channels // 2, :, :]
        offset2 = offset[:, offset_channels // 2:, :, :]
        if self.dcnv2:
            offset_1 = offset1[:, :18, :, :]
            mask_1 = offset1[:, 18:, :, :].sigmoid()
            offset_2 = offset2[:, :18, :, :]
            mask_2 = offset2[:, 18:, :, :].sigmoid()
            fused21 = self.fuse2_1(reference_feat, offset_1, mask_1)
            fused22 = self.fuse2_2(query_feat, offset_2, mask_2)
        else:
            fused21 = self.fuse2_1(reference_feat, offset1)
            fused22 = self.fuse2_2(query_feat, offset2)
        concat_feat = cat([fused21, fused22], dim=1)
        gate = torch.sigmoid(self.gate(concat_feat))  # N, 1, H, W
        aligned_feat = self.align_tower(concat_feat)  # N, C, H, W
        aligned_feat = gate * query_feat + (1 - gate) * aligned_feat

        return aligned_feat


class ProtoNet(nn.Module):
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        """
        TODO: support deconv and variable channel width
        """
        # official protonet has a relu after each conv
        super().__init__()
        # fmt: off
        mask_dim          = cfg.MODEL.BASIS_MODULE.NUM_BASES
        planes            = cfg.MODEL.BASIS_MODULE.CONVS_DIM
        self.in_features  = cfg.MODEL.BASIS_MODULE.IN_FEATURES
        self.loss_on      = cfg.MODEL.BASIS_MODULE.LOSS_ON
        # seg loss by condtional inst
        self.condition_loss_on = cfg.MODEL.BASIS_MODULE.CONDTION_INST_LOSS_ON
        self.out_stride = input_shape[self.in_features[0]].stride

        norm              = cfg.MODEL.BASIS_MODULE.NORM
        num_convs         = cfg.MODEL.BASIS_MODULE.NUM_CONVS
        self.visualize    = cfg.MODEL.BLENDMASK.VISUALIZE
        # fmt: on

        feature_channels = {k: v.channels for k, v in input_shape.items()}

        conv_block = conv_with_kaiming_uniform(norm, True)  # conv relu bn
        self.refine = nn.ModuleList()
        for in_feature in self.in_features:
            self.refine.append(conv_block(
                feature_channels[in_feature], planes, 3, 1))
        tower = []
        for i in range(num_convs):
            tower.append(
                conv_block(planes, planes, 3, 1))
        tower.append(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        tower.append(
            conv_block(planes, planes, 3, 1))

        tower.append(
            nn.Conv2d(planes, mask_dim, 1))
        self.add_module('tower', nn.Sequential(*tower))

        if self.loss_on:
            # fmt: off
            self.common_stride   = cfg.MODEL.BASIS_MODULE.COMMON_STRIDE
            num_classes          = cfg.MODEL.BASIS_MODULE.NUM_CLASSES + 1  # only difference
            self.sem_loss_weight = cfg.MODEL.BASIS_MODULE.LOSS_WEIGHT
            # fmt: on

            inplanes = feature_channels[self.in_features[0]]
            self.seg_head = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=3,
                                                    stride=1, padding=1, bias=False),
                                          nn.BatchNorm2d(planes),
                                          nn.ReLU(),
                                          nn.Conv2d(planes, planes, kernel_size=3,
                                                    stride=1, padding=1, bias=False),
                                          nn.BatchNorm2d(planes),
                                          nn.ReLU(),
                                          nn.Conv2d(planes, num_classes, kernel_size=1,
                                                    stride=1))

        if self.condition_loss_on:
            num_classes = cfg.MODEL.FCOS.NUM_CLASSES
            self.focal_loss_alpha = cfg.MODEL.FCOS.LOSS_ALPHA
            self.focal_loss_gamma = cfg.MODEL.FCOS.LOSS_GAMMA

            in_channels = feature_channels[self.in_features[0]]
            self.seg_head = nn.Sequential(
                conv_block(in_channels, planes, kernel_size=3, stride=1),
                conv_block(planes, planes, kernel_size=3, stride=1),
            )

            self.logits = nn.Conv2d(planes, num_classes, kernel_size=1, stride=1)

            prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            torch.nn.init.constant_(self.logits.bias, bias_value)

    def forward(self, features, targets=None, gt_instances=None):
        for i, f in enumerate(self.in_features):
            if i == 0:
                x = self.refine[i](features[f])
            else:
                x_p = self.refine[i](features[f])
                x_p = F.interpolate(x_p, x.size()[2:], mode="bilinear", align_corners=False)
                x = x + x_p
        outputs = {"bases": [self.tower(x)]}
        losses = {}
        # auxiliary thing semantic loss
        if self.training and self.loss_on:
            sem_out = self.seg_head(features[self.in_features[0]])
            # resize target to reduce memory
            gt_sem = targets.unsqueeze(1).float()
            gt_sem = F.interpolate(
                gt_sem, scale_factor=1 / self.common_stride)
            seg_loss = F.cross_entropy(
                sem_out, gt_sem.squeeze(1).long())
            losses['loss_basis_sem'] = seg_loss * self.sem_loss_weight

        if self.training and self.condition_loss_on:
            logits_pred = self.logits(self.seg_head(
                features[self.in_features[0]]
            ))

            # compute semantic targets
            semantic_targets = []
            for per_im_gt in gt_instances:
                h, w = per_im_gt.gt_bitmasks_full.size()[-2:]
                areas = per_im_gt.gt_bitmasks_full.sum(dim=-1).sum(dim=-1)
                areas = areas[:, None, None].repeat(1, h, w)
                areas[per_im_gt.gt_bitmasks_full == 0] = INF
                areas = areas.permute(1, 2, 0).reshape(h * w, -1)
                min_areas, inds = areas.min(dim=1)
                per_im_sematic_targets = per_im_gt.gt_classes[inds] + 1
                per_im_sematic_targets[min_areas == INF] = 0
                per_im_sematic_targets = per_im_sematic_targets.reshape(h, w)
                semantic_targets.append(per_im_sematic_targets)

            semantic_targets = torch.stack(semantic_targets, dim=0)

            # resize target to reduce memory
            semantic_targets = semantic_targets[
                               :, None, self.out_stride // 2::self.out_stride,
                               self.out_stride // 2::self.out_stride
                               ]

            # prepare one-hot targets
            num_classes = logits_pred.size(1)
            class_range = torch.arange(
                num_classes, dtype=logits_pred.dtype,
                device=logits_pred.device
            )[:, None, None]
            class_range = class_range + 1
            one_hot = (semantic_targets == class_range).float()
            num_pos = (one_hot > 0).sum().float().clamp(min=1.0)

            loss_sem = sigmoid_focal_loss_jit(
                logits_pred, one_hot,
                alpha=self.focal_loss_alpha,
                gamma=self.focal_loss_gamma,
                reduction="sum",
            ) / num_pos
            losses['loss_sem'] = loss_sem

        elif self.visualize and hasattr(self, "seg_head"):
            outputs["seg_thing_out"] = self.seg_head(features[self.in_features[0]])
        return outputs, losses


class Blender(object):
    def __init__(self, cfg):

        # fmt: off
        self.pooler_resolution = cfg.MODEL.BLENDMASK.BOTTOM_RESOLUTION
        sampling_ratio         = cfg.MODEL.BLENDMASK.POOLER_SAMPLING_RATIO
        pooler_type            = cfg.MODEL.BLENDMASK.POOLER_TYPE
        pooler_scales          = cfg.MODEL.BLENDMASK.POOLER_SCALES
        self.attn_size         = cfg.MODEL.BLENDMASK.ATTN_SIZE
        self.top_interp        = cfg.MODEL.BLENDMASK.TOP_INTERP
        num_bases              = cfg.MODEL.BASIS_MODULE.NUM_BASES
        # fmt: on

        self.attn_len = num_bases * self.attn_size * self.attn_size

        self.pooler = ROIPooler(
            output_size=self.pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
            canonical_level=2)

    def __call__(self, bases, proposals, gt_instances):
        if gt_instances is not None:
            # training
            # reshape attns
            dense_info = proposals["instances"]
            attns = dense_info.top_feats
            pos_inds = dense_info.pos_inds
            if pos_inds.numel() == 0:
                return None, {"loss_mask": sum([x.sum() * 0 for x in attns]) + bases[0].sum() * 0}

            gt_inds = dense_info.gt_inds

            rois = self.pooler(bases, [x.gt_boxes for x in gt_instances])
            rois = rois[gt_inds]
            pred_mask_logits = self.merge_bases(rois, attns)

            # gen targets
            gt_masks = []
            for instances_per_image in gt_instances:
                if len(instances_per_image.gt_boxes.tensor) == 0:
                    continue
                gt_mask_per_image = instances_per_image.gt_masks.crop_and_resize(
                    instances_per_image.gt_boxes.tensor, self.pooler_resolution
                ).to(device=pred_mask_logits.device)
                gt_masks.append(gt_mask_per_image)
            gt_masks = cat(gt_masks, dim=0)
            gt_masks = gt_masks[gt_inds]
            N = gt_masks.size(0)
            gt_masks = gt_masks.view(N, -1)

            gt_ctr = dense_info.gt_ctrs
            loss_denorm = proposals["loss_denorm"]
            mask_losses = F.binary_cross_entropy_with_logits(
                pred_mask_logits, gt_masks.to(dtype=torch.float32), reduction="none")
            mask_loss = ((mask_losses.mean(dim=-1) * gt_ctr).sum()
                         / loss_denorm)
            return None, {"loss_mask": mask_loss}
        else:
            # no proposals
            total_instances = sum([len(x) for x in proposals])
            if total_instances == 0:
                # add empty pred_masks results
                for box in proposals:
                    box.pred_masks = box.pred_classes.view(
                        -1, 1, self.pooler_resolution, self.pooler_resolution)
                return proposals, {}
            rois = self.pooler(bases, [x.pred_boxes for x in proposals])
            attns = cat([x.top_feat for x in proposals], dim=0)
            pred_mask_logits = self.merge_bases(rois, attns).sigmoid()
            pred_mask_logits = pred_mask_logits.view(
                -1, 1, self.pooler_resolution, self.pooler_resolution)
            start_ind = 0
            for box in proposals:
                end_ind = start_ind + len(box)
                box.pred_masks = pred_mask_logits[start_ind:end_ind]
                start_ind = end_ind
            return proposals, {}

    def merge_bases(self, rois, coeffs, location_to_inds=None):
        # merge predictions
        N = coeffs.size(0)
        if location_to_inds is not None:
            rois = rois[location_to_inds]
        N, B, H, W = rois.size()

        coeffs = coeffs.view(N, -1, self.attn_size, self.attn_size)
        coeffs = F.interpolate(coeffs, (H, W),
                               mode=self.top_interp).softmax(dim=1)
        masks_preds = (rois * coeffs).sum(dim=1)
        return masks_preds.view(N, -1)


class ModuleListDial(nn.ModuleList):
    def __init__(self, modules=None):
        super(ModuleListDial, self).__init__(modules)
        self.cur_position = 0

    def forward(self, x):
        result = self[self.cur_position](x)
        self.cur_position += 1
        if self.cur_position >= len(self):
            self.cur_position = 0
        return result


class SipMaskTrackHead(nn.Module):
    def __init__(self, cfg):
        super(SipMaskTrackHead, self).__init__()
        self.stacked_convs = cfg.MODEL.BLENDMASK.TRACKHEAD.NUM_TRACK_CONVS
        self.use_deformable = cfg.MODEL.BLENDMASK.TRACKHEAD.USE_DEFORMABLE
        self.in_channels = cfg.MODEL.BLENDMASK.TRACKHEAD.IN_CHANNELS
        self.feat_channels = cfg.MODEL.BLENDMASK.TRACKHEAD.FEAT_CHANNELS
        self.norm = None if cfg.MODEL.BLENDMASK.TRACKHEAD.NORM == 'none' else cfg.MODEL.BLENDMASK.TRACKHEAD.NORM
        self.in_features = cfg.MODEL.BLENDMASK.TRACKHEAD.IN_FEATURES
        self.track_feat_channels = cfg.MODEL.BLENDMASK.TRACKHEAD.TRACK_FEAT_CHANNELS
        self._init_layers()


    def _init_layers(self):
        tower = []
        for i in range(self.stacked_convs):
            in_channels = self.in_channels if i == 0 else self.feat_channels
            if self.use_deformable and i == self.stacked_convs - 1:
                conv_func = DFConv2d
            else:
                conv_func = nn.Conv2d
            tower.append(conv_func(in_channels=in_channels,
                                   out_channels=self.feat_channels,
                                   kernel_size=3,
                                   padding=1,
                                   stride=1,
                                   bias=self.norm is None))
            if self.norm == "GN":
                tower.append(nn.GroupNorm(32, in_channels))
            elif self.norm == "NaiveGN":
                tower.append(NaiveGroupNorm(32, in_channels))
            elif self.norm == "BN":
                tower.append(ModuleListDial([
                    nn.BatchNorm2d(in_channels) for _ in range(self.num_levels)
                ]))
            elif self.norm == "SyncBN":
                tower.append(ModuleListDial([
                    NaiveSyncBatchNorm(in_channels) for _ in range(self.num_levels)
                ]))
            tower.append(nn.ReLU())
        self.add_module('track_tower', nn.Sequential(*tower))

        self.sipmask_track = nn.Conv2d(self.feat_channels * 3, 512, 1, padding=0)

    def _train_forward(self, query_feats, reference_feats):
        count = 0
        query_track_feats = []
        reference_track_feats = []
        for query_feat, reference_feat in zip(query_feats, reference_feats):
            if count < 3:
                query_track_feat = self.track_tower(query_feat)
                query_track_feat = F.interpolate(query_track_feat, scale_factor=(2 ** count),
                                                 mode='bilinear', align_corners=False)
                query_track_feats.append(query_track_feat)
                reference_track_feat = self.track_tower(reference_feat)
                reference_track_feat = F.interpolate(reference_track_feat, scale_factor=(2 ** count),
                                                     mode='bilinear', align_corners=False)
                reference_track_feats.append(reference_track_feat)
            else:
                break
            count += 1
        query_track_feats = cat(query_track_feats, dim=1)
        query_track = self.sipmask_track(query_track_feats)
        reference_track_feats = cat(reference_track_feats, dim=1)
        reference_track = self.sipmask_track(reference_track_feats)
        return query_track, reference_track

    def _inference_forward(self, query_feats):
        count = 0
        query_track_feats = []
        for query_feat in query_feats:
            if count < 3:
                query_track_feat = self.track_tower(query_feat)
                query_track_feat = F.interpolate(query_track_feat, scale_factor=(2 ** count),
                                                 mode='bilinear', align_corners=False)
                query_track_feats.append(query_track_feat)
            else:
                break
            count += 1
        query_track_feats = cat(query_track_feats, dim=1)
        query_track = self.sipmask_track(query_track_feats)
        return query_track

    def forward(self, query_feats, reference_feats=None):
        query_feats = [query_feats[f] for f in self.in_features]
        if self.training:
            reference_feats = [reference_feats[f] for f in self.in_features]
            query_track, reference_track = self._train_forward(query_feats, reference_feats)
            return query_track, reference_track
        else:
            query_track = self._inference_forward(query_feats)
            return query_track


class BlendMaskVIS(nn.Module):
    """
    Main class for BlendMask architectures (see https://arxiv.org/abd/1901.02446).
    """

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)
        self.instance_loss_weight = cfg.MODEL.BLENDMASK.INSTANCE_LOSS_WEIGHT
        # model building
        self.backbone = cfg.build_backbone(cfg)
        self.proposal_generator = cfg.build_proposal_generator(cfg, self.backbone.output_shape())
        self.blender = build_blender(cfg)
        self.basis_module = build_basis_module(cfg, self.backbone.output_shape())
        self.temporal_align = DynamicTemporalAlign(cfg, self.backbone.output_shape())
        self.mask_out_stride = cfg.MODEL.BLENDMASK.MASK_OUT_STRIDE

        self.is_dynamic_head = True
        self.budget_loss_lambda = 0.1

        # options when combining instance & semantic outputs
        self.combine_on = cfg.MODEL.PANOPTIC_FPN.COMBINE.ENABLED
        if self.combine_on:
            self.panoptic_module = cfg.build_sem_seg_head(cfg, self.backbone.output_shape())
            self.combine_overlap_threshold = cfg.MODEL.PANOPTIC_FPN.COMBINE.OVERLAP_THRESH
            self.combine_stuff_area_limit = cfg.MODEL.PANOPTIC_FPN.COMBINE.STUFF_AREA_LIMIT
            self.combine_instances_confidence_threshold = (
                cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH)

        # build top module
        in_channels = cfg.MODEL.FPN.OUT_CHANNELS
        num_bases = cfg.MODEL.BASIS_MODULE.NUM_BASES
        attn_size = cfg.MODEL.BLENDMASK.ATTN_SIZE
        attn_len = num_bases * attn_size * attn_size
        self.top_layer = nn.Conv2d(
            in_channels, attn_len,
            kernel_size=3, stride=1, padding=1)
        torch.nn.init.normal_(self.top_layer.weight, std=0.01)
        torch.nn.init.constant_(self.top_layer.bias, 0)
        self.mask_format = cfg.INPUT.MASK_FORMAT

        self._init_track(cfg)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def _init_track(self, cfg):
        self.track_on = cfg.MODEL.TRACK_ON
        if not self.track_on:
            return
        else:
            self.track_head = build_track_head(cfg)
            self.amplitude = cfg.MODEL.BLENDMASK.TRACKHEAD.AMPLITUDE
            self.prev_roi_feats = None
            self.prev_bboxes = None
            self.prev_det_labels = None
            self.match_coef = cfg.MODEL.BLENDMASK.TRACKHEAD.MATCH_COEFF
            self.prev_fpn_feats = None

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.

        For now, each item in the list is a dict that contains:
            image: Tensor, image in (C, H, W) format.
            instances: Instances
            sem_seg: semantic segmentation ground truth.
            Other information that's included in the original dicts, such as:
                "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.

        Returns:
            list[dict]: each dict is the results for one image. The dict
                contains the following keys:
                "instances": see :meth:`GeneralizedRCNN.forward` for its format.
                "sem_seg": see :meth:`SemanticSegmentor.forward` for its format.
                "panoptic_seg": available when `PANOPTIC_FPN.COMBINE.ENABLED`.
                    See the return value of
                    :func:`combine_semantic_and_instance_outputs` for its format.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        features = self.backbone(images.tensor)
        if self.training:
            images_reference = [x["image_reference"].to(self.device) for x in batched_inputs]
            images_reference = [self.normalizer(x) for x in images_reference]
            images_reference = ImageList.from_tensors(images_reference, self.backbone.size_divisibility)
            reference_features = self.backbone(images_reference.tensor)
            features = self.temporal_align(reference_features, features)
        else:
            is_first = batched_inputs[0].get("is_first", False)
            if is_first:
                self.prev_fpn_feats = features
            else:
                features = self.temporal_align(self.prev_fpn_feats, features)
                self.prev_fpn_feats = features

        if self.combine_on:
            if "sem_seg" in batched_inputs[0]:
                gt_sem = [x["sem_seg"].to(self.device) for x in batched_inputs]
                gt_sem = ImageList.from_tensors(
                    gt_sem, self.backbone.size_divisibility, self.panoptic_module.ignore_value
                ).tensor
            else:
                gt_sem = None
            sem_seg_results, sem_seg_losses = self.panoptic_module(features, gt_sem)

        if "basis_sem" in batched_inputs[0]:
            basis_sem = [x["basis_sem"].to(self.device) for x in batched_inputs]
            basis_sem = ImageList.from_tensors(
                basis_sem, self.backbone.size_divisibility, 0).tensor
        else:
            basis_sem = None

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            self.add_bitmasks(gt_instances, images.tensor.size(-2), images.tensor.size(-1), self.mask_format)
        else:
            gt_instances = None

        basis_out, basis_losses = self.basis_module(features, basis_sem, gt_instances=gt_instances)

        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        proposals, proposal_losses = self.proposal_generator(
            images, features, gt_instances, self.top_layer)

        detector_results, detector_losses = self.blender(
            basis_out["bases"], proposals, gt_instances)

        if self.training:
            reference_gt_boxes = [x["instances_reference"].gt_boxes for x in batched_inputs]
            query_track, reference_track = self.track_head(features, reference_features)
            track_losses = self._forward_track_head_train(proposals, query_track, reference_track,
                                                          gt_instances, reference_gt_boxes)
            losses = {}
            losses.update(basis_losses)
            losses.update({k: v * self.instance_loss_weight for k, v in detector_losses.items()})
            losses.update(proposal_losses)
            if self.combine_on:
                losses.update(sem_seg_losses)
            losses.update(track_losses)

            if self.is_dynamic_head and self.budget_loss_lambda != 0:
                soft_cost, used_cost, full_cost = get_module_running_cost(self)
                loss_budget = (soft_cost / full_cost).mean() * self.budget_loss_lambda
                storage = get_event_storage()
                storage.put_scalar("complxity_ratio", (used_cost / full_cost).mean())
                losses.update({"loss_budget": loss_budget})

            return losses

        query_track = self.track_head(features)
        processed_results = []
        for i, (detector_result, input_per_image, image_size) in enumerate(zip(
                detector_results, batched_inputs, images.image_sizes)):
            detector_result = self._forward_track_heads_test(detector_result, query_track, is_first)
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            detector_r = detector_postprocess(detector_result, height, width)
            processed_result = {"instances": detector_r}
            if self.combine_on:
                sem_seg_r = sem_seg_postprocess(
                    sem_seg_results[i], image_size, height, width)
                processed_result["sem_seg"] = sem_seg_r
            if "seg_thing_out" in basis_out:
                seg_thing_r = sem_seg_postprocess(
                    basis_out["seg_thing_out"], image_size, height, width)
                processed_result["sem_thing_seg"] = seg_thing_r
            if self.basis_module.visualize:
                processed_result["bases"] = basis_out["bases"]
            processed_results.append(processed_result)

        return processed_results

    def _extract_box_feature_center_single(self, track_feats, boxes):
        track_box_feats = track_feats.new_zeros(boxes.size(0), self.track_head.track_feat_channels)

        ref_feat_stride = 8
        boxes_center_xs = torch.floor((boxes[:, 0] + boxes[:, 2]) / 2.0 / ref_feat_stride).long()
        boxes_center_ys = torch.floor((boxes[:, 1] + boxes[:, 3]) / 2.0 / ref_feat_stride).long()

        aa = track_feats.permute(1, 2, 0)
        bb = aa[boxes_center_ys, boxes_center_xs, :]
        track_box_feats += bb

        return track_box_feats

    def _forward_track_head_train(self, proposals, query_track, reference_track, gt_instances, reference_gt_boxes):
        num_images = query_track.size(0)
        pred_instances = proposals["instances"]  # level first, all images concat
        gt_pids = [x.gt_pids for x in gt_instances]

        # if 0 <= self.max_proposals < len(pred_instances):
        #     inds = torch.randperm(len(pred_instances), device=query_track.device).long()
        #     pred_instances = pred_instances[inds[:self.max_proposals]]
        locations = pred_instances.locations  # n, 2
        reg_pred = pred_instances.reg_pred  # n, 4
        fpn_levels = pred_instances.fpn_levels  # n,
        im_ids = pred_instances.im_inds  # n, fpn first
        gt_inds_relative = pred_instances.gt_inds_relative  # n,

        reg_pred = reg_pred * (2 ** (fpn_levels + 3).view(-1, 1))
        detections = torch.stack([
            locations[:, 0] - reg_pred[:, 0],
            locations[:, 1] - reg_pred[:, 1],
            locations[:, 0] + reg_pred[:, 2],
            locations[:, 1] + reg_pred[:, 3]
        ], dim=1)                   # [n, 4]

        loss_match = 0
        n_total = 0
        for i in range(num_images):
            instance_index_this_image = torch.where(im_ids == i)[0]  # which instance belong to this image
            if len(instance_index_this_image) == 0:  # no instance
                loss_match += detections[instance_index_this_image].sum() * 0
                continue
            detections_this_image = detections[instance_index_this_image]  # detection results belong to this image
            gt_pids_this_image = gt_pids[i]  # gt pids of this image
            gt_inds_relative_this_image = gt_inds_relative[instance_index_this_image]
            gt_pids_this_image = gt_pids_this_image[gt_inds_relative_this_image]

            reference_boxes_this_image = reference_gt_boxes[i].tensor
            # random_offset = reference_boxes_this_image.new_empty(reference_boxes_this_image.shape[0], 4).uniform_(
            #     -self.amplitude, self.amplitude
            # )
            # # before jittering
            # cxcy = (reference_boxes_this_image[:, 2:4] + reference_boxes_this_image[:, :2]) / 2
            # wh = (reference_boxes_this_image[:, 2:4] - reference_boxes_this_image[:, :2]).abs()
            # # after jittering
            # new_cxcy = cxcy + wh * random_offset[:, :2]
            # new_wh = wh * (1 + random_offset[:, 2:])
            # new_x1y1 = new_cxcy - new_wh / 2
            # new_x2y2 = new_cxcy + new_wh / 2
            # new_boxes = cat([new_x1y1, new_x2y2], dim=1)

            query_track_feat = self._extract_box_feature_center_single(query_track[i],
                                                                       detections_this_image)  # [n, 512]
            reference_track_feat = self._extract_box_feature_center_single(reference_track[i], reference_boxes_this_image)  # [m, 512]
            prod = torch.mm(query_track_feat, torch.transpose(reference_track_feat, 0, 1))
            n = prod.size(0)
            dummy = torch.zeros(n, 1, device=torch.cuda.current_device())
            prod_ext = cat([dummy, prod], dim=1)
            loss_match += F.cross_entropy(prod_ext, gt_pids_this_image, reduction='mean')
            n_total += len(gt_pids_this_image)

        loss_match = loss_match / num_images

        losses = {"loss_match": loss_match}

        return losses

    def _forward_track_heads_test(self, proposals, query_track, is_first):
        det_bboxes = proposals.pred_boxes.tensor
        det_labels = proposals.pred_classes
        det_scores = proposals.scores
        if det_bboxes.size(0) == 0:
            proposals.pred_obj_ids = torch.ones((det_bboxes.shape[0]), dtype=torch.int) * (-1)
            return proposals
        det_roi_feats = self._extract_box_feature_center_single(query_track[0], det_bboxes)
        if is_first or (not is_first and self.prev_bboxes is None):
            det_obj_ids = torch.arange(det_bboxes.size(0))
            self.prev_roi_feats = det_roi_feats
            self.prev_bboxes = det_bboxes
            self.prev_det_labels = det_labels
            proposals.pred_obj_ids = det_obj_ids
        else:
            assert self.prev_roi_feats is not None
            prod = torch.mm(det_roi_feats, torch.transpose(self.prev_roi_feats, 0, 1))
            n = prod.size(0)
            dummy = torch.zeros(n, 1, device=torch.cuda.current_device())
            match_score = cat([dummy, prod], dim=1)
            mat_logprob = F.log_softmax(match_score, dim=1)
            label_delta = (self.prev_det_labels == det_labels.view(-1, 1)).float()
            bbox_ious = pairwise_iou_tensor(det_bboxes, self.prev_bboxes)
            comp_scores = self.compute_comp_scores(mat_logprob,
                                                   det_scores.view(-1, 1),
                                                   bbox_ious,
                                                   label_delta,
                                                   add_bbox_dummy=True)
            match_likelihood, match_ids = torch.max(comp_scores, dim=1)
            match_ids = match_ids.cpu().numpy().astype(np.int32)
            det_obj_ids = torch.ones((match_ids.shape[0]), dtype=torch.int) * (-1)
            best_match_scores = np.ones((self.prev_bboxes.size(0))) * (-100)
            for idx, match_id in enumerate(match_ids):
                if match_id == 0:
                    det_obj_ids[idx] = self.prev_roi_feats.size(0)
                    self.prev_roi_feats = cat((self.prev_roi_feats, det_roi_feats[idx][None]), dim=0)
                    self.prev_bboxes = cat((self.prev_bboxes, det_bboxes[idx][None]), dim=0)
                    self.prev_det_labels = cat((self.prev_det_labels, det_labels[idx][None]), dim=0)
                else:
                    obj_id = match_id - 1
                    match_score = comp_scores[idx, match_id]
                    if match_score > best_match_scores[obj_id]:
                        det_obj_ids[idx] = obj_id
                        best_match_scores[obj_id] = match_score
                        self.prev_roi_feats[obj_id] = det_roi_feats[idx]
                        self.prev_bboxes[obj_id] = det_bboxes[idx]
            proposals.pred_obj_ids = det_obj_ids
        return proposals

    def compute_comp_scores(self, match_ll, bbox_scores, bbox_ious, label_delta, add_bbox_dummy):
        if add_bbox_dummy:
            dummy_iou = torch.ones(bbox_ious.size(0), 1, device=torch.cuda.current_device()) * 0
            bbox_ious = cat([dummy_iou, bbox_ious], dim=1)
            dummy_label = torch.ones(bbox_ious.size(0), 1, device=torch.cuda.current_device())
            label_delta = cat([dummy_label, label_delta], dim=1)
        if self.match_coef is None:
            return match_ll
        else:
            assert len(self.match_coef) == 3
            return match_ll + self.match_coef[0] * torch.log(bbox_scores) + \
                   self.match_coef[1] * bbox_ious + self.match_coef[2] * label_delta

    def add_bitmasks(self, instances, im_h, im_w, mask_format):
        if mask_format == 'polygon':
            for per_im_gt_inst in instances:
                if not per_im_gt_inst.has("gt_masks"):
                    continue
                polygons = per_im_gt_inst.get("gt_masks").polygons
                per_im_bitmasks = []
                per_im_bitmasks_full = []
                for per_polygons in polygons:
                    bitmask = polygons_to_bitmask(per_polygons, im_h, im_w)
                    bitmask = torch.from_numpy(bitmask).to(self.device).float()
                    start = int(self.mask_out_stride // 2)
                    bitmask_full = bitmask.clone()
                    bitmask = bitmask[start::self.mask_out_stride, start::self.mask_out_stride]

                    assert bitmask.size(0) * self.mask_out_stride == im_h
                    assert bitmask.size(1) * self.mask_out_stride == im_w

                    per_im_bitmasks.append(bitmask)
                    per_im_bitmasks_full.append(bitmask_full)

                per_im_gt_inst.gt_bitmasks = torch.stack(per_im_bitmasks, dim=0)
                per_im_gt_inst.gt_bitmasks_full = torch.stack(per_im_bitmasks_full, dim=0)
        else:
            for per_im_gt_inst in instances:
                if not per_im_gt_inst.has("gt_masks"):
                    continue
                bitmasks = per_im_gt_inst.get("gt_masks").tensor
                pad_size = bitmasks.shape
                mask_h = pad_size[-2]
                mask_w = pad_size[-1]
                pad_masks = bitmasks.new_full((pad_size[0], im_h, im_w), 0)
                pad_masks[:, :mask_h, :mask_w] = bitmasks
                pad_masks_full = pad_masks.clone()
                start = int(self.mask_out_stride // 2)
                pad_masks = pad_masks[:, start::self.mask_out_stride, start::self.mask_out_stride]
                per_im_gt_inst.gt_bitmasks = pad_masks
                per_im_gt_inst.gt_bitmasks_full = pad_masks_full


def build_track_head(cfg):

    sipmask_trackhead = SipMaskTrackHead(cfg)
    return sipmask_trackhead