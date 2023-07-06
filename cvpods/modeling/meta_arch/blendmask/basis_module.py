from typing import Dict
import math

import torch
from torch import nn
from torch.nn import functional as F

from cvpods.layers import ShapeSpec
from cvpods.layers.conv_with_kaiming_uniform import conv_with_kaiming_uniform
from cvpods.modeling.losses import sigmoid_focal_loss_jit
from cvpods.modeling.meta_arch.boundary_mask_rcnn.boundary_mask_rcnn import boundary_loss_func

INF = 100000000

"""
Registry for basis module, which produces global bases from feature maps.

The registered object will be called with `obj(cfg, input_shape)`.
The call should return a `nn.Module` object.
"""


def build_basis_module(cfg, input_shape):
    return ProtoNet(cfg, input_shape)

def build_boundary_bases_module(cfg, input_shape):
    return ProtoNetWithBoundary(cfg, input_shape)


class ProtoNetWithBoundary(nn.Module):
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        """
        TODO: support deconv and variable channel width
        """
        # official protonet has a relu after each conv
        super().__init__()
        # fmt: off
        mask_dim          = cfg.MODEL.BASIS_MODULE.NUM_BASES + 1        # +1 for boundary basis
        planes            = cfg.MODEL.BASIS_MODULE.CONVS_DIM            # 128
        self.in_features  = cfg.MODEL.BASIS_MODULE.IN_FEATURES          # p3 p4 p5
        self.loss_on      = cfg.MODEL.BASIS_MODULE.LOSS_ON              # False
        # seg loss by condtional inst
        self.condition_loss_on = cfg.MODEL.BASIS_MODULE.CONDTION_INST_LOSS_ON       # True
        self.out_stride = input_shape[self.in_features[0]].stride       # 8

        norm              = cfg.MODEL.BASIS_MODULE.NORM
        num_convs         = cfg.MODEL.BASIS_MODULE.NUM_CONVS            # 3
        self.visualize    = cfg.MODEL.BLENDMASK.VISUALIZE
        self.boundary_loss_weight = cfg.MODEL.BASIS_MODULE.BOUNDARY_LOSS_WEIGHT
        # fmt: on

        feature_channels = {k: v.channels for k, v in input_shape.items()}

        conv_block = conv_with_kaiming_uniform(norm, True)  # conv relu bn
        self.refine = nn.ModuleList()
        for in_feature in self.in_features:
            self.refine.append(conv_block(
                feature_channels[in_feature], planes, 3, 1))            # 3x3 conv refine for p3 p4 p5
        tower = []
        for i in range(num_convs):
            tower.append(
                conv_block(planes, planes, 3, 1))
        tower.append(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))      # stride8->4
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

            in_channels = feature_channels[self.in_features[0]]                 # p3
            self.seg_head = nn.Sequential(
                conv_block(in_channels, planes, kernel_size=3, stride=1),
                conv_block(planes, planes, kernel_size=3, stride=1)
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
                # x_p = aligned_bilinear(x_p, x.size(3) // x_p.size(3))
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
            ))                 # N C H W stride 8

            # compute semantic targets
            semantic_targets = []
            for per_im_gt in gt_instances:          # Loop for each image
                h, w = per_im_gt.gt_bitmasks_full.size()[-2:]
                areas = per_im_gt.gt_bitmasks_full.sum(dim=-1).sum(dim=-1)      # [num_instance_this_image, ]
                areas = areas[:, None, None].repeat(1, h, w)                    # [num_instance_this_image, h, w]
                areas[per_im_gt.gt_bitmasks_full == 0] = INF
                areas = areas.permute(1, 2, 0).reshape(h * w, -1)               # [h, w, num_instance_this_image] -> [h*w, num_instances_this_image]
                min_areas, inds = areas.min(dim=1)                              # [h*w, ]
                per_im_sematic_targets = per_im_gt.gt_classes[inds] + 1         # 1-based 0 for baskground
                per_im_sematic_targets[min_areas == INF] = 0
                per_im_sematic_targets = per_im_sematic_targets.reshape(h, w)   # [h, w]
                semantic_targets.append(per_im_sematic_targets)

            semantic_targets = torch.stack(semantic_targets, dim=0)             # [num_image, h, w]

            # resize target to reduce memory
            semantic_targets = semantic_targets[
                               :, None, self.out_stride // 2::self.out_stride,
                               self.out_stride // 2::self.out_stride
                               ]                                                # [num_image, 1, h, w] stride8 size

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

        if self.training:
            # boundary loss
            instances_masks = []
            boundary_pred = outputs["bases"][0][:, -1, :, :]                           # num_images, h, w stride4 size
            for gt_instances_per_image in gt_instances:
                if len(gt_instances_per_image) == 0:
                    continue
                gt_bit_masks = gt_instances_per_image.gt_bitmasks                   # num_instances, h, w stride4 size
                instances_mask = gt_bit_masks.sum(dim=0)                            # h, w
                instances_mask = (instances_mask >= 1)                              # h, w (0 or 1)
                instances_masks.append(instances_mask)
            if len(instances_masks) == 0:
                losses["loss_boundary_basis"] = boundary_pred.sum() * 0
            else:
                instances_masks = torch.stack(instances_masks, dim=0).float()           # num_image, h, w     stride4 size
                losses["loss_boundary_basis"] = boundary_loss_func(boundary_pred, instances_masks) * self.boundary_loss_weight

        return outputs, losses


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
                conv_block(planes, planes, kernel_size=3, stride=1)
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
                # x_p = aligned_bilinear(x_p, x.size(3) // x_p.size(3))
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
