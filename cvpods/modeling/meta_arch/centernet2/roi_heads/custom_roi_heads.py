# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from torch import nn

from cvpods.layers import ShapeSpec


from cvpods.modeling.matcher import Matcher
from cvpods.modeling.poolers import ROIPooler
from cvpods.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs, fast_rcnn_inference


from cvpods.utils import get_event_storage

from cvpods.modeling.box_regression import Box2BoxTransform
from cvpods.modeling.roi_heads.fast_rcnn import fast_rcnn_inference
from cvpods.modeling.roi_heads.roi_heads import StandardROIHeads
from cvpods.modeling.roi_heads.cascade_rcnn import CascadeROIHeads
from .custom_fast_rcnn import CustomFastRCNNOutputLayers


class CustomROIHeads(StandardROIHeads):
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        self._init_box_head(cfg)

    @classmethod
    def _init_box_head(self, cfg):
        ret = super()._init_box_head(cfg)
        del ret['box_predictor']
        ret['box_predictor'] = CustomFastRCNNOutputLayers(
            cfg, ret['box_head'].output_shape)
        self.debug = cfg.DEBUG
        if self.debug:
            self.debug_show_name = cfg.DEBUG_SHOW_NAME
            self.save_debug = cfg.SAVE_DEBUG
            self.vis_thresh = cfg.VIS_THRESH
            self.pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(
                torch.device(cfg.MODEL.DEVICE)).view(3, 1, 1)
            self.pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(
                torch.device(cfg.MODEL.DEVICE)).view(3, 1, 1)
        return ret

    def forward(self, images, features, proposals, targets=None):
        """
        enable debug
        """
        if not self.debug:
            del images
        if self.training:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        if self.training:
            losses = self._forward_box(features, proposals)
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            if self.debug:
                from cvpods.modeling.meta_arch.centernet2.dense_heads.debug import debug_second_stage
                denormalizer = lambda x: x * self.pixel_std + self.pixel_mean
                debug_second_stage(
                    [denormalizer(images[0].clone())],
                    pred_instances, proposals=proposals,
                    debug_show_name=self.debug_show_name)
            return pred_instances, {}


class CustomCascadeROIHeads(CascadeROIHeads):
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        self._init_box_head(cfg)

    def _init_box_head(self, cfg):
        self.in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.box_in_features = self.in_features
        self.mult_proposal_score = cfg.MODEL.ROI_BOX_HEAD.MULT_PROPOSAL_SCORE
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / self.feature_strides[k] for k in self.in_features)
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        cascade_bbox_reg_weights = cfg.MODEL.ROI_BOX_CASCADE_HEAD.BBOX_REG_WEIGHTS
        cascade_ious = cfg.MODEL.ROI_BOX_CASCADE_HEAD.IOUS
        self.num_cascade_stages = len(cascade_ious)
        assert len(cascade_bbox_reg_weights) == self.num_cascade_stages
        assert cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG, \
            "CascadeROIHeads only support class-agnostic regression now!"
        assert cascade_ious[0] == cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS[0]
        # fmt: on

        in_channels = [self.feature_channels[f] for f in self.in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        self.box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        pooled_shape = ShapeSpec(
            channels=in_channels, width=pooler_resolution, height=pooler_resolution
        )

        self.box_head = nn.ModuleList()
        self.box_predictor = nn.ModuleList()
        self.box2box_transform = []
        self.proposal_matchers = []
        for k in range(self.num_cascade_stages):
            box_head = cfg.build_box_head(cfg, pooled_shape)
            self.box_head.append(box_head)
            self.box_predictor.append(
                FastRCNNOutputLayers(
                    box_head.output_size, self.num_classes, cls_agnostic_bbox_reg=True
                )
            )
            self.box2box_transform.append(Box2BoxTransform(weights=cascade_bbox_reg_weights[k]))

            if k == 0:
                # The first matching is done by the matcher of ROIHeads (self.proposal_matcher).
                self.proposal_matchers.append(None)
            else:
                self.proposal_matchers.append(
                    Matcher([cascade_ious[k]], [0, 1], allow_low_quality_matches=False)
                )

    def forward(self, images, features, proposals, targets=None,
                reference_features=None, is_first=False, reference_gt_boxes= None):
        del images
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)

        features_list = [features[f] for f in self.in_features]

        if self.training:
            # Need targets to box head
            losses = self._forward_box(features_list, proposals, targets)
            losses.update(self._forward_mask(features_list, proposals))
            losses.update(self._forward_keypoint(features_list, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features_list, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def _forward_box(self, features, proposals, targets=None):
        if (not self.training) and self.mult_proposal_score:
            if len(proposals) > 0 and proposals[0].has('scores'):
                proposal_scores = [
                    p.get('scores') for p in proposals]
            else:
                proposal_scores = [
                    p.get('objectness_logits') for p in proposals]

        head_outputs = []
        image_sizes = [x.image_size for x in proposals]
        for k in range(self.num_cascade_stages):
            if k > 0:
                # The output boxes of the previous stage are the input proposals of the next stage
                proposals = self._create_proposals_from_boxes(
                    head_outputs[-1].predict_boxes(), image_sizes
                )
                if self.training:
                    proposals = self._match_and_label_boxes(proposals, k, targets)
            head_outputs.append(self._run_stage(features, proposals, k))

        if self.training:
            losses = {}
            storage = get_event_storage()
            for stage, output in enumerate(head_outputs):
                with storage.name_scope("stage{}".format(stage)):
                    stage_losses = output.losses()
                losses.update({k + "_stage{}".format(stage): v for k, v in stage_losses.items()})
            return losses
        else:
            # Each is a list[Tensor] of length #image. Each tensor is Ri x (K+1)
            scores_per_stage = [h.predict_probs() for h in head_outputs]

            # Average the scores across heads
            scores = [
                sum(list(scores_per_image)) * (1.0 / self.num_cascade_stages)
                for scores_per_image in zip(*scores_per_stage)
            ]

            if self.mult_proposal_score:
                scores = [(s * ps[:, None]) ** 0.5 \
                    for s, ps in zip(scores, proposal_scores)]

            # Use the boxes of the last head
            boxes = head_outputs[-1].predict_boxes()
            pred_instances, _ = fast_rcnn_inference(
                boxes,
                scores,
                image_sizes,
                self.test_score_thresh,
                self.test_nms_thresh,
                self.test_nms_type,
                self.test_detections_per_img,
            )
            return pred_instances