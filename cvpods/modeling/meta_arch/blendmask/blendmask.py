"""
   BlendMask Using CVPODs
"""


import torch
from torch import nn
from cvpods.structures import ImageList
from cvpods.modeling.postprocessing import detector_postprocess, sem_seg_postprocess
from cvpods.modeling.meta_arch.panoptic_fpn import combine_semantic_and_instance_outputs
from cvpods.structures.masks import polygons_to_bitmask

from .blender import build_blender
from .basis_module import build_basis_module


class BlendMask(nn.Module):
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
        self.mask_out_stride = cfg.MODEL.BLENDMASK.MASK_OUT_STRIDE

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

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

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
            self.add_bitmasks(gt_instances, images.tensor.size(-2), images.tensor.size(-1))
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
            losses = {}
            losses.update(basis_losses)
            losses.update({k: v * self.instance_loss_weight for k, v in detector_losses.items()})
            losses.update(proposal_losses)
            if self.combine_on:
                losses.update(sem_seg_losses)
            return losses

        processed_results = []
        for i, (detector_result, input_per_image, image_size) in enumerate(zip(
                detector_results, batched_inputs, images.image_sizes)):
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

            if self.combine_on:
                panoptic_r = combine_semantic_and_instance_outputs(
                    detector_r,
                    sem_seg_r.argmax(dim=0),
                    self.combine_overlap_threshold,
                    self.combine_stuff_area_limit,
                    self.combine_instances_confidence_threshold)
                processed_results[-1]["panoptic_seg"] = panoptic_r
        return processed_results

    def add_bitmasks(self, instances, im_h, im_w):
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