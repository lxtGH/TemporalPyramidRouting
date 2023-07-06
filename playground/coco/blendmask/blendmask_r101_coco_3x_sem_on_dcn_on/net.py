import logging

from cvpods.layers import ShapeSpec
from cvpods.modeling.meta_arch.blendmask.blendmask import BlendMask
from cvpods.modeling.backbone import Backbone
from cvpods.modeling.backbone.fpn import build_retinanet_resnet_fpn_backbone
from cvpods.modeling.proposal_generator.fcos import FCOS


def build_backbone(cfg, input_shape=None):
    """
    Build a backbone from `cfg.MODEL.BACKBONE.NAME`.

    Returns:
        an instance of :class:`Backbone`
    """
    if input_shape is None:
        input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))

    backbone = build_retinanet_resnet_fpn_backbone(cfg, input_shape)
    assert isinstance(backbone, Backbone)
    return backbone


def build_proposal_generator(cfg, input_shape):
    return FCOS(cfg, input_shape)


def build_model(cfg):

    cfg.build_backbone = build_backbone
    cfg.build_proposal_generator = build_proposal_generator

    model = BlendMask(cfg)
    logger = logging.getLogger(__name__)
    logger.info("Model:\n{}".format(model))
    return model
