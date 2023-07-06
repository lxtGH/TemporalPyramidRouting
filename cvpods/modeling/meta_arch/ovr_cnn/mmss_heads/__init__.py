import torch
from torch import nn

from .grounding_head import GroundingHead
from .transformer_head import TransformerHead


def build_grounding_head(cfg, v_dim, l_dim, *args, **kwargs):
    model = GroundingHead(cfg, v_dim, l_dim)
    return model


def build_transformer_head(cfg, v_dim, l_dim, loc_dim, backbone, *args, **kwargs):
    model = TransformerHead(cfg, v_dim, l_dim, loc_dim, backbone)
    return model


def build_mmss_heads(cfg, *args, **kwargs):
    heads = {}
    for head_type in cfg.MODEL.MMSS_HEAD.TYPES:
        if head_type == "TransformerHead":
            heads[head_type] = build_transformer_head(cfg, *args, **kwargs)
        elif head_type == "GroundingHead":
            heads[head_type] = build_grounding_head(cfg, *args, **kwargs)
        else:
            raise NotImplementedError("No such head")

    if cfg.MODEL.MMSS_HEAD.TIE_VL_PROJECTION_WEIGHTS:
        weight = heads[cfg.MODEL.MMSS_HEAD.DEFAULT_HEAD].v2l_projection.weight
        bias = heads[cfg.MODEL.MMSS_HEAD.DEFAULT_HEAD].v2l_projection.bias
        for head_type in cfg.MODEL.MMSS_HEAD.TYPES:
            if head_type == cfg.MODEL.MMSS_HEAD.DEFAULT_HEAD:
                continue
            if not hasattr(heads[head_type], 'v2l_projection'):
                continue
            assert weight.shape[0] == heads[head_type].v2l_projection.weight.shape[0]
            assert weight.shape[1] == heads[head_type].v2l_projection.weight.shape[1]
            heads[head_type].v2l_projection.weight = weight
            heads[head_type].v2l_projection.bias = bias
            
    return nn.ModuleDict(heads)
