from collections import OrderedDict

from torch import nn
from . import transformers
from . import word_embedding


def build_bert_backbone(cfg):
    body = transformers.BERT(cfg)
    model = nn.Sequential(OrderedDict([("body", body)]))
    model.out_channels = body.out_channels
    return model


def build_embedding_backbone(cfg):
    body = word_embedding.WordEmbedding(cfg)
    model = nn.Sequential(OrderedDict([("body", body)]))
    model.out_channels = body.out_channels
    return model