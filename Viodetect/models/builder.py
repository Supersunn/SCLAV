from fvcore.common.registry import Registry
from Viodetect.utils.build import build_from_cfg

MODELS = Registry("model")
HEADS = MODELS
NECKS = MODELS
LOSSES = MODELS


def build_model(cfg):
    return build_from_cfg(cfg, MODELS)


def build_neck(cfg):
    neck = build_from_cfg(cfg, NECKS)
    neck.init_weights()
    return neck


def build_head(cfg):
    head = build_from_cfg(cfg, HEADS)
    head.init_weights()
    return head


def build_loss(cfg):
    return build_from_cfg(cfg, LOSSES)
