from ssd.modeling import registry
from .vgg import VGG,  VGGSelfAttnFPNTFP
from .mobilenet import MobileNetV2
from .efficient_net import EfficientNet

__all__ = ['build_backbone', 'VGG', 'MobileNetV2', 'EfficientNet', 'VGGSelfAttnFPNTFP']


def build_backbone(cfg):
    return registry.BACKBONES[cfg.MODEL.BACKBONE.NAME](cfg, cfg.MODEL.BACKBONE.PRETRAINED)
