from .ssd_detector import  SSDDetectorWithDALMultiFER

_DETECTION_META_ARCHITECTURES = {
    "SSDDetectorWithDALMultiFER": SSDDetectorWithDALMultiFER,
}


def build_detection_model(cfg):
    meta_arch = _DETECTION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)
