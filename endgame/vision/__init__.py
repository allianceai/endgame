from __future__ import annotations

"""Vision module: CV backbones, TTA, WBF, augmentation, segmentation."""

from endgame.vision.augmentation import AugmentationPipeline
from endgame.vision.backbone import VisionBackbone
from endgame.vision.segmentation import SegmentationModel
from endgame.vision.tta import TestTimeAugmentation
from endgame.vision.wbf import WeightedBoxesFusion, soft_nms

__all__ = [
    "VisionBackbone",
    "TestTimeAugmentation",
    "WeightedBoxesFusion",
    "soft_nms",
    "AugmentationPipeline",
    "SegmentationModel",
]
