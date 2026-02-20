"""Tests for vision module."""

import numpy as np
import pytest


class TestTTA:
    """Tests for Test Time Augmentation."""

    def test_tta_classifier(self):
        """Test TTA for classification."""
        pytest.importorskip("torch")
        from endgame.vision import TestTimeAugmentation

        # Create a simple mock model
        class MockModel:
            def __call__(self, x):
                import torch
                return torch.rand(x.shape[0], 10)

        model = MockModel()
        tta = TestTimeAugmentation(
            augmentations=["hflip", "vflip"],
            merge_mode="mean",
        )

        # Create dummy input as numpy (predict converts to torch internally)
        dummy_input = np.random.rand(4, 3, 224, 224).astype(np.float32)
        output = tta.predict(model, dummy_input)

        assert output.shape == (4, 10)

    def test_tta_augmentations(self):
        """Test different TTA augmentations."""
        pytest.importorskip("torch")
        from endgame.vision.tta import TestTimeAugmentation

        tta = TestTimeAugmentation(
            augmentations=["hflip", "vflip", "rotate90"],
        )

        # Should store the specified augmentations
        assert len(tta.augmentations) == 3
        assert "hflip" in tta.augmentations
        assert "vflip" in tta.augmentations
        assert "rotate90" in tta.augmentations


class TestWBF:
    """Tests for Weighted Boxes Fusion."""

    def test_wbf_single_model(self):
        """Test WBF with single model predictions."""
        from endgame.vision.wbf import WeightedBoxesFusion

        wbf = WeightedBoxesFusion(iou_threshold=0.5, skip_box_threshold=0.01)

        # Single model predictions
        boxes_list = [
            np.array([[0.1, 0.1, 0.5, 0.5], [0.6, 0.6, 0.9, 0.9]])
        ]
        scores_list = [np.array([0.9, 0.8])]
        labels_list = [np.array([0, 1])]

        boxes, scores, labels = wbf.fuse(
            boxes_list, scores_list, labels_list,
        )

        assert len(boxes) == 2
        assert len(scores) == 2
        assert len(labels) == 2

    def test_wbf_multiple_models(self):
        """Test WBF merging multiple model predictions."""
        from endgame.vision.wbf import WeightedBoxesFusion

        wbf = WeightedBoxesFusion(iou_threshold=0.5, skip_box_threshold=0.01)

        # Two models with overlapping boxes
        boxes_list = [
            np.array([[0.1, 0.1, 0.5, 0.5]]),
            np.array([[0.12, 0.12, 0.52, 0.52]]),  # Overlapping
        ]
        scores_list = [np.array([0.9]), np.array([0.85])]
        labels_list = [np.array([0]), np.array([0])]

        boxes, scores, labels = wbf.fuse(
            boxes_list, scores_list, labels_list,
        )

        # Should merge into single box
        assert len(boxes) == 1
        assert scores[0] > 0.85  # Fused score

    def test_wbf_with_weights(self):
        """Test WBF with model weights."""
        from endgame.vision.wbf import WeightedBoxesFusion

        wbf = WeightedBoxesFusion(
            iou_threshold=0.5, weights=[2.0, 1.0],
        )

        boxes_list = [
            np.array([[0.1, 0.1, 0.5, 0.5]]),
            np.array([[0.1, 0.1, 0.5, 0.5]]),
        ]
        scores_list = [np.array([0.9]), np.array([0.6])]
        labels_list = [np.array([0]), np.array([0])]

        boxes, scores, labels = wbf.fuse(
            boxes_list, scores_list, labels_list,
        )

        assert len(boxes) == 1

    def test_soft_nms(self):
        """Test Soft-NMS implementation."""
        from endgame.vision.wbf import soft_nms

        boxes = np.array([
            [0.1, 0.1, 0.5, 0.5],
            [0.15, 0.15, 0.55, 0.55],  # Overlapping
            [0.8, 0.8, 0.95, 0.95],    # Separate
        ])
        scores = np.array([0.9, 0.85, 0.7])

        kept_boxes, kept_scores = soft_nms(
            boxes, scores, iou_threshold=0.3, sigma=0.5,
        )

        # Should return some boxes
        assert len(kept_boxes) > 0
        assert len(kept_scores) > 0
        assert len(kept_boxes) == len(kept_scores)


class TestBackbone:
    """Tests for backbone utilities."""

    def test_get_backbone(self):
        """Test backbone loading."""
        torch = pytest.importorskip("torch")
        timm = pytest.importorskip("timm")
        from endgame.vision.backbone import VisionBackbone

        backbone = VisionBackbone(
            architecture="resnet18",
            pretrained=False,
        )

        model = backbone.get_model()
        assert model is not None

    def test_backbone_feature_extraction(self):
        """Test feature extraction from backbone."""
        torch = pytest.importorskip("torch")
        timm = pytest.importorskip("timm")
        from endgame.vision.backbone import VisionBackbone

        backbone = VisionBackbone(
            architecture="resnet18",
            pretrained=False,
        )

        dummy_input = np.random.rand(1, 3, 224, 224).astype(np.float32)
        features = backbone.extract_features(dummy_input)

        # Should return feature vectors
        assert isinstance(features, np.ndarray)
        assert features.ndim == 2
        assert features.shape[0] == 1
