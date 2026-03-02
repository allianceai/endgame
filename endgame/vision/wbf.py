from __future__ import annotations

"""Weighted Boxes Fusion and NMS variants for object detection."""

from typing import Literal

import numpy as np


def box_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute IoU between two boxes.

    Parameters
    ----------
    box1, box2 : ndarray
        Boxes in format [x1, y1, x2, y2].

    Returns
    -------
    float
        Intersection over Union.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - inter

    if union <= 0:
        return 0.0

    return inter / union


class WeightedBoxesFusion:
    """Weighted Boxes Fusion for object detection ensembling.

    Unlike NMS which discards overlapping boxes, WBF averages
    overlapping boxes weighted by confidence scores.

    Fused coordinates: Σ(confidence_i × coord_i) / Σ(confidence_i)

    Parameters
    ----------
    iou_threshold : float, default=0.5
        IoU threshold for box clustering.
    skip_box_threshold : float, default=0.0
        Minimum confidence to consider a box.
    conf_type : str, default='avg'
        Confidence fusion: 'avg', 'max', 'box_and_model_avg'.
    weights : List[float], optional
        Model weights (for multi-model fusion).

    Examples
    --------
    >>> wbf = WeightedBoxesFusion(iou_threshold=0.5)
    >>> fused_boxes, fused_scores, fused_labels = wbf.fuse(
    ...     boxes_list, scores_list, labels_list
    ... )
    """

    def __init__(
        self,
        iou_threshold: float = 0.5,
        skip_box_threshold: float = 0.0,
        conf_type: Literal["avg", "max", "box_and_model_avg"] = "avg",
        weights: list[float] | None = None,
    ):
        self.iou_threshold = iou_threshold
        self.skip_box_threshold = skip_box_threshold
        self.conf_type = conf_type
        self.weights = weights

    def fuse(
        self,
        boxes_list: list[np.ndarray],
        scores_list: list[np.ndarray],
        labels_list: list[np.ndarray],
        image_size: tuple[int, int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fuse boxes from multiple models using WBF.

        Parameters
        ----------
        boxes_list : List[ndarray]
            Boxes from each model. Each array has shape (N, 4) with
            format [x1, y1, x2, y2] in normalized (0-1) or absolute coords.
        scores_list : List[ndarray]
            Confidence scores from each model. Shape (N,).
        labels_list : List[ndarray]
            Class labels from each model. Shape (N,).
        image_size : Tuple[int, int], optional
            Image size (H, W) for normalization.

        Returns
        -------
        fused_boxes : ndarray
            Fused boxes of shape (M, 4).
        fused_scores : ndarray
            Fused confidence scores of shape (M,).
        fused_labels : ndarray
            Fused class labels of shape (M,).
        """
        n_models = len(boxes_list)

        # Apply model weights
        weights = self.weights or [1.0] * n_models

        # Collect all boxes
        all_boxes = []
        all_scores = []
        all_labels = []
        all_model_indices = []

        for model_idx, (boxes, scores, labels) in enumerate(
            zip(boxes_list, scores_list, labels_list)
        ):
            for i in range(len(boxes)):
                if scores[i] >= self.skip_box_threshold:
                    all_boxes.append(boxes[i])
                    all_scores.append(scores[i] * weights[model_idx])
                    all_labels.append(labels[i])
                    all_model_indices.append(model_idx)

        if len(all_boxes) == 0:
            return np.array([]), np.array([]), np.array([])

        all_boxes = np.array(all_boxes)
        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)
        all_model_indices = np.array(all_model_indices)

        # Process each class separately
        unique_labels = np.unique(all_labels)
        fused_boxes = []
        fused_scores = []
        fused_labels = []

        for label in unique_labels:
            mask = all_labels == label
            label_boxes = all_boxes[mask]
            label_scores = all_scores[mask]
            label_model_indices = all_model_indices[mask]

            # Sort by score
            order = np.argsort(-label_scores)
            label_boxes = label_boxes[order]
            label_scores = label_scores[order]
            label_model_indices = label_model_indices[order]

            # Cluster boxes
            used = np.zeros(len(label_boxes), dtype=bool)
            clusters = []

            for i in range(len(label_boxes)):
                if used[i]:
                    continue

                cluster = [(i, label_scores[i], label_boxes[i], label_model_indices[i])]
                used[i] = True

                for j in range(i + 1, len(label_boxes)):
                    if used[j]:
                        continue

                    iou = box_iou(label_boxes[i], label_boxes[j])
                    if iou >= self.iou_threshold:
                        cluster.append((j, label_scores[j], label_boxes[j], label_model_indices[j]))
                        used[j] = True

                clusters.append(cluster)

            # Fuse each cluster
            for cluster in clusters:
                if len(cluster) == 1:
                    _, score, box, _ = cluster[0]
                    fused_boxes.append(box)
                    fused_scores.append(score)
                    fused_labels.append(label)
                else:
                    # Weighted average of coordinates
                    weights_sum = sum(item[1] for item in cluster)
                    fused_box = np.zeros(4)
                    for _, score, box, _ in cluster:
                        fused_box += score * box
                    fused_box /= weights_sum

                    # Fuse confidence
                    if self.conf_type == "avg":
                        fused_score = np.mean([item[1] for item in cluster])
                    elif self.conf_type == "max":
                        fused_score = np.max([item[1] for item in cluster])
                    elif self.conf_type == "box_and_model_avg":
                        # Average per-model, then average models
                        model_scores = {}
                        for _, score, _, model_idx in cluster:
                            if model_idx not in model_scores:
                                model_scores[model_idx] = []
                            model_scores[model_idx].append(score)
                        fused_score = np.mean([
                            np.mean(scores) for scores in model_scores.values()
                        ])
                    else:
                        fused_score = np.mean([item[1] for item in cluster])

                    # Boost score based on number of models
                    n_models_in_cluster = len(set(item[3] for item in cluster))
                    fused_score = min(fused_score * (1 + 0.1 * (n_models_in_cluster - 1)), 1.0)

                    fused_boxes.append(fused_box)
                    fused_scores.append(fused_score)
                    fused_labels.append(label)

        if len(fused_boxes) == 0:
            return np.array([]), np.array([]), np.array([])

        fused_boxes = np.array(fused_boxes)
        fused_scores = np.array(fused_scores)
        fused_labels = np.array(fused_labels)

        # Sort by score
        order = np.argsort(-fused_scores)
        return fused_boxes[order], fused_scores[order], fused_labels[order]


def soft_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.5,
    sigma: float = 0.5,
    score_threshold: float = 0.001,
    method: Literal["linear", "gaussian"] = "gaussian",
) -> tuple[np.ndarray, np.ndarray]:
    """Soft-NMS: Decays confidence of overlapping boxes instead of removing.

    Parameters
    ----------
    boxes : ndarray
        Boxes of shape (N, 4) in format [x1, y1, x2, y2].
    scores : ndarray
        Confidence scores of shape (N,).
    iou_threshold : float, default=0.5
        IoU threshold for decay.
    sigma : float, default=0.5
        Gaussian decay sigma.
    score_threshold : float, default=0.001
        Minimum score to keep a box.
    method : str, default='gaussian'
        Decay method: 'linear' or 'gaussian'.

    Returns
    -------
    kept_boxes : ndarray
        Boxes after soft-NMS.
    kept_scores : ndarray
        Decayed scores.
    """
    boxes = boxes.copy()
    scores = scores.copy()

    kept_boxes = []
    kept_scores = []

    while len(scores) > 0:
        # Find max score
        max_idx = np.argmax(scores)
        max_box = boxes[max_idx].copy()
        max_score = scores[max_idx]

        if max_score < score_threshold:
            break

        kept_boxes.append(max_box)
        kept_scores.append(max_score)

        # Remove current box before decay
        mask = np.ones(len(boxes), dtype=bool)
        mask[max_idx] = False
        boxes = boxes[mask]
        scores = scores[mask]

        if len(scores) == 0:
            break

        # Compute IoU with remaining boxes
        ious = np.array([box_iou(max_box, box) for box in boxes])

        # Decay scores
        if method == "gaussian":
            decay = np.exp(-(ious ** 2) / sigma)
        else:  # linear
            decay = np.where(ious > iou_threshold, 1 - ious, 1.0)

        scores = scores * decay

        # Remove boxes below threshold
        keep = scores >= score_threshold
        boxes = boxes[keep]
        scores = scores[keep]

    if len(kept_boxes) == 0:
        return np.array([]).reshape(0, 4), np.array([])

    return np.array(kept_boxes), np.array(kept_scores)
