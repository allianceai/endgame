from __future__ import annotations

"""Test Time Augmentation for vision models."""

from collections.abc import Callable
from typing import Any, Literal

import numpy as np

from endgame.core.base import EndgameEstimator


class TestTimeAugmentation(EndgameEstimator):
    """Test Time Augmentation: Average predictions across augmented images.

    Reduces prediction variance without retraining by applying
    augmentations at inference time and averaging results.

    Parameters
    ----------
    augmentations : List[str], default=['identity', 'hflip']
        Augmentations to apply:
        - 'identity': Original image
        - 'hflip': Horizontal flip
        - 'vflip': Vertical flip
        - 'rotate90', 'rotate180', 'rotate270'
        - 'transpose': Transpose (swap axes)
    merge_mode : str, default='mean'
        How to merge predictions: 'mean', 'max', 'gmean'.
    use_deaugment : bool, default=True
        Whether to reverse augmentations on outputs (for segmentation).

    Examples
    --------
    >>> tta = TestTimeAugmentation(
    ...     augmentations=['identity', 'hflip', 'vflip', 'rotate90'],
    ...     merge_mode='mean'
    ... )
    >>> predictions = tta.predict(model, images)
    """

    def __init__(
        self,
        augmentations: list[str] = None,
        merge_mode: Literal["mean", "max", "gmean"] = "mean",
        use_deaugment: bool = True,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(random_state=random_state, verbose=verbose)
        self.augmentations = augmentations or ["identity", "hflip"]
        self.merge_mode = merge_mode
        self.use_deaugment = use_deaugment

    def _apply_augmentation(
        self,
        image: np.ndarray,
        aug_name: str,
    ) -> np.ndarray:
        """Apply a single augmentation to an image.

        Parameters
        ----------
        image : ndarray
            Image array of shape (H, W, C) or (C, H, W).
        aug_name : str
            Augmentation name.

        Returns
        -------
        ndarray
            Augmented image.
        """
        if aug_name == "identity":
            return image

        # Determine if channels first or last
        if image.shape[0] <= 4:  # Likely channels first
            axis = (1, 2)
        else:
            axis = (0, 1)

        if aug_name == "hflip":
            return np.flip(image, axis=axis[-1])

        if aug_name == "vflip":
            return np.flip(image, axis=axis[0])

        if aug_name == "rotate90":
            return np.rot90(image, k=1, axes=axis)

        if aug_name == "rotate180":
            return np.rot90(image, k=2, axes=axis)

        if aug_name == "rotate270":
            return np.rot90(image, k=3, axes=axis)

        if aug_name == "transpose":
            if image.shape[0] <= 4:
                return np.transpose(image, (0, 2, 1))
            return np.transpose(image, (1, 0, 2))

        raise ValueError(f"Unknown augmentation: {aug_name}")

    def _deaugment_prediction(
        self,
        prediction: np.ndarray,
        aug_name: str,
    ) -> np.ndarray:
        """Reverse augmentation on prediction (for segmentation masks).

        Parameters
        ----------
        prediction : ndarray
            Prediction array.
        aug_name : str
            Augmentation name to reverse.

        Returns
        -------
        ndarray
            De-augmented prediction.
        """
        if aug_name == "identity":
            return prediction

        # Reverse augmentations
        if aug_name == "hflip":
            return np.flip(prediction, axis=-1)

        if aug_name == "vflip":
            return np.flip(prediction, axis=-2)

        if aug_name == "rotate90":
            return np.rot90(prediction, k=-1, axes=(-2, -1))

        if aug_name == "rotate180":
            return np.rot90(prediction, k=-2, axes=(-2, -1))

        if aug_name == "rotate270":
            return np.rot90(prediction, k=-3, axes=(-2, -1))

        if aug_name == "transpose":
            return np.transpose(prediction, tuple(range(prediction.ndim - 2)) + (-1, -2))

        return prediction

    def predict(
        self,
        model: Any,
        images: np.ndarray,
        predict_fn: Callable | None = None,
    ) -> np.ndarray:
        """Apply TTA and return averaged predictions.

        Parameters
        ----------
        model : Any
            Model with predict method.
        images : ndarray
            Input images of shape (N, C, H, W) or (N, H, W, C).
        predict_fn : Callable, optional
            Custom prediction function. If None, uses model.predict().

        Returns
        -------
        ndarray
            TTA-averaged predictions.
        """
        if predict_fn is None:
            def predict_fn(x):
                return model(x) if callable(model) else model.predict(x)

        all_predictions = []

        for aug_name in self.augmentations:
            self._log(f"Applying augmentation: {aug_name}")

            # Augment all images
            augmented = np.array([
                self._apply_augmentation(img, aug_name)
                for img in images
            ])

            # Predict
            try:
                import torch
                if isinstance(augmented, np.ndarray):
                    augmented = torch.tensor(augmented, dtype=torch.float32)
                with torch.no_grad():
                    preds = predict_fn(augmented)
                    if isinstance(preds, torch.Tensor):
                        preds = preds.cpu().numpy()
            except ImportError:
                preds = predict_fn(augmented)

            # De-augment if needed (for segmentation)
            if self.use_deaugment and preds.ndim > 2:
                preds = np.array([
                    self._deaugment_prediction(p, aug_name)
                    for p in preds
                ])

            all_predictions.append(preds)

        # Merge predictions
        all_predictions = np.array(all_predictions)

        if self.merge_mode == "mean":
            result = np.mean(all_predictions, axis=0)
        elif self.merge_mode == "max":
            result = np.max(all_predictions, axis=0)
        elif self.merge_mode == "gmean":
            # Geometric mean (for probabilities)
            result = np.exp(np.mean(np.log(all_predictions + 1e-10), axis=0))
        else:
            raise ValueError(f"Unknown merge mode: {self.merge_mode}")

        return result

    def predict_proba(
        self,
        model: Any,
        images: np.ndarray,
    ) -> np.ndarray:
        """Alias for predict() for classification tasks."""
        return self.predict(model, images)
