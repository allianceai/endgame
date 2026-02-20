"""Vision augmentation pipelines.

Provides preset-based augmentation pipelines using albumentations,
with built-in MixUp and CutMix support.
"""

from typing import Any

import numpy as np

# Albumentations import (lazy loaded)
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False


def _check_albumentations():
    """Check if albumentations is available."""
    if not HAS_ALBUMENTATIONS:
        raise ImportError(
            "albumentations is required for AugmentationPipeline. "
            "Install with: pip install albumentations"
        )


class AugmentationPipeline:
    """Albumentations-based augmentation pipeline with presets.

    Provides ready-to-use augmentation configurations optimized for
    different competition scenarios.

    Parameters
    ----------
    preset : str, default='standard'
        Augmentation preset:
        - 'light': Basic geometric (flip, rotate)
        - 'standard': + color jitter, blur
        - 'heavy': + GridDropout, CoarseDropout
        - 'medical': + stain-like augmentations
        - 'satellite': + MultiScaleDegrade, elastic
        - 'custom': User-defined (use custom_transforms)
    image_size : int or Tuple[int, int], default=224
        Target image size (height, width).
    normalize : bool, default=True
        Whether to normalize images (ImageNet stats by default).
    mean : Tuple[float, ...], optional
        Normalization mean (default: ImageNet).
    std : Tuple[float, ...], optional
        Normalization std (default: ImageNet).
    to_tensor : bool, default=True
        Whether to convert to PyTorch tensor.
    custom_transforms : List, optional
        Custom albumentations transforms (for preset='custom').
    p_augment : float, default=1.0
        Probability of applying augmentation pipeline.

    Attributes
    ----------
    train_transform : A.Compose
        Training augmentation pipeline.
    val_transform : A.Compose
        Validation augmentation pipeline (resize + normalize only).
    test_transform : A.Compose
        Test augmentation pipeline (same as val).

    Examples
    --------
    >>> from endgame.vision import AugmentationPipeline
    >>> # Standard augmentation
    >>> aug = AugmentationPipeline(preset='standard', image_size=384)
    >>> # Apply to image
    >>> augmented = aug.train_transform(image=image)['image']
    >>> # For validation
    >>> validated = aug.val_transform(image=image)['image']
    >>> # Heavy augmentation with MixUp
    >>> aug = AugmentationPipeline(preset='heavy', image_size=224)
    >>> img1_aug, img2_aug, lam = aug.mixup(img1, img2, alpha=0.4)
    """

    # ImageNet normalization
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        preset: str = "standard",
        image_size: int | tuple[int, int] = 224,
        normalize: bool = True,
        mean: tuple[float, ...] | None = None,
        std: tuple[float, ...] | None = None,
        to_tensor: bool = True,
        custom_transforms: list | None = None,
        p_augment: float = 1.0,
    ):
        _check_albumentations()

        self.preset = preset
        self.image_size = (image_size, image_size) if isinstance(image_size, int) else image_size
        self.normalize = normalize
        self.mean = mean or self.IMAGENET_MEAN
        self.std = std or self.IMAGENET_STD
        self.to_tensor = to_tensor
        self.custom_transforms = custom_transforms
        self.p_augment = p_augment

        # Build transforms
        self.train_transform = self._build_train_transform()
        self.val_transform = self._build_val_transform()
        self.test_transform = self.val_transform

    def _build_train_transform(self) -> A.Compose:
        """Build training augmentation pipeline."""
        transforms = []

        # Resize
        transforms.append(A.Resize(self.image_size[0], self.image_size[1]))

        # Add preset-specific augmentations
        if self.preset == "light":
            transforms.extend(self._get_light_augmentations())
        elif self.preset == "standard":
            transforms.extend(self._get_standard_augmentations())
        elif self.preset == "heavy":
            transforms.extend(self._get_heavy_augmentations())
        elif self.preset == "medical":
            transforms.extend(self._get_medical_augmentations())
        elif self.preset == "satellite":
            transforms.extend(self._get_satellite_augmentations())
        elif self.preset == "custom":
            if self.custom_transforms:
                transforms.extend(self.custom_transforms)
        else:
            raise ValueError(f"Unknown preset: {self.preset}")

        # Normalize
        if self.normalize:
            transforms.append(A.Normalize(mean=self.mean, std=self.std))

        # To tensor
        if self.to_tensor:
            transforms.append(ToTensorV2())

        return A.Compose(transforms, p=self.p_augment)

    def _build_val_transform(self) -> A.Compose:
        """Build validation/test augmentation pipeline."""
        transforms = [
            A.Resize(self.image_size[0], self.image_size[1]),
        ]

        if self.normalize:
            transforms.append(A.Normalize(mean=self.mean, std=self.std))

        if self.to_tensor:
            transforms.append(ToTensorV2())

        return A.Compose(transforms)

    def _get_light_augmentations(self) -> list:
        """Light augmentations: basic geometric."""
        return [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=15,
                p=0.5,
            ),
        ]

    def _get_standard_augmentations(self) -> list:
        """Standard augmentations: geometric + color."""
        return [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.15,
                rotate_limit=30,
                border_mode=0,
                p=0.5,
            ),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0)),
                A.GaussianBlur(blur_limit=(3, 7)),
                A.MotionBlur(blur_limit=7),
            ], p=0.3),
            A.OneOf([
                A.OpticalDistortion(distort_limit=0.05),
                A.GridDistortion(num_steps=5, distort_limit=0.05),
            ], p=0.2),
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=0.5,
            ),
            A.OneOf([
                A.CLAHE(clip_limit=4.0),
                A.Equalize(),
            ], p=0.2),
        ]

    def _get_heavy_augmentations(self) -> list:
        """Heavy augmentations: aggressive + dropout."""
        return [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.15,
                scale_limit=0.2,
                rotate_limit=45,
                border_mode=0,
                p=0.7,
            ),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 100.0)),
                A.GaussianBlur(blur_limit=(3, 11)),
                A.MotionBlur(blur_limit=11),
                A.MedianBlur(blur_limit=7),
            ], p=0.4),
            A.OneOf([
                A.OpticalDistortion(distort_limit=0.1),
                A.GridDistortion(num_steps=5, distort_limit=0.1),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50),
            ], p=0.3),
            A.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.15,
                p=0.6,
            ),
            A.OneOf([
                A.CLAHE(clip_limit=4.0),
                A.Equalize(),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
            ], p=0.3),
            A.OneOf([
                A.CoarseDropout(
                    max_holes=8,
                    max_height=self.image_size[0] // 8,
                    max_width=self.image_size[1] // 8,
                    fill_value=0,
                ),
                A.GridDropout(ratio=0.3, random_offset=True),
            ], p=0.3),
            A.RandomShadow(p=0.2),
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.1),
        ]

    def _get_medical_augmentations(self) -> list:
        """Medical imaging augmentations."""
        return [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.15,
                rotate_limit=15,
                border_mode=0,
                p=0.5,
            ),
            A.OneOf([
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                A.GridDistortion(num_steps=5, distort_limit=0.1),
            ], p=0.3),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0)),
                A.GaussianBlur(blur_limit=(3, 5)),
            ], p=0.2),
            # Brightness/contrast for staining variations
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5,
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=10,
                p=0.3,
            ),
            A.CoarseDropout(
                max_holes=4,
                max_height=self.image_size[0] // 10,
                max_width=self.image_size[1] // 10,
                fill_value=255,
                p=0.2,
            ),
        ]

    def _get_satellite_augmentations(self) -> list:
        """Satellite/aerial imagery augmentations."""
        return [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.2,
                scale_limit=0.2,
                rotate_limit=180,
                border_mode=0,
                p=0.7,
            ),
            A.OneOf([
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                A.GridDistortion(num_steps=5, distort_limit=0.1),
                A.OpticalDistortion(distort_limit=0.1, shift_limit=0.1),
            ], p=0.3),
            A.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.1,
                p=0.5,
            ),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 100.0)),
                A.GaussianBlur(blur_limit=(3, 7)),
                A.MotionBlur(blur_limit=7),
            ], p=0.3),
            A.RandomShadow(p=0.3),
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.2),
            A.CoarseDropout(
                max_holes=6,
                max_height=self.image_size[0] // 6,
                max_width=self.image_size[1] // 6,
                fill_value=0,
                p=0.3,
            ),
        ]

    def __call__(
        self,
        image: np.ndarray,
        mask: np.ndarray | None = None,
        train: bool = True,
    ) -> dict[str, Any]:
        """Apply augmentation to image (and optionally mask).

        Parameters
        ----------
        image : ndarray
            Input image (H, W, C).
        mask : ndarray, optional
            Segmentation mask (H, W).
        train : bool, default=True
            Whether to use training or validation transform.

        Returns
        -------
        Dict
            Dictionary with 'image' and optionally 'mask'.
        """
        transform = self.train_transform if train else self.val_transform

        if mask is not None:
            return transform(image=image, mask=mask)
        return transform(image=image)

    @staticmethod
    def mixup(
        image1: np.ndarray,
        image2: np.ndarray,
        alpha: float = 0.4,
    ) -> tuple[np.ndarray, float]:
        """Apply MixUp augmentation.

        Interpolates between two images and their labels.

        Parameters
        ----------
        image1 : ndarray
            First image.
        image2 : ndarray
            Second image.
        alpha : float, default=0.4
            Beta distribution parameter.

        Returns
        -------
        mixed_image : ndarray
            Mixed image.
        lam : float
            Mixing coefficient for labels: y = lam * y1 + (1 - lam) * y2
        """
        if alpha <= 0:
            return image1, 1.0

        lam = np.random.beta(alpha, alpha)
        mixed = lam * image1.astype(np.float32) + (1 - lam) * image2.astype(np.float32)
        return mixed.astype(image1.dtype), lam

    @staticmethod
    def cutmix(
        image1: np.ndarray,
        image2: np.ndarray,
        alpha: float = 1.0,
    ) -> tuple[np.ndarray, float]:
        """Apply CutMix augmentation.

        Cuts a patch from image2 and pastes onto image1.

        Parameters
        ----------
        image1 : ndarray
            Base image (H, W, C).
        image2 : ndarray
            Image to cut patch from.
        alpha : float, default=1.0
            Beta distribution parameter.

        Returns
        -------
        mixed_image : ndarray
            Image with cut-pasted region.
        lam : float
            Mixing coefficient (area ratio).
        """
        if alpha <= 0:
            return image1, 1.0

        h, w = image1.shape[:2]

        lam = np.random.beta(alpha, alpha)

        # Sample cut box
        cut_ratio = np.sqrt(1.0 - lam)
        cut_w = int(w * cut_ratio)
        cut_h = int(h * cut_ratio)

        # Random center
        cx = np.random.randint(w)
        cy = np.random.randint(h)

        # Box coordinates
        x1 = np.clip(cx - cut_w // 2, 0, w)
        y1 = np.clip(cy - cut_h // 2, 0, h)
        x2 = np.clip(cx + cut_w // 2, 0, w)
        y2 = np.clip(cy + cut_h // 2, 0, h)

        # Apply cutmix
        mixed = image1.copy()
        mixed[y1:y2, x1:x2] = image2[y1:y2, x1:x2]

        # Actual lambda (based on actual cut area)
        lam = 1 - (x2 - x1) * (y2 - y1) / (w * h)

        return mixed, lam

    @staticmethod
    def cutout(
        image: np.ndarray,
        n_holes: int = 1,
        hole_size: int | tuple[int, int] = 64,
        fill_value: int = 0,
    ) -> np.ndarray:
        """Apply Cutout augmentation.

        Parameters
        ----------
        image : ndarray
            Input image.
        n_holes : int, default=1
            Number of holes to cut.
        hole_size : int or tuple, default=64
            Size of holes (height, width).
        fill_value : int, default=0
            Value to fill holes with.

        Returns
        -------
        ndarray
            Image with cutout regions.
        """
        h, w = image.shape[:2]
        if isinstance(hole_size, int):
            hole_size = (hole_size, hole_size)

        mask = np.ones_like(image)

        for _ in range(n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - hole_size[0] // 2, 0, h)
            y2 = np.clip(y + hole_size[0] // 2, 0, h)
            x1 = np.clip(x - hole_size[1] // 2, 0, w)
            x2 = np.clip(x + hole_size[1] // 2, 0, w)

            mask[y1:y2, x1:x2] = 0

        return image * mask + fill_value * (1 - mask)

    def get_transform_list(self) -> list[str]:
        """Get list of transforms in the training pipeline.

        Returns
        -------
        List[str]
            Names of transforms in order.
        """
        return [t.__class__.__name__ for t in self.train_transform.transforms]
