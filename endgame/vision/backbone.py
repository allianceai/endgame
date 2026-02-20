"""Vision backbone wrappers using timm."""

from typing import Any

import numpy as np

from endgame.core.base import EndgameEstimator


class VisionBackbone(EndgameEstimator):
    """Unified interface for vision backbones via timm.

    Provides easy access to SOTA vision models with competition-tuned defaults.

    Parameters
    ----------
    architecture : str, default='efficientnet_b0'
        Model architecture from timm library.
        Popular choices:
        - 'efficientnet_b0' to 'efficientnet_b7' (fast, reliable)
        - 'swin_base_patch4_window7_224' (Swin Transformer)
        - 'convnext_base' (modern CNN)
        - 'vit_base_patch16_224' (Vision Transformer)
        - 'eva02_large_patch14_448' (SOTA 2024)
    pretrained : bool, default=True
        Use pretrained weights.
    num_classes : int, optional
        Output classes. None for feature extraction.
    in_chans : int, default=3
        Number of input channels.
    drop_rate : float, default=0.0
        Dropout rate.
    drop_path_rate : float, default=0.0
        Drop path rate for stochastic depth.
    global_pool : str, default='avg'
        Global pooling type: 'avg', 'max', 'avgmax', 'catavgmax'.

    Examples
    --------
    >>> from endgame.vision import VisionBackbone
    >>> model = VisionBackbone('efficientnet_b4', num_classes=10)
    >>> # Training loop with your data...
    """

    def __init__(
        self,
        architecture: str = "efficientnet_b0",
        pretrained: bool = True,
        num_classes: int | None = None,
        in_chans: int = 3,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        global_pool: str = "avg",
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(random_state=random_state, verbose=verbose)
        self.architecture = architecture
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate
        self.global_pool = global_pool

        self._model = None
        self._device = None

    def _create_model(self):
        """Create the timm model."""
        try:
            import timm
            import torch
        except ImportError:
            raise ImportError(
                "timm and torch are required for VisionBackbone. "
                "Install with: pip install timm torch"
            )

        self._model = timm.create_model(
            self.architecture,
            pretrained=self.pretrained,
            num_classes=self.num_classes or 0,
            in_chans=self.in_chans,
            drop_rate=self.drop_rate,
            drop_path_rate=self.drop_path_rate,
            global_pool=self.global_pool if self.num_classes else '',
        )

        # Set device
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = self._model.to(self._device)

        return self._model

    def get_model(self):
        """Get the underlying timm model.

        Returns
        -------
        nn.Module
            The timm model.
        """
        if self._model is None:
            self._create_model()
        return self._model

    def get_config(self) -> dict[str, Any]:
        """Get model configuration.

        Returns
        -------
        Dict
            Model configuration including input size, mean, std.
        """
        try:
            import timm
        except ImportError:
            raise ImportError("timm is required")

        if self._model is None:
            self._create_model()

        data_config = timm.data.resolve_model_data_config(self._model)
        return {
            "input_size": data_config["input_size"],
            "mean": data_config["mean"],
            "std": data_config["std"],
            "interpolation": data_config["interpolation"],
        }

    def get_transforms(self, is_training: bool = True):
        """Get transforms for this model.

        Parameters
        ----------
        is_training : bool, default=True
            Whether to get training or validation transforms.

        Returns
        -------
        Transform
            Albumentations or torchvision transform.
        """
        try:
            import timm
        except ImportError:
            raise ImportError("timm is required")

        if self._model is None:
            self._create_model()

        data_config = timm.data.resolve_model_data_config(self._model)

        if is_training:
            return timm.data.create_transform(**data_config, is_training=True)
        return timm.data.create_transform(**data_config, is_training=False)

    def extract_features(self, images) -> np.ndarray:
        """Extract features from images.

        Parameters
        ----------
        images : Tensor or array-like
            Input images.

        Returns
        -------
        ndarray
            Feature vectors.
        """
        import torch

        if self._model is None:
            self._create_model()

        self._model.eval()

        with torch.no_grad():
            if not isinstance(images, torch.Tensor):
                images = torch.tensor(images)
            images = images.to(self._device)

            features = self._model.forward_features(images)

            # Global pooling if needed
            if features.dim() == 4:
                features = features.mean(dim=[2, 3])

            return features.cpu().numpy()

    @staticmethod
    def list_models(filter_str: str = "") -> list[str]:
        """List available timm models.

        Parameters
        ----------
        filter_str : str
            Filter string for model names.

        Returns
        -------
        List[str]
            Available model names.
        """
        try:
            import timm
        except ImportError:
            raise ImportError("timm is required")

        return timm.list_models(filter_str)
