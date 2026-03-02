from __future__ import annotations

"""Neural network forecasting models via Darts.

This module wraps Darts' neural forecasting models with sklearn-compatible
interfaces for Kaggle competitions.

Darts provides state-of-the-art neural architectures including:
- N-BEATS: Neural Basis Expansion Analysis
- N-HiTS: Neural Hierarchical Interpolation
- TFT: Temporal Fusion Transformer
- PatchTST: Patch Time Series Transformer
- DLinear: Linear decomposition model
- TimesNet: Temporal 2D-Variation modeling

Installation
------------
pip install darts[torch]

Examples
--------
>>> from endgame.timeseries import NBEATSForecaster
>>> model = NBEATSForecaster(input_chunk_length=30, output_chunk_length=7)
>>> model.fit(train_series)
>>> forecast = model.predict(horizon=7)
"""

from typing import Any

import numpy as np

from endgame.timeseries.base import (
    BaseForecaster,
    validate_forecast_input,
)

# Check for darts availability
try:
    import darts
    from darts import TimeSeries
    from darts.models import (
        DLinearModel,
        NBEATSModel,
        NHiTSModel,
        TFTModel,
    )
    HAS_DARTS = True

    # Optional models (may require additional setup)
    try:
        from darts.models import PatchTSTModel
        HAS_PATCHTST = True
    except ImportError:
        HAS_PATCHTST = False

    try:
        from darts.models import TimesNetModel
        HAS_TIMESNET = True
    except ImportError:
        HAS_TIMESNET = False

except ImportError:
    HAS_DARTS = False
    HAS_PATCHTST = False
    HAS_TIMESNET = False


def _check_darts():
    """Raise ImportError if darts is not installed."""
    if not HAS_DARTS:
        raise ImportError(
            "darts is required for neural forecasting models. "
            "Install with: pip install darts[torch]"
        )


class DartsForecasterWrapper(BaseForecaster):
    """Base wrapper for Darts neural models.

    Provides common functionality for wrapping Darts models
    with sklearn-compatible interface.

    Parameters
    ----------
    model_class : class
        The Darts model class to wrap.
    input_chunk_length : int, default=30
        Length of input sequences (lookback window).
    output_chunk_length : int, default=7
        Length of output sequences (forecast horizon).
    model_kwargs : dict, optional
        Additional arguments for the model.
    trainer_kwargs : dict, optional
        Arguments for PyTorch Lightning trainer.
    random_state : int, optional
        Random seed.
    verbose : bool, default=False
        Enable verbose output.
    """

    def __init__(
        self,
        model_class,
        input_chunk_length: int = 30,
        output_chunk_length: int = 7,
        model_kwargs: dict[str, Any] | None = None,
        trainer_kwargs: dict[str, Any] | None = None,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(random_state=random_state, verbose=verbose)
        self.model_class = model_class
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.model_kwargs = model_kwargs or {}
        self.trainer_kwargs = trainer_kwargs or {}

        self._model = None
        self._series = None

    def _to_timeseries(self, y: np.ndarray) -> TimeSeries:
        """Convert numpy array to Darts TimeSeries."""
        _check_darts()
        return TimeSeries.from_values(y)

    def fit(
        self,
        y: Any,
        X: Any | None = None,
        val_y: Any | None = None,
        **fit_params,
    ) -> DartsForecasterWrapper:
        """Fit the neural forecaster.

        Parameters
        ----------
        y : array-like
            Training time series.
        X : array-like, optional
            Covariates (future-known).
        val_y : array-like, optional
            Validation series for early stopping.

        Returns
        -------
        self
            Fitted forecaster.
        """
        _check_darts()

        y, X_arr = validate_forecast_input(y, X)
        y = y.flatten()

        self.y_ = y.copy()
        self.n_samples_ = len(y)

        if len(y) < self.input_chunk_length + self.output_chunk_length:
            raise ValueError(
                f"Series length ({len(y)}) must be >= "
                f"input_chunk_length ({self.input_chunk_length}) + "
                f"output_chunk_length ({self.output_chunk_length})"
            )

        # Convert to TimeSeries
        self._series = self._to_timeseries(y)

        # Handle covariates
        past_covariates = None
        future_covariates = None
        if X_arr is not None:
            # Treat X as past covariates by default
            past_covariates = self._to_timeseries(X_arr)

        # Validation series
        val_series = None
        if val_y is not None:
            val_y = self._to_numpy(val_y).flatten()
            val_series = self._to_timeseries(val_y)

        # Default trainer kwargs for competition use
        default_trainer = {
            'max_epochs': 100,
            'accelerator': 'auto',
            'enable_progress_bar': self.verbose,
            'logger': False,
        }
        trainer_kwargs = {**default_trainer, **self.trainer_kwargs}

        # Create model
        self._model = self.model_class(
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.output_chunk_length,
            random_state=self.random_state,
            pl_trainer_kwargs=trainer_kwargs,
            **self.model_kwargs,
        )

        # Fit model
        self._model.fit(
            series=self._series,
            past_covariates=past_covariates,
            val_series=val_series,
        )

        self._log(f"Fitted {self.model_class.__name__}")
        self.is_fitted_ = True
        return self

    def predict(
        self,
        horizon: int,
        X: Any | None = None,
    ) -> np.ndarray:
        """Generate neural forecasts.

        Parameters
        ----------
        horizon : int
            Number of steps ahead to forecast.
        X : array-like, optional
            Future covariates.

        Returns
        -------
        np.ndarray
            Point forecasts.
        """
        self._check_is_fitted()
        horizon = self._validate_horizon(horizon)

        # Handle horizons longer than output_chunk_length
        n_predictions = (horizon + self.output_chunk_length - 1) // self.output_chunk_length

        future_covariates = None
        if X is not None:
            X_arr = self._to_numpy(X)
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(-1, 1)
            future_covariates = self._to_timeseries(X_arr)

        # Generate predictions
        predictions = self._model.predict(
            n=horizon,
            series=self._series,
            future_covariates=future_covariates,
        )

        return predictions.values().flatten()[:horizon]

    def predict_interval(
        self,
        horizon: int,
        coverage: float = 0.95,
        X: Any | None = None,
        num_samples: int = 100,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate probabilistic forecasts.

        Parameters
        ----------
        horizon : int
            Forecast horizon.
        coverage : float, default=0.95
            Coverage probability.
        X : array-like, optional
            Future covariates.
        num_samples : int, default=100
            Number of samples for Monte Carlo estimation.

        Returns
        -------
        tuple
            (point_forecast, lower, upper)
        """
        self._check_is_fitted()
        horizon = self._validate_horizon(horizon)

        future_covariates = None
        if X is not None:
            X_arr = self._to_numpy(X)
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(-1, 1)
            future_covariates = self._to_timeseries(X_arr)

        # Generate probabilistic predictions
        predictions = self._model.predict(
            n=horizon,
            series=self._series,
            future_covariates=future_covariates,
            num_samples=num_samples,
        )

        values = predictions.all_values()  # Shape: (horizon, 1, num_samples)

        alpha = 1 - coverage
        lower_q = alpha / 2
        upper_q = 1 - alpha / 2

        point = np.mean(values, axis=2).flatten()[:horizon]
        lower = np.quantile(values, lower_q, axis=2).flatten()[:horizon]
        upper = np.quantile(values, upper_q, axis=2).flatten()[:horizon]

        return point, lower, upper


class NBEATSForecaster(DartsForecasterWrapper):
    """N-BEATS (Neural Basis Expansion Analysis) forecaster.

    State-of-the-art neural architecture for time series forecasting
    based on backward and forward residual links.

    Parameters
    ----------
    input_chunk_length : int, default=30
        Lookback window length.
    output_chunk_length : int, default=7
        Forecast horizon for training.
    num_stacks : int, default=30
        Number of stacks.
    num_blocks : int, default=1
        Number of blocks per stack.
    num_layers : int, default=4
        Number of fully connected layers per block.
    layer_widths : int, default=256
        Width of fully connected layers.
    expansion_coefficient_dim : int, default=5
        Dimension of expansion coefficients.
    generic_architecture : bool, default=True
        Whether to use generic architecture (vs interpretable).
    batch_size : int, default=32
        Training batch size.
    n_epochs : int, default=100
        Number of training epochs.
    learning_rate : float, default=1e-3
        Learning rate.
    random_state : int, optional
        Random seed.
    verbose : bool, default=False
        Enable verbose output.

    Examples
    --------
    >>> model = NBEATSForecaster(input_chunk_length=30, output_chunk_length=7)
    >>> model.fit(train_data)
    >>> forecast = model.predict(horizon=7)

    References
    ----------
    Oreshkin et al. (2020). "N-BEATS: Neural basis expansion analysis
    for interpretable time series forecasting."
    """

    def __init__(
        self,
        input_chunk_length: int = 30,
        output_chunk_length: int = 7,
        num_stacks: int = 30,
        num_blocks: int = 1,
        num_layers: int = 4,
        layer_widths: int = 256,
        expansion_coefficient_dim: int = 5,
        generic_architecture: bool = True,
        batch_size: int = 32,
        n_epochs: int = 100,
        learning_rate: float = 1e-3,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        _check_darts()

        model_kwargs = {
            'num_stacks': num_stacks,
            'num_blocks': num_blocks,
            'num_layers': num_layers,
            'layer_widths': layer_widths,
            'expansion_coefficient_dim': expansion_coefficient_dim,
            'generic_architecture': generic_architecture,
            'batch_size': batch_size,
            'n_epochs': n_epochs,
            'optimizer_kwargs': {'lr': learning_rate},
        }

        super().__init__(
            model_class=NBEATSModel,
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            model_kwargs=model_kwargs,
            random_state=random_state,
            verbose=verbose,
        )

        # Store for sklearn get_params
        self.num_stacks = num_stacks
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.layer_widths = layer_widths
        self.expansion_coefficient_dim = expansion_coefficient_dim
        self.generic_architecture = generic_architecture
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate


class NHITSForecaster(DartsForecasterWrapper):
    """N-HiTS (Neural Hierarchical Interpolation) forecaster.

    Improved version of N-BEATS with hierarchical interpolation
    for better long-horizon forecasting.

    Parameters
    ----------
    input_chunk_length : int, default=30
        Lookback window length.
    output_chunk_length : int, default=7
        Forecast horizon for training.
    num_stacks : int, default=3
        Number of stacks.
    num_blocks : int, default=1
        Number of blocks per stack.
    num_layers : int, default=2
        Number of FC layers per block.
    layer_widths : int, default=512
        Width of FC layers.
    batch_size : int, default=32
        Training batch size.
    n_epochs : int, default=100
        Number of training epochs.
    learning_rate : float, default=1e-3
        Learning rate.
    random_state : int, optional
        Random seed.
    verbose : bool, default=False
        Enable verbose output.

    References
    ----------
    Challu et al. (2022). "N-HiTS: Neural Hierarchical Interpolation
    for Time Series Forecasting."
    """

    def __init__(
        self,
        input_chunk_length: int = 30,
        output_chunk_length: int = 7,
        num_stacks: int = 3,
        num_blocks: int = 1,
        num_layers: int = 2,
        layer_widths: int = 512,
        batch_size: int = 32,
        n_epochs: int = 100,
        learning_rate: float = 1e-3,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        _check_darts()

        model_kwargs = {
            'num_stacks': num_stacks,
            'num_blocks': num_blocks,
            'num_layers': num_layers,
            'layer_widths': layer_widths,
            'batch_size': batch_size,
            'n_epochs': n_epochs,
            'optimizer_kwargs': {'lr': learning_rate},
        }

        super().__init__(
            model_class=NHiTSModel,
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            model_kwargs=model_kwargs,
            random_state=random_state,
            verbose=verbose,
        )

        self.num_stacks = num_stacks
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.layer_widths = layer_widths
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate


class TFTForecaster(DartsForecasterWrapper):
    """Temporal Fusion Transformer forecaster.

    Multi-horizon forecasting with interpretable attention mechanism.
    Handles static, known future, and observed inputs.

    Parameters
    ----------
    input_chunk_length : int, default=30
        Lookback window length.
    output_chunk_length : int, default=7
        Forecast horizon.
    hidden_size : int, default=64
        Hidden state size.
    lstm_layers : int, default=1
        Number of LSTM layers.
    num_attention_heads : int, default=4
        Number of attention heads.
    hidden_continuous_size : int, default=8
        Hidden size for continuous variable processing.
    dropout : float, default=0.1
        Dropout rate.
    batch_size : int, default=32
        Training batch size.
    n_epochs : int, default=100
        Number of training epochs.
    learning_rate : float, default=1e-3
        Learning rate.
    add_relative_index : bool, default=True
        Whether to add relative time index as feature.
    random_state : int, optional
        Random seed.
    verbose : bool, default=False
        Enable verbose output.

    References
    ----------
    Lim et al. (2021). "Temporal Fusion Transformers for Interpretable
    Multi-horizon Time Series Forecasting."
    """

    def __init__(
        self,
        input_chunk_length: int = 30,
        output_chunk_length: int = 7,
        hidden_size: int = 64,
        lstm_layers: int = 1,
        num_attention_heads: int = 4,
        hidden_continuous_size: int = 8,
        dropout: float = 0.1,
        batch_size: int = 32,
        n_epochs: int = 100,
        learning_rate: float = 1e-3,
        add_relative_index: bool = True,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        _check_darts()

        model_kwargs = {
            'hidden_size': hidden_size,
            'lstm_layers': lstm_layers,
            'num_attention_heads': num_attention_heads,
            'hidden_continuous_size': hidden_continuous_size,
            'dropout': dropout,
            'batch_size': batch_size,
            'n_epochs': n_epochs,
            'optimizer_kwargs': {'lr': learning_rate},
            'add_relative_index': add_relative_index,
        }

        super().__init__(
            model_class=TFTModel,
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            model_kwargs=model_kwargs,
            random_state=random_state,
            verbose=verbose,
        )

        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_continuous_size = hidden_continuous_size
        self.dropout = dropout
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.add_relative_index = add_relative_index


class PatchTSTForecaster(DartsForecasterWrapper):
    """Patch Time Series Transformer forecaster.

    Transformer model that uses patching for efficient long sequence modeling.

    Parameters
    ----------
    input_chunk_length : int, default=32
        Lookback window length (should be divisible by patch_length).
    output_chunk_length : int, default=7
        Forecast horizon.
    patch_length : int, default=16
        Length of each patch.
    d_model : int, default=128
        Dimension of the model.
    nhead : int, default=4
        Number of attention heads.
    num_encoder_layers : int, default=3
        Number of encoder layers.
    dim_feedforward : int, default=256
        Feedforward network dimension.
    dropout : float, default=0.1
        Dropout rate.
    batch_size : int, default=32
        Training batch size.
    n_epochs : int, default=100
        Number of training epochs.
    learning_rate : float, default=1e-3
        Learning rate.
    random_state : int, optional
        Random seed.
    verbose : bool, default=False
        Enable verbose output.

    References
    ----------
    Nie et al. (2023). "A Time Series is Worth 64 Words: Long-term
    Forecasting with Transformers."
    """

    def __init__(
        self,
        input_chunk_length: int = 32,
        output_chunk_length: int = 7,
        patch_length: int = 16,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        batch_size: int = 32,
        n_epochs: int = 100,
        learning_rate: float = 1e-3,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        if not HAS_PATCHTST:
            raise ImportError(
                "PatchTSTModel requires a recent version of darts. "
                "Install with: pip install -U darts[torch]"
            )

        model_kwargs = {
            'patch_length': patch_length,
            'd_model': d_model,
            'nhead': nhead,
            'num_encoder_layers': num_encoder_layers,
            'dim_feedforward': dim_feedforward,
            'dropout': dropout,
            'batch_size': batch_size,
            'n_epochs': n_epochs,
            'optimizer_kwargs': {'lr': learning_rate},
        }

        super().__init__(
            model_class=PatchTSTModel,
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            model_kwargs=model_kwargs,
            random_state=random_state,
            verbose=verbose,
        )

        self.patch_length = patch_length
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate


class DLinearForecaster(DartsForecasterWrapper):
    """DLinear forecaster.

    Simple linear model with trend-seasonal decomposition.
    Often competitive with complex transformers while being much faster.

    Parameters
    ----------
    input_chunk_length : int, default=30
        Lookback window length.
    output_chunk_length : int, default=7
        Forecast horizon.
    shared_weights : bool, default=False
        Whether to share weights across series.
    kernel_size : int, default=25
        Kernel size for moving average decomposition.
    batch_size : int, default=32
        Training batch size.
    n_epochs : int, default=100
        Number of training epochs.
    learning_rate : float, default=1e-3
        Learning rate.
    random_state : int, optional
        Random seed.
    verbose : bool, default=False
        Enable verbose output.

    References
    ----------
    Zeng et al. (2023). "Are Transformers Effective for Time Series
    Forecasting?"
    """

    def __init__(
        self,
        input_chunk_length: int = 30,
        output_chunk_length: int = 7,
        shared_weights: bool = False,
        kernel_size: int = 25,
        batch_size: int = 32,
        n_epochs: int = 100,
        learning_rate: float = 1e-3,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        _check_darts()

        model_kwargs = {
            'shared_weights': shared_weights,
            'kernel_size': kernel_size,
            'batch_size': batch_size,
            'n_epochs': n_epochs,
            'optimizer_kwargs': {'lr': learning_rate},
        }

        super().__init__(
            model_class=DLinearModel,
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            model_kwargs=model_kwargs,
            random_state=random_state,
            verbose=verbose,
        )

        self.shared_weights = shared_weights
        self.kernel_size = kernel_size
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate


class TimesNetForecaster(DartsForecasterWrapper):
    """TimesNet forecaster.

    Temporal 2D-variation modeling for time series analysis.
    Converts 1D time series to 2D tensors for pattern extraction.

    Parameters
    ----------
    input_chunk_length : int, default=30
        Lookback window length.
    output_chunk_length : int, default=7
        Forecast horizon.
    hidden_size : int, default=64
        Hidden dimension size.
    num_encoder_layers : int, default=2
        Number of encoder layers.
    num_kernels : int, default=6
        Number of inception kernels.
    batch_size : int, default=32
        Training batch size.
    n_epochs : int, default=100
        Number of training epochs.
    learning_rate : float, default=1e-3
        Learning rate.
    random_state : int, optional
        Random seed.
    verbose : bool, default=False
        Enable verbose output.

    References
    ----------
    Wu et al. (2023). "TimesNet: Temporal 2D-Variation Modeling for
    General Time Series Analysis."
    """

    def __init__(
        self,
        input_chunk_length: int = 30,
        output_chunk_length: int = 7,
        hidden_size: int = 64,
        num_encoder_layers: int = 2,
        num_kernels: int = 6,
        batch_size: int = 32,
        n_epochs: int = 100,
        learning_rate: float = 1e-3,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        if not HAS_TIMESNET:
            raise ImportError(
                "TimesNetModel requires a recent version of darts. "
                "Install with: pip install -U darts[torch]"
            )

        model_kwargs = {
            'hidden_size': hidden_size,
            'num_encoder_layers': num_encoder_layers,
            'num_kernels': num_kernels,
            'batch_size': batch_size,
            'n_epochs': n_epochs,
            'optimizer_kwargs': {'lr': learning_rate},
        }

        super().__init__(
            model_class=TimesNetModel,
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            model_kwargs=model_kwargs,
            random_state=random_state,
            verbose=verbose,
        )

        self.hidden_size = hidden_size
        self.num_encoder_layers = num_encoder_layers
        self.num_kernels = num_kernels
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
