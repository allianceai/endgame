"""Lightweight ONNX Runtime inference server.

Wraps an ONNX model in an sklearn-like ``predict`` / ``predict_proba``
interface for fast, portable inference without the original training
framework.

Examples
--------
>>> from endgame.persistence import ModelServer
>>> server = ModelServer("model.onnx")
>>> predictions = server.predict(X_test)
>>> probabilities = server.predict_proba(X_test)
"""

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _ensure_onnxruntime():
    """Import onnxruntime, raising a clear error if unavailable."""
    try:
        import onnxruntime
        return onnxruntime
    except ImportError:
        raise ImportError(
            "onnxruntime is required for ONNX model serving. "
            "Install it with: pip install onnxruntime  "
            "(or onnxruntime-gpu for GPU support)"
        )


class ModelServer:
    """Lightweight inference server using ONNX Runtime.

    Loads an ONNX model and provides ``predict`` and ``predict_proba``
    methods that match the sklearn estimator interface. Supports both
    CPU and GPU execution providers.

    Args:
        model_path: Path to the ``.onnx`` model file.
        providers: ONNX Runtime execution providers. Defaults to
            ``["CPUExecutionProvider"]``. Use
            ``["CUDAExecutionProvider", "CPUExecutionProvider"]`` for GPU.
        session_options: Optional ``ort.SessionOptions`` for tuning
            thread count, graph optimisation level, etc.

    Attributes:
        session: The underlying ``ort.InferenceSession``.
        input_names: Names of the model's input tensors.
        output_names: Names of the model's output tensors.
        metadata: ONNX model metadata properties (if any).

    Examples:
        Basic inference::

            >>> from endgame.persistence import ModelServer
            >>> server = ModelServer("model.onnx")
            >>> preds = server.predict(X_test)

        With probability outputs::

            >>> proba = server.predict_proba(X_test)
            >>> proba.shape
            (100, 2)

        GPU inference::

            >>> server = ModelServer(
            ...     "model.onnx",
            ...     providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            ... )

        Custom session options::

            >>> import onnxruntime as ort
            >>> opts = ort.SessionOptions()
            >>> opts.intra_op_num_threads = 4
            >>> server = ModelServer("model.onnx", session_options=opts)
    """

    def __init__(
        self,
        model_path: str | Path,
        providers: Sequence[str] | None = None,
        session_options: Any | None = None,
    ) -> None:
        ort = _ensure_onnxruntime()

        self._model_path = Path(model_path)
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found: {self._model_path}"
            )

        if providers is None:
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(
            str(self._model_path),
            providers=list(providers),
            sess_options=session_options,
        )

        self.input_names: list[str] = [
            inp.name for inp in self.session.get_inputs()
        ]
        self.output_names: list[str] = [
            out.name for out in self.session.get_outputs()
        ]

        # Extract metadata if available
        model_meta = self.session.get_modelmeta()
        self.metadata: dict[str, str] = dict(
            model_meta.custom_metadata_map
        ) if model_meta.custom_metadata_map else {}

        logger.info(
            "Loaded ONNX model from %s (inputs=%s, outputs=%s, providers=%s)",
            self._model_path,
            self.input_names,
            self.output_names,
            providers,
        )

    def _prepare_input(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Prepare input array for ONNX Runtime inference.

        Handles dtype conversion and shape normalization.

        Args:
            X: Input feature array.

        Returns:
            Dictionary mapping input names to numpy arrays.
        """
        arr = np.asarray(X)

        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        # Determine expected input type from the model
        input_info = self.session.get_inputs()[0]
        expected_type = input_info.type

        if "float" in expected_type.lower() or "tensor(float)" in expected_type:
            arr = arr.astype(np.float32)
        elif "double" in expected_type.lower():
            arr = arr.astype(np.float64)
        elif "int64" in expected_type.lower():
            arr = arr.astype(np.int64)
        elif "int32" in expected_type.lower():
            arr = arr.astype(np.int32)
        else:
            # Default to float32 for broadest compatibility
            if not np.issubdtype(arr.dtype, np.floating):
                arr = arr.astype(np.float32)

        return {self.input_names[0]: arr}

    def _run(self, X: np.ndarray) -> list[np.ndarray]:
        """Run inference and return all outputs.

        Args:
            X: Input feature array.

        Returns:
            List of output arrays from the ONNX model.
        """
        feed = self._prepare_input(X)
        return self.session.run(self.output_names, feed)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions from the ONNX model.

        For classifiers exported via skl2onnx, the first output is
        typically the predicted label. For regressors, it is the
        predicted value.

        Args:
            X: Input feature array of shape ``(n_samples, n_features)``.

        Returns:
            Predictions of shape ``(n_samples,)`` for single-output
            models, or ``(n_samples, n_outputs)`` for multi-output.

        Examples:
            >>> server = ModelServer("model.onnx")
            >>> preds = server.predict(X_test)
            >>> preds.shape
            (100,)
        """
        outputs = self._run(X)

        # First output is the prediction (label or value)
        result = outputs[0]

        # Flatten single-column outputs
        result = np.asarray(result)
        if result.ndim == 2 and result.shape[1] == 1:
            result = result.ravel()

        return result

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate class probability estimates from the ONNX model.

        For classifiers exported via skl2onnx, the second output is
        typically the probability array. If only one output exists,
        it is returned directly.

        Args:
            X: Input feature array of shape ``(n_samples, n_features)``.

        Returns:
            Probability array of shape ``(n_samples, n_classes)``.

        Raises:
            RuntimeError: If the model does not produce probability outputs.

        Examples:
            >>> server = ModelServer("model.onnx")
            >>> proba = server.predict_proba(X_test)
            >>> proba.shape
            (100, 2)
        """
        outputs = self._run(X)

        if len(outputs) < 2:
            # Single output model -- return it as probabilities
            # (e.g., torch-exported models with softmax output)
            result = np.asarray(outputs[0])
            if result.ndim == 1:
                # Binary classification: single probability -> two columns
                result = np.column_stack([1.0 - result, result])
            return result

        # skl2onnx classifiers: output[0] = labels, output[1] = probabilities
        proba = outputs[1]

        # skl2onnx may return a list of dicts (when zipmap=True)
        # or a numpy array (when zipmap=False)
        if isinstance(proba, list) and len(proba) > 0:
            if isinstance(proba[0], dict):
                # Convert list of dicts to array
                keys = sorted(proba[0].keys())
                proba = np.array([[d[k] for k in keys] for d in proba])

        return np.asarray(proba, dtype=np.float64)

    def predict_raw(self, X: np.ndarray) -> list[np.ndarray]:
        """Return all raw ONNX model outputs without post-processing.

        Useful for debugging or when the model has non-standard outputs.

        Args:
            X: Input feature array of shape ``(n_samples, n_features)``.

        Returns:
            List of all output arrays from the ONNX model.

        Examples:
            >>> server = ModelServer("model.onnx")
            >>> outputs = server.predict_raw(X_test)
            >>> len(outputs)
            2
        """
        return self._run(X)

    @property
    def input_shapes(self) -> list[list | None]:
        """Return the expected input shapes.

        Returns:
            List of shape lists, one per input. Dimensions may be
            ``None`` for dynamic axes.
        """
        return [inp.shape for inp in self.session.get_inputs()]

    @property
    def output_shapes(self) -> list[list | None]:
        """Return the output shapes.

        Returns:
            List of shape lists, one per output. Dimensions may be
            ``None`` for dynamic axes.
        """
        return [out.shape for out in self.session.get_outputs()]

    def __repr__(self) -> str:
        providers = self.session.get_providers()
        return (
            f"ModelServer(path='{self._model_path}', "
            f"inputs={self.input_names}, "
            f"outputs={self.output_names}, "
            f"providers={providers})"
        )
