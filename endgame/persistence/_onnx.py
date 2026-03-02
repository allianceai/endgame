from __future__ import annotations

"""ONNX export for sklearn-compatible estimators.

Converts fitted estimators to ONNX format for portable, framework-agnostic
inference. Auto-detects the best conversion backend based on estimator type:

- sklearn / sklearn-compatible models -> skl2onnx
- Tree-based GBDTs (LightGBM, XGBoost, CatBoost) -> skl2onnx (with converters)
- PyTorch ``nn.Module``-backed models -> ``torch.onnx.export``
- Fallback for unsupported sklearn models -> hummingbird-ml

Examples
--------
>>> import endgame as eg
>>> from sklearn.ensemble import RandomForestClassifier
>>> model = RandomForestClassifier().fit(X_train, y_train)
>>> eg.export_onnx(model, "/tmp/model.onnx", sample_input=X_train[:1])
'/tmp/model.onnx'
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np

from endgame.persistence._detection import find_torch_modules, has_torch_modules

logger = logging.getLogger(__name__)

# ONNX file extension
ONNX_EXT = ".onnx"

# Mapping from numpy dtype to ONNX TensorProto type names
_NP_DTYPE_TO_ONNX = {
    np.float32: "FloatTensorType",
    np.float64: "DoubleTensorType",
    np.int64: "Int64TensorType",
    np.int32: "Int32TensorType",
}


def _ensure_skl2onnx():
    """Import skl2onnx, raising a clear error if unavailable."""
    try:
        import skl2onnx
        return skl2onnx
    except ImportError:
        raise ImportError(
            "skl2onnx is required for ONNX export of sklearn models. "
            "Install it with: pip install skl2onnx"
        )


def _ensure_onnx():
    """Import onnx, raising a clear error if unavailable."""
    try:
        import onnx
        return onnx
    except ImportError:
        raise ImportError(
            "onnx is required for ONNX export. "
            "Install it with: pip install onnx"
        )


def _ensure_hummingbird():
    """Import hummingbird-ml, raising a clear error if unavailable."""
    try:
        from hummingbird.ml import convert as hb_convert
        return hb_convert
    except ImportError:
        raise ImportError(
            "hummingbird-ml is required for this conversion backend. "
            "Install it with: pip install hummingbird-ml"
        )


def _is_gbdt(estimator: Any) -> bool:
    """Check if the estimator is a tree-based GBDT model.

    Detects LightGBM, XGBoost, and CatBoost models, including endgame
    wrappers around them.

    Args:
        estimator: The estimator to check.

    Returns:
        True if the estimator is a GBDT.
    """
    cls_name = type(estimator).__name__
    module_name = type(estimator).__module__ or ""

    # Direct LightGBM / XGBoost / CatBoost instances
    gbdt_modules = ("lightgbm", "xgboost", "catboost")
    if any(mod in module_name for mod in gbdt_modules):
        return True

    # Endgame wrappers
    gbdt_wrapper_names = (
        "LGBMWrapper", "XGBWrapper", "CatBoostWrapper", "GBDTWrapper",
    )
    if cls_name in gbdt_wrapper_names:
        return True

    return False


def _is_sklearn_compatible(estimator: Any) -> bool:
    """Check if the estimator follows the sklearn API.

    Args:
        estimator: The estimator to check.

    Returns:
        True if the estimator has ``fit`` and ``predict`` methods.
    """
    return hasattr(estimator, "fit") and hasattr(estimator, "predict")


def _unwrap_estimator(estimator: Any) -> Any:
    """Unwrap endgame wrappers to get the underlying library estimator.

    Endgame GBDT wrappers store the underlying model in ``model_`` after
    fitting. For ONNX conversion, we need the raw LightGBM/XGBoost/CatBoost
    object.

    Args:
        estimator: An endgame wrapper or raw estimator.

    Returns:
        The underlying estimator, or the input unchanged.
    """
    # Endgame GBDT wrappers store the fitted model in model_
    if hasattr(estimator, "model_") and _is_gbdt(estimator):
        return estimator.model_
    return estimator


def _detect_onnx_backend(estimator: Any, preferred: str) -> str:
    """Determine the best ONNX conversion backend for the estimator.

    Args:
        estimator: The fitted estimator.
        preferred: User-requested backend. ``"auto"`` enables detection.

    Returns:
        Backend name: ``"skl2onnx"``, ``"torch"``, or ``"hummingbird"``.

    Raises:
        ValueError: If the preferred backend is not recognised.
    """
    valid_backends = ("auto", "skl2onnx", "hummingbird", "torch")
    if preferred not in valid_backends:
        raise ValueError(
            f"Unknown ONNX backend '{preferred}'. "
            f"Choose from: {valid_backends}"
        )

    if preferred != "auto":
        return preferred

    # PyTorch-backed estimators
    if has_torch_modules(estimator):
        return "torch"

    # Everything else goes through skl2onnx (which has converters for
    # sklearn, LightGBM, XGBoost, and CatBoost)
    return "skl2onnx"


def _infer_initial_types(
    sample_input: np.ndarray | None,
    estimator: Any,
) -> list:
    """Build the ``initial_types`` list required by skl2onnx.

    Args:
        sample_input: A sample input array for shape/dtype inference.
            If ``None``, attempts to infer from the estimator's
            ``n_features_in_`` attribute.
        estimator: The fitted estimator (used for fallback shape inference).

    Returns:
        A list of ``(name, type)`` pairs suitable for
        ``skl2onnx.convert_sklearn``.

    Raises:
        ValueError: If the input shape cannot be determined.
    """
    skl2onnx = _ensure_skl2onnx()
    from skl2onnx.common.data_types import (
        FloatTensorType,
        Int32TensorType,
        Int64TensorType,
    )

    type_map = {
        np.float32: FloatTensorType,
        np.float64: FloatTensorType,  # cast to float32 for ONNX compat
        np.int64: Int64TensorType,
        np.int32: Int32TensorType,
    }

    if sample_input is not None:
        arr = np.asarray(sample_input)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        n_features = arr.shape[1]
        dtype = arr.dtype.type

        tensor_type = type_map.get(dtype, FloatTensorType)
        return [("X", tensor_type([None, n_features]))]

    # Fallback: infer from estimator metadata
    n_features = getattr(estimator, "n_features_in_", None)
    if n_features is None:
        n_features = getattr(estimator, "_n_features_in", None)

    if n_features is not None:
        return [("X", FloatTensorType([None, n_features]))]

    raise ValueError(
        "Cannot infer input shape. Provide a sample_input array or ensure "
        "the estimator has n_features_in_ set (call fit() first)."
    )


def _register_gbdt_converters() -> None:
    """Register skl2onnx converters for LightGBM, XGBoost, and CatBoost.

    These converters are provided by their respective ``onnxmltools``
    integration packages and need to be registered before conversion.
    """
    # LightGBM
    try:
        import lightgbm
        import skl2onnx
        from onnxmltools.convert.lightgbm.operator_converters.LightGbm import (
            convert_lightgbm,  # noqa: F401
        )
        from skl2onnx.common.shape_calculator import (
            calculate_linear_classifier_output_shapes,
            calculate_linear_regressor_output_shapes,
        )

        skl2onnx.update_registered_converter(
            lightgbm.LGBMClassifier,
            "LightGbmLGBMClassifier",
            calculate_linear_classifier_output_shapes,
            convert_lightgbm,
            options={"zipmap": [True, False, "columns"]},
        )
        skl2onnx.update_registered_converter(
            lightgbm.LGBMRegressor,
            "LightGbmLGBMRegressor",
            calculate_linear_regressor_output_shapes,
            convert_lightgbm,
        )
        logger.debug("Registered LightGBM ONNX converters")
    except ImportError:
        logger.debug("LightGBM ONNX converter not available")

    # XGBoost
    try:
        import skl2onnx
        import xgboost
        from onnxmltools.convert.xgboost.operator_converters.XGBoost import (
            convert_xgboost,  # noqa: F401
        )
        from skl2onnx.common.shape_calculator import (
            calculate_linear_classifier_output_shapes,
            calculate_linear_regressor_output_shapes,
        )

        skl2onnx.update_registered_converter(
            xgboost.XGBClassifier,
            "XGBoostXGBClassifier",
            calculate_linear_classifier_output_shapes,
            convert_xgboost,
            options={"zipmap": [True, False, "columns"]},
        )
        skl2onnx.update_registered_converter(
            xgboost.XGBRegressor,
            "XGBoostXGBRegressor",
            calculate_linear_regressor_output_shapes,
            convert_xgboost,
        )
        logger.debug("Registered XGBoost ONNX converters")
    except ImportError:
        logger.debug("XGBoost ONNX converter not available")

    # CatBoost
    try:
        import catboost
        import skl2onnx
        from onnxmltools.convert.catboost.operator_converters.CatBoost import (
            convert_catboost,  # noqa: F401
        )
        from skl2onnx.common.shape_calculator import (
            calculate_linear_classifier_output_shapes,
            calculate_linear_regressor_output_shapes,
        )

        skl2onnx.update_registered_converter(
            catboost.CatBoostClassifier,
            "CatBoostCatBoostClassifier",
            calculate_linear_classifier_output_shapes,
            convert_catboost,
        )
        skl2onnx.update_registered_converter(
            catboost.CatBoostRegressor,
            "CatBoostCatBoostRegressor",
            calculate_linear_regressor_output_shapes,
            convert_catboost,
        )
        logger.debug("Registered CatBoost ONNX converters")
    except ImportError:
        logger.debug("CatBoost ONNX converter not available")


def _export_skl2onnx(
    estimator: Any,
    path: Path,
    sample_input: np.ndarray | None,
    opset_version: int,
) -> Path:
    """Export using skl2onnx (sklearn, LightGBM, XGBoost, CatBoost).

    Args:
        estimator: Fitted sklearn-compatible estimator.
        path: Output file path.
        sample_input: Sample input for shape inference.
        opset_version: ONNX opset version.

    Returns:
        Path to the saved ONNX file.
    """
    skl2onnx = _ensure_skl2onnx()
    onnx = _ensure_onnx()

    raw_estimator = _unwrap_estimator(estimator)

    # Register GBDT converters if needed
    if _is_gbdt(estimator):
        _register_gbdt_converters()

    initial_types = _infer_initial_types(sample_input, estimator)

    model_name = type(estimator).__name__

    try:
        onnx_model = skl2onnx.convert_sklearn(
            raw_estimator,
            name=model_name,
            initial_types=initial_types,
            target_opset=opset_version,
            options={id(raw_estimator): {"zipmap": False}}
            if hasattr(raw_estimator, "predict_proba")
            else None,
        )
    except Exception as exc:
        raise RuntimeError(
            f"skl2onnx conversion failed for {model_name}: {exc}. "
            f"Try backend='hummingbird' as an alternative."
        ) from exc

    onnx.save_model(onnx_model, str(path))
    logger.info("Exported %s to ONNX via skl2onnx: %s", model_name, path)
    return path


def _export_hummingbird(
    estimator: Any,
    path: Path,
    sample_input: np.ndarray | None,
    opset_version: int,
) -> Path:
    """Export using hummingbird-ml (fallback for unsupported models).

    Hummingbird converts traditional ML models to tensor computations
    (PyTorch/ONNX) for faster inference. It supports a broader set of
    sklearn-compatible models than skl2onnx.

    Args:
        estimator: Fitted sklearn-compatible estimator.
        path: Output file path.
        sample_input: Sample input for shape inference.
        opset_version: ONNX opset version.

    Returns:
        Path to the saved ONNX file.
    """
    hb_convert = _ensure_hummingbird()
    _ensure_onnx()

    raw_estimator = _unwrap_estimator(estimator)

    if sample_input is not None:
        test_input = np.asarray(sample_input, dtype=np.float32)
        if test_input.ndim == 1:
            test_input = test_input.reshape(1, -1)
    else:
        # Hummingbird can sometimes work without test input, but
        # shape inference is more reliable with one
        test_input = None

    model_name = type(estimator).__name__

    try:
        hb_model = hb_convert(
            raw_estimator,
            "onnx",
            test_input=test_input,
            extra_config={"onnx_target_opset": opset_version},
        )
        hb_model.save(str(path))
    except Exception as exc:
        raise RuntimeError(
            f"Hummingbird conversion failed for {model_name}: {exc}. "
            f"This model may not be supported for ONNX export."
        ) from exc

    logger.info(
        "Exported %s to ONNX via hummingbird: %s", model_name, path
    )
    return path


def _export_torch(
    estimator: Any,
    path: Path,
    sample_input: np.ndarray | None,
    opset_version: int,
) -> Path:
    """Export a PyTorch-backed estimator via ``torch.onnx.export``.

    Extracts the ``nn.Module`` from the estimator and traces it with
    the provided sample input.

    Args:
        estimator: Estimator containing one or more ``nn.Module`` attributes.
        path: Output file path.
        sample_input: Sample input array (required for torch tracing).
        opset_version: ONNX opset version.

    Returns:
        Path to the saved ONNX file.

    Raises:
        ValueError: If no sample input is provided or no ``nn.Module`` found.
    """
    import torch

    _ensure_onnx()

    if sample_input is None:
        raise ValueError(
            "sample_input is required for PyTorch ONNX export. "
            "Provide a representative input array."
        )

    modules = find_torch_modules(estimator)
    if not modules:
        raise ValueError(
            f"No nn.Module found in {type(estimator).__name__}. "
            f"Use backend='skl2onnx' or 'hummingbird' instead."
        )

    # Use the first module found (primary model)
    attr_name, module = next(iter(modules.items()))
    logger.info("Exporting nn.Module from attribute '%s'", attr_name)

    module.eval()

    # Prepare input tensor
    arr = np.asarray(sample_input, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    dummy_input = torch.from_numpy(arr)

    # Move to same device as model
    try:
        device = next(module.parameters()).device
        dummy_input = dummy_input.to(device)
    except StopIteration:
        pass  # No parameters, stay on CPU

    model_name = type(estimator).__name__

    try:
        torch.onnx.export(
            module,
            dummy_input,
            str(path),
            opset_version=opset_version,
            input_names=["X"],
            output_names=["output"],
            dynamic_axes={
                "X": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
        )
    except Exception as exc:
        raise RuntimeError(
            f"torch.onnx.export failed for {model_name}: {exc}"
        ) from exc

    logger.info("Exported %s to ONNX via torch: %s", model_name, path)
    return path


def export_onnx(
    estimator: Any,
    path: str | Path,
    sample_input: np.ndarray | None = None,
    opset_version: int = 15,
    backend: str = "auto",
) -> str:
    """Export a fitted estimator to ONNX format.

    Auto-detects the best conversion backend based on the estimator type:

    - sklearn models -> skl2onnx
    - Tree-based GBDTs (LightGBM, XGBoost, CatBoost) -> skl2onnx
      (with registered converters from onnxmltools)
    - PyTorch ``nn.Module``-backed models -> ``torch.onnx.export``
    - Fallback -> hummingbird-ml

    Args:
        estimator: Fitted sklearn-compatible estimator.
        path: Output file path. The ``.onnx`` extension is added
            automatically if not present.
        sample_input: Sample input array for shape inference. Required
            for PyTorch models; strongly recommended for all others.
        opset_version: ONNX opset version. Default is 15, which provides
            broad operator coverage.
        backend: Conversion backend. ``"auto"`` selects the best backend
            based on estimator type. Explicit options: ``"skl2onnx"``,
            ``"hummingbird"``, ``"torch"``.

    Returns:
        Path to the saved ONNX file.

    Raises:
        ValueError: If the backend is unknown or input shape cannot be
            inferred.
        RuntimeError: If the ONNX conversion fails.

    Examples:
        Export a scikit-learn model::

            >>> from sklearn.ensemble import RandomForestClassifier
            >>> import endgame as eg
            >>> model = RandomForestClassifier(n_estimators=10).fit(X, y)
            >>> eg.export_onnx(model, "rf_model.onnx", sample_input=X[:1])
            'rf_model.onnx'

        Export a LightGBM model::

            >>> from endgame.models.wrappers import LGBMWrapper
            >>> model = LGBMWrapper(task='classification').fit(X, y)
            >>> eg.export_onnx(model, "lgbm.onnx", sample_input=X[:1])
            'lgbm.onnx'

        Export with a specific backend::

            >>> eg.export_onnx(model, "model.onnx", backend='hummingbird')
            'model.onnx'
    """
    dest = Path(path)
    if dest.suffix != ONNX_EXT:
        dest = dest.with_suffix(ONNX_EXT)

    # Ensure parent directory exists
    dest.parent.mkdir(parents=True, exist_ok=True)

    resolved = _detect_onnx_backend(estimator, preferred=backend)
    logger.info(
        "Exporting %s with backend '%s' (requested: '%s')",
        type(estimator).__name__,
        resolved,
        backend,
    )

    if resolved == "torch":
        result = _export_torch(estimator, dest, sample_input, opset_version)
    elif resolved == "hummingbird":
        result = _export_hummingbird(
            estimator, dest, sample_input, opset_version
        )
    else:
        result = _export_skl2onnx(
            estimator, dest, sample_input, opset_version
        )

    return str(result)
