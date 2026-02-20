"""Model persistence for Endgame estimators.

Save and load any sklearn-compatible estimator, including PyTorch-backed
models, with automatic backend detection and metadata tracking. Export
fitted models to ONNX format for portable inference.

Examples
--------
>>> import endgame as eg
>>> from sklearn.linear_model import LogisticRegression
>>> model = LogisticRegression().fit(X_train, y_train)
>>> eg.save(model, "/tmp/my_model")
'/tmp/my_model.egm'
>>> loaded = eg.load("/tmp/my_model.egm")

ONNX export and inference::

>>> eg.export_onnx(model, "/tmp/model.onnx", sample_input=X_train[:1])
'/tmp/model.onnx'
>>> from endgame.persistence import ModelServer
>>> server = ModelServer("/tmp/model.onnx")
>>> server.predict(X_test)
"""

from endgame.persistence._core import load, save
from endgame.persistence._metadata import ModelMetadata
from endgame.persistence._onnx import export_onnx
from endgame.persistence._server import ModelServer

__all__ = ["save", "load", "ModelMetadata", "export_onnx", "ModelServer"]
