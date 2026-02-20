Endgame Documentation
=====================

**Endgame** is a comprehensive machine learning toolkit providing 300+ estimators,
transformers, and visualizers across tabular, time series, signal processing, CV,
NLP, audio, and multimodal domains. It unifies state-of-the-art and classical
methods under a consistent scikit-learn-compatible API.

.. code-block:: python

   import endgame as eg

   # Quick model comparison
   result = eg.quick.compare(X, y, task='classification')

   # Full pipeline
   model = eg.models.LGBMWrapper(preset='endgame')
   model.fit(X_train, y_train)
   predictions = model.predict(X_test)

Key Features
------------

- **100+ models** with sklearn-compatible API
- **Polars-powered** preprocessing for speed
- **Competition-winning defaults** via preset system
- **Conformal prediction** and probability calibration
- **Comprehensive signal processing** (45 transforms)
- **AutoML framework** matching AutoGluon's simplicity
- **42 interactive visualizations** for model interpretation

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   guides/installation
   guides/quickstart

.. toctree::
   :maxdepth: 2
   :caption: User Guides

   guides/models
   guides/preprocessing
   guides/ensembles
   guides/calibration
   guides/timeseries
   guides/signal
   guides/automl
   guides/explainability
   guides/visualization
   guides/tracking
   guides/mcp_server

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/validation
   api/preprocessing
   api/models
   api/ensemble
   api/calibration
   api/explain
   api/fairness
   api/anomaly
   api/tune
   api/quick
   api/semi_supervised
   api/persistence
   api/feature_selection
   api/dimensionality_reduction
   api/clustering
   api/visualization
   api/signal
   api/timeseries
   api/benchmark
   api/automl
   api/tracking
   api/mcp
   api/utils

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
