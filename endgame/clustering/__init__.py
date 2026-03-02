from __future__ import annotations

"""Clustering module: 16 algorithms spanning centroid, density, hierarchical,
distribution, graph, and scalable methods plus automatic method selection.

Example
-------
>>> from endgame.clustering import AutoCluster, KMeansClusterer, HDBSCANClusterer
>>>
>>> # Automatic method selection
>>> ac = AutoCluster(n_clusters='auto', detect_noise=True)
>>> labels = ac.fit_predict(X)
>>> print(f"Selected: {ac.selected_method_}, k={ac.n_clusters_}")
>>>
>>> # Direct usage
>>> km = KMeansClusterer(n_clusters=5, random_state=42)
>>> labels = km.fit_predict(X)
>>>
>>> # Density-based (noise detection)
>>> hdb = HDBSCANClusterer(min_cluster_size=15)
>>> labels = hdb.fit_predict(X)
"""

# Centroid-based
# Auto-selection
from endgame.clustering.auto import AutoCluster
from endgame.clustering.centroid import (
    KMeansClusterer,
    KStarMeansClusterer,
    MiniBatchKMeansClusterer,
)

# Density-based
from endgame.clustering.density import (
    DBSCANClusterer,
    DensityPeaksClusterer,
    HDBSCANClusterer,
    OPTICSClusterer,
)

# Distribution-based
from endgame.clustering.distribution import (
    FuzzyCMeansClusterer,
    GaussianMixtureClusterer,
)

# Graph/Spectral
from endgame.clustering.graph import (
    AffinityPropagationClusterer,
    SpectralClusterer,
)

# Hierarchical
from endgame.clustering.hierarchical import (
    AgglomerativeClusterer,
)

# Scalable
from endgame.clustering.scalable import (
    BIRCHClusterer,
    MeanShiftClusterer,
)

__all__ = [
    # Centroid-based
    "KMeansClusterer",
    "MiniBatchKMeansClusterer",
    "KStarMeansClusterer",
    # Density-based
    "DBSCANClusterer",
    "HDBSCANClusterer",
    "OPTICSClusterer",
    "DensityPeaksClusterer",
    # Hierarchical
    "AgglomerativeClusterer",
    # Distribution-based
    "GaussianMixtureClusterer",
    "FuzzyCMeansClusterer",
    # Graph/Spectral
    "SpectralClusterer",
    "AffinityPropagationClusterer",
    # Scalable
    "BIRCHClusterer",
    "MeanShiftClusterer",
    # Auto-selection
    "AutoCluster",
]

# Optional: Genie (requires genieclust)
try:
    from endgame.clustering.hierarchical import GenieClusterer
    __all__.append("GenieClusterer")
except (ImportError, NameError):
    pass

# Optional: FINCH (requires finch-clust)
try:
    from endgame.clustering.scalable import FINCHClusterer
    __all__.append("FINCHClusterer")
except (ImportError, NameError):
    pass
