"""Tests for the clustering module."""

import numpy as np
import pytest
from sklearn.datasets import make_blobs, make_moons


@pytest.fixture
def blob_data():
    """Well-separated blobs for centroid-based methods."""
    X, y = make_blobs(
        n_samples=300, n_features=5, centers=3, cluster_std=1.0, random_state=42
    )
    return X, y


@pytest.fixture
def blob_data_small():
    """Small blob dataset for fast tests."""
    X, y = make_blobs(
        n_samples=100, n_features=3, centers=3, cluster_std=0.5, random_state=42
    )
    return X, y


@pytest.fixture
def moon_data():
    """Two interleaving half-moons for density-based methods."""
    X, y = make_moons(n_samples=200, noise=0.05, random_state=42)
    return X, y


# ============================================================================
# Centroid-based
# ============================================================================


class TestKMeansClusterer:
    def test_fit_predict(self, blob_data):
        from endgame.clustering import KMeansClusterer

        X, _ = blob_data
        km = KMeansClusterer(n_clusters=3, random_state=42)
        labels = km.fit_predict(X)

        assert labels.shape == (len(X),)
        assert len(set(labels)) == 3

    def test_predict_new_data(self, blob_data):
        from endgame.clustering import KMeansClusterer

        X, _ = blob_data
        km = KMeansClusterer(n_clusters=3, random_state=42)
        km.fit(X)
        labels = km.predict(X[:10])
        assert labels.shape == (10,)

    def test_attributes(self, blob_data):
        from endgame.clustering import KMeansClusterer

        X, _ = blob_data
        km = KMeansClusterer(n_clusters=3, random_state=42)
        km.fit(X)
        assert km.cluster_centers_.shape == (3, 5)
        assert km.inertia_ > 0
        assert km.n_features_in_ == 5

    def test_transform(self, blob_data):
        from endgame.clustering import KMeansClusterer

        X, _ = blob_data
        km = KMeansClusterer(n_clusters=3, random_state=42)
        km.fit(X)
        dists = km.transform(X)
        assert dists.shape == (len(X), 3)


class TestMiniBatchKMeansClusterer:
    def test_fit_predict(self, blob_data):
        from endgame.clustering import MiniBatchKMeansClusterer

        X, _ = blob_data
        km = MiniBatchKMeansClusterer(n_clusters=3, random_state=42)
        labels = km.fit_predict(X)
        assert labels.shape == (len(X),)
        assert len(set(labels)) == 3

    def test_partial_fit(self, blob_data):
        from endgame.clustering import MiniBatchKMeansClusterer

        X, _ = blob_data
        km = MiniBatchKMeansClusterer(n_clusters=3, random_state=42)
        km.partial_fit(X[:100])
        km.partial_fit(X[100:])
        assert hasattr(km, "cluster_centers_")


class TestKStarMeansClusterer:
    def test_fit_predict_auto_k(self, blob_data):
        from endgame.clustering import KStarMeansClusterer

        X, _ = blob_data
        km = KStarMeansClusterer(k_init=2, k_max=10, random_state=42)
        labels = km.fit_predict(X)
        assert labels.shape == (len(X),)
        # Should find approximately 3 clusters
        assert 2 <= km.n_clusters_ <= 6

    def test_predict_new_data(self, blob_data):
        from endgame.clustering import KStarMeansClusterer

        X, _ = blob_data
        km = KStarMeansClusterer(random_state=42)
        km.fit(X)
        labels = km.predict(X[:10])
        assert labels.shape == (10,)

    def test_mdl_history(self, blob_data):
        from endgame.clustering import KStarMeansClusterer

        X, _ = blob_data
        km = KStarMeansClusterer(random_state=42)
        km.fit(X)
        assert len(km.mdl_history_) > 0


# ============================================================================
# Density-based
# ============================================================================


class TestDBSCANClusterer:
    def test_fit_predict(self, moon_data):
        from endgame.clustering import DBSCANClusterer

        X, _ = moon_data
        db = DBSCANClusterer(eps=0.2, min_samples=5)
        labels = db.fit_predict(X)
        assert labels.shape == (len(X),)

    def test_noise_detection(self, moon_data):
        from endgame.clustering import DBSCANClusterer

        X, _ = moon_data
        # Add outliers
        outliers = np.array([[5.0, 5.0], [-5.0, -5.0]])
        X_noisy = np.vstack([X, outliers])
        db = DBSCANClusterer(eps=0.2, min_samples=5)
        labels = db.fit_predict(X_noisy)
        # Outliers should be noise (-1)
        assert -1 in labels

    def test_attributes(self, moon_data):
        from endgame.clustering import DBSCANClusterer

        X, _ = moon_data
        db = DBSCANClusterer(eps=0.2, min_samples=5)
        db.fit(X)
        assert hasattr(db, "core_sample_indices_")
        assert hasattr(db, "n_clusters_")


class TestHDBSCANClusterer:
    def test_fit_predict(self, moon_data):
        from endgame.clustering import HDBSCANClusterer

        X, _ = moon_data
        hdb = HDBSCANClusterer(min_cluster_size=10)
        labels = hdb.fit_predict(X)
        assert labels.shape == (len(X),)

    def test_probabilities(self, moon_data):
        from endgame.clustering import HDBSCANClusterer

        X, _ = moon_data
        hdb = HDBSCANClusterer(min_cluster_size=10)
        hdb.fit(X)
        assert hasattr(hdb, "probabilities_")
        assert hdb.probabilities_.shape == (len(X),)

    def test_n_clusters(self, blob_data_small):
        from endgame.clustering import HDBSCANClusterer

        X, _ = blob_data_small
        hdb = HDBSCANClusterer(min_cluster_size=5)
        hdb.fit(X)
        assert hdb.n_clusters_ >= 1


class TestOPTICSClusterer:
    def test_fit_predict(self, moon_data):
        from endgame.clustering import OPTICSClusterer

        X, _ = moon_data
        optics = OPTICSClusterer(min_samples=10)
        labels = optics.fit_predict(X)
        assert labels.shape == (len(X),)

    def test_reachability(self, moon_data):
        from endgame.clustering import OPTICSClusterer

        X, _ = moon_data
        optics = OPTICSClusterer(min_samples=10)
        optics.fit(X)
        assert hasattr(optics, "reachability_")
        assert optics.reachability_.shape == (len(X),)


class TestDensityPeaksClusterer:
    def test_fit_predict(self, blob_data_small):
        from endgame.clustering import DensityPeaksClusterer

        X, _ = blob_data_small
        dpc = DensityPeaksClusterer(n_clusters=3, percent=2.0)
        labels = dpc.fit_predict(X)
        assert labels.shape == (len(X),)
        assert len(set(labels)) == 3

    def test_auto_k(self, blob_data_small):
        from endgame.clustering import DensityPeaksClusterer

        X, _ = blob_data_small
        dpc = DensityPeaksClusterer(n_clusters=None, percent=2.0)
        labels = dpc.fit_predict(X)
        assert labels.shape == (len(X),)
        assert dpc.n_clusters_ >= 1

    def test_rho_delta(self, blob_data_small):
        from endgame.clustering import DensityPeaksClusterer

        X, _ = blob_data_small
        dpc = DensityPeaksClusterer(n_clusters=3)
        dpc.fit(X)
        assert dpc.rho_.shape == (len(X),)
        assert dpc.delta_.shape == (len(X),)
        assert len(dpc.centers_) == 3


# ============================================================================
# Hierarchical
# ============================================================================


class TestAgglomerativeClusterer:
    def test_fit_predict(self, blob_data):
        from endgame.clustering import AgglomerativeClusterer

        X, _ = blob_data
        agg = AgglomerativeClusterer(n_clusters=3, linkage="ward")
        labels = agg.fit_predict(X)
        assert labels.shape == (len(X),)
        assert len(set(labels)) == 3

    def test_linkage_options(self, blob_data):
        from endgame.clustering import AgglomerativeClusterer

        X, _ = blob_data
        for linkage in ["ward", "average", "complete", "single"]:
            metric = "euclidean" if linkage == "ward" else "euclidean"
            agg = AgglomerativeClusterer(n_clusters=3, linkage=linkage, metric=metric)
            labels = agg.fit_predict(X)
            assert labels.shape == (len(X),)

    def test_distance_threshold(self, blob_data_small):
        from endgame.clustering import AgglomerativeClusterer

        X, _ = blob_data_small
        agg = AgglomerativeClusterer(
            n_clusters=None, distance_threshold=5.0, linkage="ward"
        )
        agg.fit(X)
        assert agg.n_clusters_ >= 1


# ============================================================================
# Distribution-based
# ============================================================================


class TestGaussianMixtureClusterer:
    def test_fit_predict(self, blob_data):
        from endgame.clustering import GaussianMixtureClusterer

        X, _ = blob_data
        gmm = GaussianMixtureClusterer(n_components=3, random_state=42)
        labels = gmm.fit_predict(X)
        assert labels.shape == (len(X),)
        assert len(set(labels)) == 3

    def test_predict_proba(self, blob_data):
        from endgame.clustering import GaussianMixtureClusterer

        X, _ = blob_data
        gmm = GaussianMixtureClusterer(n_components=3, random_state=42)
        gmm.fit(X)
        proba = gmm.predict_proba(X)
        assert proba.shape == (len(X), 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_bic_aic(self, blob_data):
        from endgame.clustering import GaussianMixtureClusterer

        X, _ = blob_data
        gmm = GaussianMixtureClusterer(n_components=3, random_state=42)
        gmm.fit(X)
        assert isinstance(gmm.bic_, float)
        assert isinstance(gmm.aic_, float)

    def test_select_n_components(self, blob_data):
        from endgame.clustering import GaussianMixtureClusterer

        X, _ = blob_data
        gmm = GaussianMixtureClusterer(random_state=42)
        best_k = gmm.select_n_components(X, k_range=range(1, 8))
        assert 1 <= best_k <= 7

    def test_covariance_types(self, blob_data_small):
        from endgame.clustering import GaussianMixtureClusterer

        X, _ = blob_data_small
        for cov_type in ["full", "tied", "diag", "spherical"]:
            gmm = GaussianMixtureClusterer(
                n_components=3, covariance_type=cov_type, random_state=42
            )
            labels = gmm.fit_predict(X)
            assert labels.shape == (len(X),)


class TestFuzzyCMeansClusterer:
    def test_fit_predict(self, blob_data_small):
        from endgame.clustering import FuzzyCMeansClusterer

        X, _ = blob_data_small
        fcm = FuzzyCMeansClusterer(n_clusters=3, random_state=42)
        labels = fcm.fit_predict(X)
        assert labels.shape == (len(X),)
        assert len(set(labels)) == 3

    def test_membership(self, blob_data_small):
        from endgame.clustering import FuzzyCMeansClusterer

        X, _ = blob_data_small
        fcm = FuzzyCMeansClusterer(n_clusters=3, random_state=42)
        fcm.fit(X)
        assert fcm.membership_.shape == (len(X), 3)
        # Membership should sum to ~1 for each point
        np.testing.assert_allclose(fcm.membership_.sum(axis=1), 1.0, atol=1e-4)

    def test_predict_memberships(self, blob_data_small):
        from endgame.clustering import FuzzyCMeansClusterer

        X, _ = blob_data_small
        fcm = FuzzyCMeansClusterer(n_clusters=3, random_state=42)
        fcm.fit(X)
        U = fcm.predict_memberships(X[:10])
        assert U.shape == (10, 3)
        np.testing.assert_allclose(U.sum(axis=1), 1.0, atol=1e-4)

    def test_fuzziness_parameter(self, blob_data_small):
        from endgame.clustering import FuzzyCMeansClusterer

        X, _ = blob_data_small
        # Higher m = softer assignments (more uniform membership)
        fcm_hard = FuzzyCMeansClusterer(n_clusters=3, m=1.5, random_state=42)
        fcm_soft = FuzzyCMeansClusterer(n_clusters=3, m=4.0, random_state=42)
        fcm_hard.fit(X)
        fcm_soft.fit(X)
        # Soft should have more entropy (more uniform membership)
        entropy_hard = -np.sum(fcm_hard.membership_ * np.log(fcm_hard.membership_ + 1e-10))
        entropy_soft = -np.sum(fcm_soft.membership_ * np.log(fcm_soft.membership_ + 1e-10))
        assert entropy_soft > entropy_hard


# ============================================================================
# Graph/Spectral
# ============================================================================


class TestSpectralClusterer:
    def test_fit_predict(self, moon_data):
        from endgame.clustering import SpectralClusterer

        X, _ = moon_data
        sc = SpectralClusterer(n_clusters=2, random_state=42)
        labels = sc.fit_predict(X)
        assert labels.shape == (len(X),)
        assert len(set(labels)) == 2

    def test_affinity_options(self, blob_data_small):
        from endgame.clustering import SpectralClusterer

        X, _ = blob_data_small
        for affinity in ["rbf", "nearest_neighbors"]:
            sc = SpectralClusterer(n_clusters=3, affinity=affinity, random_state=42)
            labels = sc.fit_predict(X)
            assert labels.shape == (len(X),)


class TestAffinityPropagationClusterer:
    def test_fit_predict(self, blob_data_small):
        from endgame.clustering import AffinityPropagationClusterer

        X, _ = blob_data_small
        ap = AffinityPropagationClusterer(random_state=42)
        labels = ap.fit_predict(X)
        assert labels.shape == (len(X),)
        assert ap.n_clusters_ >= 1

    def test_predict_new_data(self, blob_data_small):
        from endgame.clustering import AffinityPropagationClusterer

        X, _ = blob_data_small
        ap = AffinityPropagationClusterer(random_state=42)
        ap.fit(X)
        labels = ap.predict(X[:10])
        assert labels.shape == (10,)

    def test_exemplars(self, blob_data_small):
        from endgame.clustering import AffinityPropagationClusterer

        X, _ = blob_data_small
        ap = AffinityPropagationClusterer(random_state=42)
        ap.fit(X)
        assert len(ap.cluster_centers_indices_) == ap.n_clusters_
        assert ap.cluster_centers_.shape[0] == ap.n_clusters_


# ============================================================================
# Scalable
# ============================================================================


class TestBIRCHClusterer:
    def test_fit_predict(self, blob_data):
        from endgame.clustering import BIRCHClusterer

        X, _ = blob_data
        birch = BIRCHClusterer(n_clusters=3)
        labels = birch.fit_predict(X)
        assert labels.shape == (len(X),)

    def test_predict_new_data(self, blob_data):
        from endgame.clustering import BIRCHClusterer

        X, _ = blob_data
        birch = BIRCHClusterer(n_clusters=3)
        birch.fit(X)
        labels = birch.predict(X[:10])
        assert labels.shape == (10,)

    def test_partial_fit(self, blob_data):
        from endgame.clustering import BIRCHClusterer

        X, _ = blob_data
        birch = BIRCHClusterer(n_clusters=3)
        birch.partial_fit(X[:100])
        birch.partial_fit(X[100:])
        assert hasattr(birch, "model_")


class TestMeanShiftClusterer:
    def test_fit_predict(self, blob_data_small):
        from endgame.clustering import MeanShiftClusterer

        X, _ = blob_data_small
        ms = MeanShiftClusterer()
        labels = ms.fit_predict(X)
        assert labels.shape == (len(X),)
        assert ms.n_clusters_ >= 1

    def test_predict_new_data(self, blob_data_small):
        from endgame.clustering import MeanShiftClusterer

        X, _ = blob_data_small
        ms = MeanShiftClusterer()
        ms.fit(X)
        labels = ms.predict(X[:10])
        assert labels.shape == (10,)


# ============================================================================
# AutoCluster
# ============================================================================


class TestAutoCluster:
    def test_auto_k(self, blob_data):
        from endgame.clustering import AutoCluster

        X, _ = blob_data
        ac = AutoCluster(n_clusters="auto", random_state=42)
        labels = ac.fit_predict(X)
        assert labels.shape == (len(X),)
        assert ac.n_clusters_ >= 1
        assert ac.selected_method_ is not None

    def test_specified_k(self, blob_data):
        from endgame.clustering import AutoCluster

        X, _ = blob_data
        ac = AutoCluster(n_clusters=3, random_state=42)
        labels = ac.fit_predict(X)
        assert labels.shape == (len(X),)
        assert len(set(labels)) == 3

    def test_detect_noise(self, moon_data):
        from endgame.clustering import AutoCluster

        X, _ = moon_data
        ac = AutoCluster(n_clusters="auto", detect_noise=True)
        labels = ac.fit_predict(X)
        assert labels.shape == (len(X),)
        # Should select a density-based method
        assert ac.selected_method_ in ("hdbscan", "dbscan")

    def test_prefer_centroid(self, blob_data):
        from endgame.clustering import AutoCluster

        X, _ = blob_data
        ac = AutoCluster(n_clusters=3, prefer="centroid", random_state=42)
        ac.fit(X)
        assert ac.selected_method_ == "kmeans"

    def test_prefer_density(self, blob_data):
        from endgame.clustering import AutoCluster

        X, _ = blob_data
        ac = AutoCluster(n_clusters="auto", prefer="density")
        ac.fit(X)
        assert ac.selected_method_ == "hdbscan"

    def test_prefer_hierarchical(self, blob_data):
        from endgame.clustering import AutoCluster

        X, _ = blob_data
        ac = AutoCluster(n_clusters=3, prefer="hierarchical")
        ac.fit(X)
        assert ac.selected_method_ == "agglomerative"

    def test_prefer_distribution(self, blob_data):
        from endgame.clustering import AutoCluster

        X, _ = blob_data
        ac = AutoCluster(n_clusters=3, prefer="distribution", random_state=42)
        ac.fit(X)
        assert ac.selected_method_ == "gmm"

    def test_invalid_preference(self, blob_data):
        from endgame.clustering import AutoCluster

        X, _ = blob_data
        ac = AutoCluster(prefer="invalid")
        with pytest.raises(ValueError, match="Unknown preference"):
            ac.fit(X)

    def test_verbose(self, blob_data, capsys):
        from endgame.clustering import AutoCluster

        X, _ = blob_data
        ac = AutoCluster(n_clusters=3, verbose=True, random_state=42)
        ac.fit(X)
        captured = capsys.readouterr()
        assert "Selected method" in captured.out


# ============================================================================
# Module imports
# ============================================================================


class TestModuleImports:
    def test_import_all(self):
        from endgame.clustering import (
            KMeansClusterer,
            MiniBatchKMeansClusterer,
            KStarMeansClusterer,
            DBSCANClusterer,
            HDBSCANClusterer,
            OPTICSClusterer,
            DensityPeaksClusterer,
            AgglomerativeClusterer,
            GaussianMixtureClusterer,
            FuzzyCMeansClusterer,
            SpectralClusterer,
            AffinityPropagationClusterer,
            BIRCHClusterer,
            MeanShiftClusterer,
            AutoCluster,
        )
        assert KMeansClusterer is not None
        assert AutoCluster is not None

    def test_import_via_endgame(self):
        import endgame as eg

        assert hasattr(eg, "clustering")
        assert hasattr(eg.clustering, "KMeansClusterer")
        assert hasattr(eg.clustering, "AutoCluster")
        assert hasattr(eg.clustering, "HDBSCANClusterer")
