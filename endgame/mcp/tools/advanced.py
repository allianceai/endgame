"""Advanced tools: cluster, detect_anomalies, forecast."""

from __future__ import annotations

import json

from mcp.server.fastmcp import FastMCP

from endgame.mcp.server import capture_stdout, error_response, ok_response
from endgame.mcp.session import SessionManager


def register(mcp: FastMCP, session: SessionManager) -> None:

    @mcp.tool()
    def cluster(
        dataset_id: str,
        method: str = "auto",
        n_clusters: int | None = None,
        params: str | dict | None = None,
    ) -> str:
        """Cluster a dataset. Methods: auto, kmeans, hdbscan, dbscan, agglomerative, gaussian_mixture.

        params: optional dict or JSON string of method-specific parameters.
        """
        try:
            ds = session.get_dataset(dataset_id)
            if isinstance(params, dict):
                extra = params
            elif isinstance(params, str):
                extra = json.loads(params)
            else:
                extra = {}

            with capture_stdout():
                import numpy as np

                X = ds.df.select_dtypes(include="number")
                if ds.target_column and ds.target_column in X.columns:
                    X = X.drop(columns=[ds.target_column])

                if X.isna().any().any():
                    X = X.fillna(X.median())

                from sklearn.preprocessing import StandardScaler
                X_scaled = StandardScaler().fit_transform(X)

                if method == "auto" or method == "kmeans":
                    from sklearn.cluster import KMeans
                    k = n_clusters or 5
                    model = KMeans(n_clusters=k, random_state=42, n_init=10, **extra)
                    labels = model.fit_predict(X_scaled)

                elif method == "hdbscan":
                    try:
                        from sklearn.cluster import HDBSCAN
                        model = HDBSCAN(**extra)
                        labels = model.fit_predict(X_scaled)
                    except ImportError:
                        return error_response("missing_dependency", "HDBSCAN requires scikit-learn >= 1.3")

                elif method == "dbscan":
                    from sklearn.cluster import DBSCAN
                    model = DBSCAN(**extra)
                    labels = model.fit_predict(X_scaled)

                elif method == "agglomerative":
                    from sklearn.cluster import AgglomerativeClustering
                    k = n_clusters or 5
                    model = AgglomerativeClustering(n_clusters=k, **extra)
                    labels = model.fit_predict(X_scaled)

                elif method == "gaussian_mixture":
                    from sklearn.mixture import GaussianMixture
                    k = n_clusters or 5
                    model = GaussianMixture(n_components=k, random_state=42, **extra)
                    labels = model.fit_predict(X_scaled)

                else:
                    return error_response(
                        "validation",
                        f"Unknown clustering method: {method}",
                        hint="Use: auto, kmeans, hdbscan, dbscan, agglomerative, gaussian_mixture",
                    )

                # Add cluster labels to dataset
                new_df = ds.df.copy()
                new_df["cluster"] = labels

                art = session.add_dataset(
                    df=new_df,
                    name=f"{ds.name}_clustered",
                    source="derived",
                    target_column=ds.target_column,
                    task_type=ds.task_type,
                )

                unique, counts = np.unique(labels[labels >= 0], return_counts=True)
                cluster_sizes = {int(u): int(c) for u, c in zip(unique, counts)}
                n_noise = int(np.sum(labels == -1))

                return ok_response({
                    "dataset_id": art.id,
                    "method": method,
                    "n_clusters": len(unique),
                    "cluster_sizes": cluster_sizes,
                    "n_noise_points": n_noise,
                })

        except KeyError as e:
            return error_response("not_found", str(e))
        except json.JSONDecodeError:
            return error_response("validation", "Invalid JSON for params")
        except Exception as e:
            return error_response("internal", str(e))

    @mcp.tool()
    def detect_anomalies(
        dataset_id: str,
        method: str = "isolation_forest",
        contamination: float = 0.05,
    ) -> str:
        """Detect anomalies/outliers in a dataset. Methods: isolation_forest, lof, elliptic_envelope."""
        try:
            ds = session.get_dataset(dataset_id)

            with capture_stdout():

                X = ds.df.select_dtypes(include="number")
                if ds.target_column and ds.target_column in X.columns:
                    X = X.drop(columns=[ds.target_column])

                if X.isna().any().any():
                    X = X.fillna(X.median())

                if method == "isolation_forest":
                    from sklearn.ensemble import IsolationForest
                    detector = IsolationForest(contamination=contamination, random_state=42)
                elif method == "lof":
                    from sklearn.neighbors import LocalOutlierFactor
                    detector = LocalOutlierFactor(contamination=contamination)
                elif method == "elliptic_envelope":
                    from sklearn.covariance import EllipticEnvelope
                    detector = EllipticEnvelope(contamination=contamination, random_state=42)
                else:
                    return error_response(
                        "validation",
                        f"Unknown method: {method}",
                        hint="Use: isolation_forest, lof, elliptic_envelope",
                    )

                if method == "lof":
                    labels = detector.fit_predict(X)
                else:
                    labels = detector.fit_predict(X)

                # -1 = outlier, 1 = inlier
                is_outlier = labels == -1
                n_outliers = int(is_outlier.sum())

                new_df = ds.df.copy()
                new_df["is_anomaly"] = is_outlier.astype(int)

                art = session.add_dataset(
                    df=new_df,
                    name=f"{ds.name}_anomaly_scored",
                    source="derived",
                    target_column=ds.target_column,
                    task_type=ds.task_type,
                )

                return ok_response({
                    "dataset_id": art.id,
                    "method": method,
                    "contamination": contamination,
                    "n_samples": len(ds.df),
                    "n_anomalies": n_outliers,
                    "anomaly_pct": round(n_outliers / len(ds.df) * 100, 2),
                })

        except KeyError as e:
            return error_response("not_found", str(e))
        except Exception as e:
            return error_response("internal", str(e))

    @mcp.tool()
    def forecast(
        dataset_id: str,
        target_column: str,
        horizon: int = 12,
        method: str = "auto",
        seasonal_period: int | None = None,
    ) -> str:
        """Forecast a time series. Methods: auto, arima, ets, theta, naive.

        The dataset should have a time-ordered column as the target.
        """
        try:
            ds = session.get_dataset(dataset_id)

            with capture_stdout():
                import numpy as np
                import pandas as pd

                if target_column not in ds.df.columns:
                    return error_response("validation", f"Column '{target_column}' not found")

                y = ds.df[target_column].values.astype(float)

                if method == "auto" or method == "arima":
                    try:
                        from statsforecast import StatsForecast
                        from statsforecast.models import AutoARIMA
                        sf_df = pd.DataFrame({
                            "unique_id": ["series"] * len(y),
                            "ds": pd.date_range("2000-01-01", periods=len(y), freq="D"),
                            "y": y,
                        })
                        sf = StatsForecast(
                            models=[AutoARIMA(season_length=seasonal_period or 1)],
                            freq="D",
                        )
                        sf.fit(sf_df)
                        fc = sf.predict(h=horizon)
                        predictions = fc["AutoARIMA"].values.tolist()
                        method_used = "AutoARIMA"
                    except ImportError:
                        return error_response(
                            "missing_dependency",
                            "statsforecast is required for ARIMA forecasting. "
                            "Install with: pip install statsforecast",
                            hint="Use method='naive' for a simple last-value forecast without extra dependencies.",
                        )

                elif method == "ets":
                    try:
                        from statsforecast import StatsForecast
                        from statsforecast.models import AutoETS
                        sf_df = pd.DataFrame({
                            "unique_id": ["series"] * len(y),
                            "ds": pd.date_range("2000-01-01", periods=len(y), freq="D"),
                            "y": y,
                        })
                        sf = StatsForecast(
                            models=[AutoETS(season_length=seasonal_period or 1)],
                            freq="D",
                        )
                        sf.fit(sf_df)
                        fc = sf.predict(h=horizon)
                        predictions = fc["AutoETS"].values.tolist()
                        method_used = "AutoETS"
                    except ImportError:
                        return error_response(
                            "missing_dependency",
                            "statsforecast is required for ETS forecasting. "
                            "Install with: pip install statsforecast",
                            hint="Use method='naive' for a simple last-value forecast without extra dependencies.",
                        )

                elif method == "theta":
                    try:
                        from statsforecast import StatsForecast
                        from statsforecast.models import Theta
                        sf_df = pd.DataFrame({
                            "unique_id": ["series"] * len(y),
                            "ds": pd.date_range("2000-01-01", periods=len(y), freq="D"),
                            "y": y,
                        })
                        sf = StatsForecast(models=[Theta()], freq="D")
                        sf.fit(sf_df)
                        fc = sf.predict(h=horizon)
                        predictions = fc["Theta"].values.tolist()
                        method_used = "Theta"
                    except ImportError:
                        return error_response(
                            "missing_dependency",
                            "statsforecast is required for Theta forecasting. "
                            "Install with: pip install statsforecast",
                            hint="Use method='naive' for a simple last-value forecast without extra dependencies.",
                        )

                elif method == "naive":
                    predictions = [float(y[-1])] * horizon
                    method_used = "naive"

                else:
                    return error_response(
                        "validation",
                        f"Unknown method: {method}",
                        hint="Use: auto, arima, ets, theta, naive",
                    )

                return ok_response({
                    "method": method_used,
                    "horizon": horizon,
                    "predictions": [round(p, 4) for p in predictions],
                    "history_length": len(y),
                    "last_value": round(float(y[-1]), 4),
                })

        except KeyError as e:
            return error_response("not_found", str(e))
        except Exception as e:
            return error_response("internal", str(e))
