"""Feature tools: engineer_features, select_features, transform_data."""

from __future__ import annotations

import inspect
import json
from typing import Any

import numpy as np
import pandas as pd
from mcp.server.fastmcp import FastMCP

from endgame.mcp.server import capture_stdout, error_response, ok_response
from endgame.mcp.session import SessionManager
from endgame.mcp.tools._timeout import MCPTimeoutError, timeout_guard

# ---------------------------------------------------------------------------
# Aggregations: pandas reductions plus per-group signal features from endgame.signal
# ---------------------------------------------------------------------------

_PANDAS_AGGS = {"mean", "std", "min", "max", "median", "sum", "count", "nunique", "first", "last", "skew", "var"}


def _signal_aggs(fs: float) -> dict:
    from endgame import signal as sig

    def demeaned(x):
        return x - x.mean()

    def dominant_freq(x):
        if len(x) < 4:
            return np.nan
        power = np.abs(np.fft.rfft(demeaned(x))) ** 2
        return float(np.fft.rfftfreq(len(x), d=1 / fs)[1:][np.argmax(power[1:])])

    def autocorr1(x):
        return float(pd.Series(x).autocorr(1)) if len(x) > 2 else np.nan

    return {
        "range": lambda x: float(x.max() - x.min()),
        "iqr": lambda x: float(np.subtract(*np.percentile(x, [75, 25]))),
        "kurt": lambda x: float(pd.Series(x).kurt()),
        "rms": lambda x: float(sig.compute_rms(x)),
        "line_length": lambda x: float(sig.compute_line_length(x)),
        "zero_crossings": lambda x: float(sig.count_zero_crossings(demeaned(x))),
        "sample_entropy": lambda x: float(sig.sample_entropy(x)) if len(x) > 10 else np.nan,
        "permutation_entropy": lambda x: float(sig.permutation_entropy(x)) if len(x) > 5 else np.nan,
        "svd_entropy": lambda x: float(sig.svd_entropy(x)) if len(x) > 5 else np.nan,
        "spectral_entropy": lambda x: float(sig.spectral_entropy(x, fs=fs)) if len(x) > 8 else np.nan,
        "higuchi_fd": lambda x: float(sig.higuchi_fd(x)) if len(x) > 20 else np.nan,
        "katz_fd": lambda x: float(sig.katz_fd(x)) if len(x) > 3 else np.nan,
        "hurst": lambda x: float(sig.hurst_exponent(x)) if len(x) > 20 else np.nan,
        "dfa": lambda x: float(sig.detrended_fluctuation(x)) if len(x) > 20 else np.nan,
        "lempel_ziv": lambda x: float(sig.lempel_ziv_complexity(x)) if len(x) > 5 else np.nan,
        "hjorth_mobility": lambda x: float(sig.compute_hjorth(x)[1]) if len(x) > 3 else np.nan,
        "hjorth_complexity": lambda x: float(sig.compute_hjorth(x)[2]) if len(x) > 3 else np.nan,
        "dominant_freq": dominant_freq,
        "autocorr1": autocorr1,
    }


AGGREGATIONS_HELP = (
    "mean, std, var, min, max, median, sum, count, nunique, first, last, skew, kurt, range, iqr, q05..q95 "
    "(any qNN), and signal features (order rows with order_by; sampling rate fs, default 10 Hz): rms, "
    "line_length, zero_crossings, sample_entropy, permutation_entropy, svd_entropy, spectral_entropy, "
    "higuchi_fd, katz_fd, hurst, dfa, lempel_ziv, hjorth_mobility, hjorth_complexity, dominant_freq, autocorr1"
)


def _aggregate(df: pd.DataFrame, by: list[str], columns: list[str], aggs: list[str], prefix: str,
               order_by: str | None, fs: float) -> pd.DataFrame:
    if order_by:
        df = df.sort_values(by + [order_by])
    grouped = df.groupby(by, sort=False, observed=True)
    custom = _signal_aggs(fs)
    parts = []
    for agg in aggs:
        if agg in _PANDAS_AGGS:
            out = grouped[columns].agg(agg)
        elif agg.startswith("q") and agg[1:].isdigit():
            out = grouped[columns].quantile(int(agg[1:]) / 100)
        elif agg in custom:
            fn = custom[agg]
            out = grouped[columns].agg(lambda s, fn=fn: fn(s.dropna().to_numpy(dtype=float)))
        else:
            raise ValueError(f"Unknown aggregation '{agg}'. Available: {AGGREGATIONS_HELP}")
        out.columns = [f"{prefix}{c}_{agg}" for c in columns]
        parts.append(out)
    return pd.concat(parts, axis=1).reset_index()


def _as_list(value) -> list:
    if value is None:
        return []
    return [value] if isinstance(value, str) else list(value)


def _to_pandas(out, index, prefix: str) -> pd.DataFrame:
    if hasattr(out, "to_pandas"):  # polars
        out = out.to_pandas()
    if isinstance(out, pd.DataFrame):
        out = out.copy()
        out.index = index
        return out
    arr = np.asarray(out)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return pd.DataFrame(arr, index=index, columns=[f"{prefix}{i}" for i in range(arr.shape[1])])


def _engineer(session: SessionManager, ds, df: pd.DataFrame, op: dict) -> tuple[pd.DataFrame, str, list[str]]:
    """Apply one engineer_features operation. Returns the new frame, a description and warnings."""
    kind = op.get("type")
    target = ds.target_column
    warnings: list[str] = []

    if kind == "aggregate":
        by = _as_list(op.get("by"))
        source_id = op.get("source")
        src = session.get_dataset(source_id).df if source_id else df
        if op.get("filter"):
            src = src.query(op["filter"])
        numeric = [c for c in src.select_dtypes(include="number").columns if c not in by and c != target]
        columns = _as_list(op.get("columns")) or numeric
        aggs = _as_list(op.get("aggs")) or ["mean", "std", "min", "max"]
        agg_df = _aggregate(src, by, columns, aggs, op.get("prefix", ""), op.get("order_by"),
                            float(op.get("fs", 10.0)))
        if source_id and source_id != ds.id:
            missing = [k for k in by if k not in df.columns]
            if missing:
                raise ValueError(f"aggregate: keys {missing} are not columns of {ds.id}")
            new = df.merge(agg_df, on=by, how="left")
            unmatched = int(new[agg_df.columns.difference(by)[0]].isna().sum()) if len(agg_df.columns) > len(by) else 0
            if unmatched:
                warnings.append(f"aggregate: {unmatched} of {len(new)} rows had no rows in {source_id}")
            return new, f"aggregate({source_id} by {by}: {len(agg_df.columns) - len(by)} features)", warnings
        if target and target in df.columns and target not in agg_df.columns:
            agg_df = agg_df.merge(df.groupby(by, sort=False, observed=True)[target].first().reset_index(), on=by)
        return agg_df, f"aggregate(by {by}: {len(agg_df)} rows, {len(agg_df.columns) - len(by)} features)", warnings

    if kind == "join":
        other = session.get_dataset(op["other"]).df
        on = _as_list(op.get("on"))
        cols = _as_list(op.get("columns"))
        if cols:
            other = other[list(dict.fromkeys(on + cols))]
        new = df.merge(other, on=on, how=op.get("how", "left"), suffixes=("", op.get("suffix", "_other")))
        if len(new) != len(df) and op.get("how", "left") == "left":
            warnings.append(f"join: {len(df)} rows became {len(new)}: keys {on} are not unique in {op['other']}")
        return new, f"join({op['other']} on {on})", warnings

    if kind == "group_normalize":
        by = _as_list(op.get("by"))
        method = op.get("method", "zscore")
        columns = _as_list(op.get("columns")) or [c for c in df.select_dtypes(include="number").columns
                                                   if c not in by and c != target]
        g = df.groupby(by, observed=True)[columns]
        tag = "_".join(by)
        if method == "zscore":
            out = (df[columns] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
        elif method == "rank":
            out = g.rank(pct=True)
        elif method == "diff_mean":
            out = df[columns] - g.transform("mean")
        elif method == "ratio_mean":
            out = df[columns] / g.transform("mean").replace(0, np.nan)
        else:
            raise ValueError("group_normalize method: zscore, rank, diff_mean or ratio_mean")
        out.columns = [f"{c}_{method}_by_{tag}" for c in columns]
        return pd.concat([df, out], axis=1), f"group_normalize({method} by {by}, {len(columns)} columns)", warnings

    if kind == "formula":
        new = df.copy()
        new[op["name"]] = new.eval(op["expr"])
        return new, f"formula({op['name']} = {op['expr']})", warnings

    if kind == "drop":
        return df.drop(columns=[c for c in _as_list(op.get("columns")) if c in df.columns]), "drop", warnings

    if kind == "datetime":
        new = df.copy()
        for col in _as_list(op.get("columns")):
            t = pd.to_datetime(new[col], errors="coerce")
            for part in op.get("parts", ["year", "month", "day", "dayofweek", "hour", "dayofyear"]):
                new[f"{col}_{part}"] = getattr(t.dt, part)
        return new, f"datetime({op.get('columns')})", warnings

    # Transformers from endgame.preprocessing
    from endgame import preprocessing as pp

    features = [c for c in df.columns if c != target]
    columns = _as_list(op.get("columns"))
    if kind == "interactions":
        cols = columns or list(df[features].select_dtypes(include="number").columns)
        tr = pp.InteractionFeatures(include_cols=cols, operations=tuple(op.get("operations", ("multiply", "divide"))),
                                    max_interactions=int(op.get("max_interactions", 100)), output_format="pandas")
        out = _to_pandas(tr.fit_transform(df[cols]), df.index, "ix_")
        new_cols = [c for c in out.columns if c not in df.columns]
        return pd.concat([df, out[new_cols]], axis=1), f"interactions({len(new_cols)} features)", warnings
    if kind in ("lags", "rolling"):
        by = _as_list(op.get("by"))
        if op.get("order_by"):
            df = df.sort_values(by + [op["order_by"]]).reset_index(drop=True)
        if kind == "lags":
            tr = pp.LagFeatures(cols=columns or None, lags=tuple(op.get("lags", (1, 2, 3))), group_cols=by or None,
                                output_format="pandas")
        else:
            tr = pp.RollingFeatures(cols=columns or None, windows=tuple(op.get("windows", (3, 7))),
                                    methods=tuple(op.get("methods", ("mean", "std"))), group_cols=by or None,
                                    output_format="pandas")
        out = _to_pandas(tr.fit_transform(df), df.index, f"{kind}_")
        new_cols = [c for c in out.columns if c not in df.columns]
        return pd.concat([df, out[new_cols]], axis=1), f"{kind}({len(new_cols)} features)", warnings
    if kind == "target_encode":
        if not target:
            raise ValueError("target_encode needs a dataset with a target column")
        cols = columns or list(df[features].select_dtypes(include=["object", "category"]).columns)
        tr = pp.SafeTargetEncoder(cols=cols, smoothing=float(op.get("smoothing", 10.0)), cv=int(op.get("cv", 5)),
                                  output_format="pandas", random_state=42)
        out = _to_pandas(tr.fit_transform(df[cols], df[target]), df.index, "te_")
        new = df.copy()
        for c in cols:
            new[f"{c}_te"] = out[c].to_numpy() if c in out.columns else np.nan
        return new, f"target_encode({cols}, out-of-fold)", warnings
    if kind == "frequency_encode":
        cols = columns or list(df[features].select_dtypes(include=["object", "category"]).columns)
        out = _to_pandas(pp.FrequencyEncoder(cols=cols, output_format="pandas").fit_transform(df[cols]), df.index,
                         "freq_")
        new = df.copy()
        for c in cols:
            new[f"{c}_freq"] = out[c].to_numpy()
        return new, f"frequency_encode({cols})", warnings
    if kind == "rank":
        by = _as_list(op.get("by"))
        cols = columns or list(df[features].select_dtypes(include="number").columns)
        ranks = df.groupby(by, observed=True)[cols].rank(pct=True) if by else df[cols].rank(pct=True)
        ranks.columns = [f"{c}_rank" + (f"_by_{'_'.join(by)}" if by else "") for c in cols]
        return pd.concat([df, ranks], axis=1), f"rank({len(cols)} columns)", warnings
    if kind == "auto_aggregate":
        by = _as_list(op.get("by"))
        tr = pp.AutoAggregator(group_cols=by, agg_cols=columns or None,
                               methods=tuple(op.get("methods", ("mean", "std", "min", "max"))),
                               output_format="pandas")
        out = _to_pandas(tr.fit_transform(df[features]), df.index, "agg_")
        new_cols = [c for c in out.columns if c not in df.columns]
        return pd.concat([df, out[new_cols]], axis=1), f"auto_aggregate(by {by}: {len(new_cols)} features)", warnings

    raise ValueError(
        f"Unknown operation '{kind}'. Operations: aggregate, join, group_normalize, formula, interactions, lags, "
        "rolling, target_encode, frequency_encode, datetime, rank, auto_aggregate, drop")


# ---------------------------------------------------------------------------
# Feature selection over endgame.feature_selection
# ---------------------------------------------------------------------------

SELECTORS = {
    "mrmr": "MRMRSelector", "mutual_info": "MutualInfoSelector", "f_test": "FTestSelector", "chi2": "Chi2Selector",
    "univariate": "UnivariateSelector", "relieff": "ReliefFSelector", "correlation": "CorrelationSelector",
    "rfe": "RFESelector", "boruta": "BorutaSelector", "sequential": "SequentialSelector",
    "genetic": "GeneticSelector", "permutation": "PermutationSelector", "shap": "SHAPSelector",
    "tree_importance": "TreeImportanceSelector", "stability": "StabilitySelector", "knockoff": "KnockoffSelector",
    "null_importance": "NullImportanceSelector", "variance": None,
}


def _default_estimator(task: str):
    import lightgbm as lgb

    cls = lgb.LGBMRegressor if task == "regression" else lgb.LGBMClassifier
    return cls(n_estimators=200, learning_rate=0.05, num_leaves=15, verbose=-1, random_state=42)


def _build_selector(method: str, task: str, n_features: int | None, params: dict):
    from endgame import feature_selection as fsel

    if method not in SELECTORS:
        raise ValueError(f"Unknown method '{method}'. Methods: {', '.join(SELECTORS)}")
    if method == "variance":
        from sklearn.feature_selection import VarianceThreshold
        return VarianceThreshold(**params)
    cls = getattr(fsel, SELECTORS[method])
    sig = inspect.signature(cls).parameters
    kw: dict[str, Any] = {}
    if "task" in sig:
        kw["task"] = "regression" if task == "regression" else "classification"
    if n_features is not None:
        for name in ("n_features", "k", "max_features"):
            if name in sig:
                kw[name] = n_features
                break
    if "estimator" in sig and "estimator" not in params:
        kw["estimator"] = _default_estimator(task)
    if method == "stability" and "base_selector" not in params:
        kw["base_selector"] = fsel.MRMRSelector(n_features=n_features or 10, task=kw.get("task", "classification"))
    if "random_state" in sig:
        kw["random_state"] = 42
    return cls(**{**kw, **params})


def _selected(selector, columns: list[str]) -> list[str]:
    if hasattr(selector, "get_support"):
        try:
            return [c for c, keep in zip(columns, selector.get_support()) if keep]
        except Exception:
            pass
    chosen = getattr(selector, "selected_features_", None)
    if chosen is None:
        raise ValueError(f"{type(selector).__name__} does not report which features it kept")
    chosen = list(chosen)
    return [columns[i] for i in chosen] if chosen and isinstance(chosen[0], (int, np.integer)) else chosen


def _scores(selector, columns: list[str]) -> dict[str, float]:
    for attr in ("scores_", "feature_importances_", "importances_"):
        values = getattr(selector, attr, None)
        if values is None:
            continue
        values = np.asarray(values, dtype=float).ravel()
        if len(values) == len(columns):
            order = np.argsort(-np.nan_to_num(values, nan=-np.inf))[:30]
            return {columns[i]: round(float(values[i]), 5) for i in order}
    return {}


def _selection_matrix(ds) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    from endgame.mcp.tools._encoding import fit_feature_encoders, identifier_columns

    X = ds.df.drop(columns=[ds.target_column])
    ids = identifier_columns(X)
    X, _ = fit_feature_encoders(X.drop(columns=ids))
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    y = ds.df[ds.target_column]
    if ds.task_type != "regression" and not pd.api.types.is_numeric_dtype(y):
        y = pd.Series(pd.factorize(y)[0], index=y.index, name=y.name)
    return X, y, ids


ENGINEER_DOC = """Build features; returns a new dataset. operations: list of dicts applied in order, each with a "type":
        - aggregate: per-entity stats of a long table, joined on: {"type":"aggregate","source":"<long ds id>",
          "by":["player_id"],"columns":["s","a"],"aggs":["mean","max","q90","sample_entropy"],"filter":"drill=='x'",
          "order_by":"time","fs":10,"prefix":"x_"}. Without "source" the dataset itself collapses to one row per key.
          Aggs: {aggs}.
        - join: {"type":"join","other":"<ds id>","on":["player_id"],"how":"left","columns":[...]}
        - group_normalize: {"type":"group_normalize","by":"position","columns":[...],"method":"zscore|rank|diff_mean|ratio_mean"}
        - formula: {"type":"formula","name":"bmi","expr":"weight / height ** 2"}
        - interactions (columns, operations multiply/divide/add/subtract, max_interactions), lags (columns, lags, by,
          order_by), rolling (columns, windows, methods, by, order_by), target_encode (out-of-fold; columns),
          frequency_encode, datetime (columns, parts), rank (columns, by), auto_aggregate (by, columns, methods), drop.
        guide("features") has recipes.""".replace("{aggs}", AGGREGATIONS_HELP)


def register(mcp: FastMCP, session: SessionManager) -> None:

    @mcp.tool(description=ENGINEER_DOC)
    def engineer_features(dataset_id: str, operations: str | list) -> str:
        try:
            ds = session.get_dataset(dataset_id)
            ops = operations if isinstance(operations, list) else json.loads(operations)
            with capture_stdout(), timeout_guard():
                df = ds.df.copy()
                applied, warnings = [], []
                for op in ops:
                    df, done, warn = _engineer(session, ds, df, op)
                    applied.append(done)
                    warnings.extend(warn)
                target = ds.target_column if ds.target_column in df.columns else None
                art = session.add_dataset(df=df.reset_index(drop=True), name=f"{ds.name}_features", source="derived",
                                          target_column=target, task_type=ds.task_type if target else None)
                new_cols = [c for c in df.columns if c not in ds.df.columns]
                return ok_response({
                    "dataset_id": art.id, "shape": list(df.shape), "operations_applied": applied,
                    "n_new_columns": len(new_cols), "new_columns": new_cols[:60], "warnings": warnings,
                })
        except MCPTimeoutError as e:
            return error_response("timeout", str(e))
        except (KeyError, ValueError) as e:
            return error_response("validation", str(e))
        except json.JSONDecodeError:
            return error_response("validation", "operations must be a list or a JSON array")
        except Exception as e:
            return error_response("internal", f"{type(e).__name__}: {e}")

    @mcp.tool()
    def select_features(
        dataset_id: str,
        method: str = "mrmr",
        n_features: int | None = None,
        params: str | dict | None = None,
        apply_to: list[str] | None = None,
    ) -> str:
        """Select features with endgame.feature_selection; returns a new dataset with the kept columns.
        method: mrmr, mutual_info, f_test, chi2, univariate, relieff, correlation (drops redundant), variance, rfe,
        sequential, genetic, boruta, permutation, shap, null_importance, tree_importance, stability, knockoff.
        Estimator-based methods default to a small LightGBM. params: constructor overrides (describe_api shows them).
        Selecting on every row makes later CV optimistic: select on a training split and pass apply_to=[test ids]."""
        try:
            ds = session.get_dataset(dataset_id)
            if not ds.target_column:
                return error_response("validation", "Dataset has no target column set")
            extra = params if isinstance(params, dict) else (json.loads(params) if params else {})
            with capture_stdout(), timeout_guard():
                X, y, ids = _selection_matrix(ds)
                selector = _build_selector(method, ds.task_type or "classification", n_features, extra)
                selector.fit(X, y)
                kept = _selected(selector, list(X.columns))
                keep_cols = ids + kept + [ds.target_column]
                art = session.add_dataset(df=ds.df[keep_cols].copy(), name=f"{ds.name}_{method}", source="derived",
                                          target_column=ds.target_column, task_type=ds.task_type)
                applied = {}
                for other_id in apply_to or []:
                    other = session.get_dataset(other_id)
                    cols = [c for c in keep_cols if c in other.df.columns]
                    applied[other_id] = session.add_dataset(
                        df=other.df[cols].copy(), name=f"{other.name}_{method}", source="derived",
                        target_column=other.target_column if other.target_column in cols else None,
                        task_type=other.task_type).id
                return ok_response({
                    "dataset_id": art.id, "method": method, "n_features_before": X.shape[1],
                    "n_selected": len(kept), "selected": kept, "scores": _scores(selector, list(X.columns)),
                    "applied_to": applied,
                    "note": "" if apply_to else "Selected using every row of this dataset; scores from CV on the "
                                                "result are optimistic. Prefer selecting on a training split.",
                })
        except MCPTimeoutError as e:
            return error_response("timeout", str(e))
        except (KeyError, ValueError) as e:
            return error_response("validation", str(e))
        except ImportError as e:
            return error_response("missing_dependency", str(e))
        except Exception as e:
            return error_response("internal", f"{type(e).__name__}: {e}")

    @mcp.tool()
    def transform_data(
        dataset_id: str,
        transformer: str,
        params: str | dict | None = None,
        columns: list[str] | None = None,
        keep_original: bool | None = None,
        apply_to: list[str] | None = None,
    ) -> str:
        """Apply any transformer class to a dataset by path, returning a new dataset: endgame.preprocessing
        (imputers, encoders, resamplers), endgame.dimensionality_reduction (PCA, UMAP), endgame.feature_selection,
        endgame.signal, endgame.anomaly, endgame.timeseries (MiniRocket), or sklearn ("sklearn.decomposition.PCA").
        columns: inputs (default every feature). keep_original: keep input columns next to the outputs (default: yes
        when the outputs are new columns). apply_to: transform other datasets with the same fitted transformer.
        Resamplers (fit_resample) return the resampled rows. list_modules / describe_api find classes."""
        try:
            ds = session.get_dataset(dataset_id)
            kwargs = params if isinstance(params, dict) else (json.loads(params) if params else {})
            with capture_stdout(), timeout_guard():
                from endgame.mcp.catalog import resolve

                cls = resolve(transformer)
                tr = cls(**kwargs)
                target = ds.target_column
                cols = columns or [c for c in ds.df.columns if c != target]
                X = ds.df[cols]
                y = ds.df[target] if target and target in ds.df.columns else None
                short = cls.__name__.lower()[:12] + "_"

                if hasattr(tr, "fit_resample"):
                    X_res, y_res = tr.fit_resample(X, y)
                    out = _to_pandas(X_res, pd.RangeIndex(len(X_res)), short)
                    if y is not None:
                        out[target] = np.asarray(y_res)
                    art = session.add_dataset(df=out, name=f"{ds.name}_{short.rstrip('_')}", source="derived",
                                              target_column=target, task_type=ds.task_type)
                    return ok_response({"dataset_id": art.id, "shape": list(out.shape),
                                        "rows_before": len(X), "rows_after": len(out)})

                fit_takes_y = "y" in inspect.signature(tr.fit).parameters
                out = tr.fit_transform(X, y) if fit_takes_y and y is not None else tr.fit_transform(X)

                def combine(frame: pd.DataFrame, transformed) -> pd.DataFrame:
                    res = _to_pandas(transformed, frame.index, short)
                    overlap = set(res.columns) & set(cols)
                    keep = keep_original if keep_original is not None else not overlap
                    base = frame if keep else frame.drop(columns=[c for c in cols if c in frame.columns])
                    base = base.drop(columns=[c for c in res.columns if c in base.columns])
                    return pd.concat([base, res], axis=1)

                new_df = combine(ds.df, out)
                art = session.add_dataset(df=new_df, name=f"{ds.name}_{short.rstrip('_')}", source="derived",
                                          target_column=target if target in new_df.columns else None,
                                          task_type=ds.task_type)
                applied = {}
                for other_id in apply_to or []:
                    other = session.get_dataset(other_id)
                    other_df = combine(other.df, tr.transform(other.df[cols]))
                    applied[other_id] = session.add_dataset(
                        df=other_df, name=f"{other.name}_{short.rstrip('_')}", source="derived",
                        target_column=other.target_column if other.target_column in other_df.columns else None,
                        task_type=other.task_type).id
                new_cols = [c for c in new_df.columns if c not in ds.df.columns]
                return ok_response({"dataset_id": art.id, "shape": list(new_df.shape), "transformer": cls.__name__,
                                    "new_columns": new_cols[:60], "n_new_columns": len(new_cols),
                                    "applied_to": applied})
        except MCPTimeoutError as e:
            return error_response("timeout", str(e))
        except (KeyError, ValueError, TypeError) as e:
            return error_response("validation", f"{type(e).__name__}: {e}",
                                  hint="describe_api(transformer) shows its parameters")
        except ImportError as e:
            return error_response("missing_dependency", str(e))
        except Exception as e:
            return error_response("internal", f"{type(e).__name__}: {e}")
