"""Ensemble tool: combine trained models' out-of-fold predictions (endgame.ensemble)."""

from __future__ import annotations

from typing import Any

import numpy as np
from mcp.server.fastmcp import FastMCP

from endgame.mcp.server import capture_stdout, error_response, ok_response
from endgame.mcp.session import SessionManager

METHODS = ("hill_climbing", "stacking", "optimized", "mean", "rank_average")


def _member_scores(art) -> np.ndarray:
    """A model's out-of-fold scores: probability of the positive class, class probabilities, or values."""
    if art.task_type == "regression":
        return np.asarray(art.oof_predictions, dtype=float)
    if art.oof_proba is None:
        raise ValueError(f"{art.id} ({art.model_type}) has no out-of-fold probabilities; ensembling classifiers "
                         "needs models with predict_proba trained with shuffled or grouped CV")
    proba = np.asarray(art.oof_proba, dtype=float)
    return proba[:, 1] if proba.shape[1] == 2 else proba


def _metric(task: str, multiclass: bool):
    """(name, function where higher is better) for scoring blends."""
    from sklearn import metrics as sk

    if task == "regression":
        return "r2", sk.r2_score
    if multiclass:
        return "neg_log_loss", lambda y, p: -sk.log_loss(y, np.clip(p, 1e-15, 1) / np.clip(p, 1e-15, 1).sum(1,
                                                                                                         keepdims=True))
    return "roc_auc", sk.roc_auc_score


class EnsembleModel:
    """Weighted (or stacked) combination of session models; predicts from raw dataset columns."""

    def __init__(self, members: list, weights: np.ndarray, task_type: str, meta: Any = None, classes=None):
        self.members, self.weights, self.task_type, self.meta, self.classes_ = members, weights, task_type, meta, classes

    def _scores(self, X) -> list[np.ndarray]:
        from endgame.mcp.tools._encoding import apply_encoders

        out = []
        for art in self.members:
            Xm, _ = apply_encoders(X.copy(), None, art)
            if self.task_type == "regression":
                out.append(np.asarray(art.estimator.predict(Xm), dtype=float))
            else:
                proba = np.asarray(art.estimator.predict_proba(Xm), dtype=float)
                out.append(proba[:, 1] if proba.shape[1] == 2 else proba)
        return out

    def _combine(self, scores: list[np.ndarray]) -> np.ndarray:
        if self.meta is not None:
            stacked = np.column_stack([s.reshape(len(s), -1) for s in scores])
            if self.task_type == "regression":
                return self.meta.predict(stacked)
            proba = self.meta.predict_proba(stacked)
            return proba[:, 1] if proba.shape[1] == 2 else proba
        return sum(w * s for w, s in zip(self.weights, scores)) / self.weights.sum()

    def predict_proba(self, X):
        p = self._combine(self._scores(X))
        return np.column_stack([1 - p, p]) if p.ndim == 1 else p

    def predict(self, X):
        p = self._combine(self._scores(X))
        if self.task_type == "regression":
            return p
        return (p >= 0.5).astype(int) if p.ndim == 1 else p.argmax(axis=1)


def register(mcp: FastMCP, session: SessionManager) -> None:

    @mcp.tool()
    def ensemble(model_ids: list[str], method: str = "hill_climbing") -> str:
        """Combine models trained by train_model/compare_models on the same dataset and folds, using their
        out-of-fold predictions (endgame.ensemble). method: hill_climbing (greedy forward selection, robust default),
        stacking (ridge/logistic meta-model, scored with nested CV), optimized (Optuna weights), mean, rank_average.
        Reports the blend's out-of-fold score next to each member's; the result is a model usable by predict and
        evaluate_model. Keep it only if it beats the best single model by more than the noise."""
        try:
            if method not in METHODS:
                return error_response("validation", f"Unknown method '{method}'", hint=f"One of: {', '.join(METHODS)}")
            arts = [session.get_model(m) for m in model_ids]
            if len(arts) < 2:
                return error_response("validation", "Need at least two models")
            if len({a.dataset_id for a in arts}) > 1:
                return error_response("validation", "All models must be trained on the same dataset")
            cvs = {str(a.cv) for a in arts if a.cv}
            if len(cvs) > 1:
                return error_response("validation", f"Models used different CV setups {sorted(cvs)}; their "
                                      "out-of-fold predictions are not comparable. Retrain with the same settings.")

            with capture_stdout():
                from sklearn.model_selection import KFold, cross_val_predict

                from endgame.mcp.tools._encoding import encode_target

                ds = session.get_dataset(arts[0].dataset_id)
                task = arts[0].task_type
                y = ds.df[ds.target_column]
                if task != "regression":
                    y, _ = encode_target(y, target_encoder=arts[0].target_encoder)
                y = np.asarray(y)
                scores = [_member_scores(a) for a in arts]
                multiclass = scores[0].ndim == 2
                rows = np.ones(len(y), dtype=bool)
                for s in scores:  # time-ordered CV leaves the earliest rows unscored
                    rows &= ~np.isnan(s.reshape(len(s), -1)).any(axis=1)
                y_r, s_r = y[rows], [s[rows] for s in scores]
                if task != "regression":
                    y_r = y_r.astype(int)
                metric_name, score = _metric(task, multiclass)
                singles = {a.id: round(float(score(y_r, s)), 4) for a, s in zip(arts, s_r)}

                meta, weights = None, np.ones(len(arts))
                if method == "mean":
                    blend = sum(s_r) / len(s_r)
                elif method == "rank_average":
                    from scipy.stats import rankdata
                    if multiclass:
                        return error_response("validation", "rank_average needs a binary or regression task")
                    blend = sum(rankdata(s) / len(s) for s in s_r) / len(s_r)
                elif method in ("hill_climbing", "optimized"):
                    from endgame.ensemble import HillClimbingEnsemble, OptimizedBlender
                    cls = HillClimbingEnsemble if method == "hill_climbing" else OptimizedBlender
                    blender = cls(metric=score, maximize=True, random_state=42)
                    blender.fit(list(s_r), y_r)
                    w = blender.weights_
                    weights = np.array([w.get(i, 0.0) for i in range(len(arts))] if isinstance(w, dict)
                                       else np.asarray(w, dtype=float))
                    blend = sum(wi * s for wi, s in zip(weights, s_r)) / weights.sum()
                else:  # stacking: meta-model on out-of-fold scores, itself scored out of fold
                    from sklearn.linear_model import LogisticRegression, RidgeCV
                    stacked = np.column_stack([s.reshape(len(s), -1) for s in s_r])
                    meta = RidgeCV() if task == "regression" else LogisticRegression(C=1.0, max_iter=2000)
                    cv = KFold(5, shuffle=True, random_state=7)
                    if task == "regression":
                        blend = cross_val_predict(meta, stacked, y_r, cv=cv)
                    else:
                        blend = cross_val_predict(meta, stacked, y_r, cv=cv, method="predict_proba")
                        blend = blend[:, 1] if blend.shape[1] == 2 else blend
                    meta.fit(stacked, y_r)
                    weights = np.abs(np.ravel(meta.coef_))[:len(arts)] if not multiclass else weights

                ens_score = float(score(y_r, blend))
                best_id = max(singles, key=singles.get)
                ensemble_est = EnsembleModel(arts, weights, task, meta=meta)
                oof = np.full(len(y), np.nan) if not multiclass else None
                if oof is not None:
                    oof[rows] = blend
                art = session.add_model(
                    estimator=ensemble_est, name=f"ensemble_{method}", model_type=f"ensemble_{method}",
                    dataset_id=arts[0].dataset_id, task_type=task, metrics={metric_name: ens_score},
                    feature_names=[], oof_predictions=oof if task == "regression" else None,
                    oof_proba=(np.column_stack([1 - oof, oof]) if oof is not None and task != "regression"
                               else (blend if multiclass else None)),
                    target_encoder=arts[0].target_encoder, cv=arts[0].cv,
                )
                total = weights.sum() or 1.0
                return ok_response({
                    "model_id": art.id, "method": method, "metric": metric_name,
                    "ensemble_score": round(ens_score, 4), "best_single": {"model_id": best_id,
                                                                           "score": singles[best_id]},
                    "gain_over_best": round(ens_score - singles[best_id], 4),
                    "members": [{"model_id": a.id, "model": a.model_type, "score": singles[a.id],
                                 "weight": round(float(w / total), 3)} for a, w in zip(arts, weights)],
                    "rows_scored": int(rows.sum()),
                })
        except KeyError as e:
            return error_response("not_found", str(e))
        except ValueError as e:
            return error_response("validation", str(e))
        except Exception as e:
            return error_response("internal", f"{type(e).__name__}: {e}")
