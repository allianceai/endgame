from __future__ import annotations

"""RuleFit++: Enhanced rule-based models.

Improvements over classic RuleFit (Friedman & Popescu, 2008):
- Multi-source rule generation (LightGBM + RandomForest + ExtraTrees)
- Soft rules via sigmoid activation for smooth decision boundaries
- Elastic net selection (handles rule collinearity)
- Gradient-boosted matching pursuit selection (alternative)
- Optional pairwise rule interaction features
- Optional threshold refinement via gradient descent
- Multinomial classification via saga solver
- Optional isotonic calibration for probability outputs
- Rule similarity merging for a more compact model

Example
-------
>>> from endgame.models.rules import RuleFitPlusClassifier
>>> clf = RuleFitPlusClassifier(soft_rules=True, selection='elasticnet')
>>> clf.fit(X_train, y_train)
>>> proba = clf.predict_proba(X_test)
>>> clf.summary()
"""

import warnings
from collections import defaultdict

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from endgame.models.rules.extraction import extract_rules_from_ensemble
from endgame.models.rules.rule import Condition, Operator, Rule, RuleEnsemble

try:
    import lightgbm as lgb

    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False


# ---------------------------------------------------------------------------
# LightGBM rule extraction
# ---------------------------------------------------------------------------

def _extract_rules_from_lgbm(model, feature_names):
    """Extract rules from a fitted LightGBM model via dump_model()."""
    dump = model.booster_.dump_model()
    all_rules = []
    for tree_idx, tree_info in enumerate(dump["tree_info"]):
        root = tree_info["tree_structure"]
        root_count = root.get("internal_count", 1)
        _lgbm_recurse(root, feature_names, tree_idx, root_count, [], all_rules)
    return RuleEnsemble(
        rules=all_rules,
        n_features=len(feature_names),
        feature_names=feature_names,
    )


def _lgbm_recurse(node, feature_names, tree_idx, root_count, conditions, out):
    """Recursively traverse a LightGBM tree node and extract rules."""
    if "split_feature" not in node:
        return

    feat_idx = node["split_feature"]
    threshold = node["threshold"]
    feat_name = feature_names[feat_idx]
    decision = node.get("decision_type", "<=")

    if decision != "<=":
        return

    left = node["left_child"]
    right = node["right_child"]

    left_cond = Condition(feat_idx, feat_name, Operator.LE, threshold)
    left_conds = conditions + [left_cond]
    if left_conds:
        count = left.get("leaf_count", left.get("internal_count", 0))
        rule = Rule(
            conditions=list(left_conds), tree_idx=tree_idx,
            node_idx=left.get("split_index", left.get("leaf_index", -1)),
        )
        rule.support = count / max(root_count, 1)
        out.append(rule)

    right_cond = Condition(feat_idx, feat_name, Operator.GT, threshold)
    right_conds = conditions + [right_cond]
    if right_conds:
        count = right.get("leaf_count", right.get("internal_count", 0))
        rule = Rule(
            conditions=list(right_conds), tree_idx=tree_idx,
            node_idx=right.get("split_index", right.get("leaf_index", -1)),
        )
        rule.support = count / max(root_count, 1)
        out.append(rule)

    _lgbm_recurse(left, feature_names, tree_idx, root_count, left_conds, out)
    _lgbm_recurse(right, feature_names, tree_idx, root_count, right_conds, out)


# ---------------------------------------------------------------------------
# Multi-source rule generation
# ---------------------------------------------------------------------------

def _generate_rules(
    X, y, task, feature_names, rule_sources, n_estimators, tree_max_depth,
    random_state,
):
    """Generate rules from multiple tree ensemble sources."""
    all_rules = []
    rng = np.random.RandomState(random_state)
    source_offset = 0

    for src in rule_sources:
        seed = int(rng.randint(0, 2**31))
        if src == "gb":
            rules = _rules_from_gb(
                X, y, task, feature_names, n_estimators, tree_max_depth, seed,
            )
        elif src == "rf":
            rules = _rules_from_rf(
                X, y, task, feature_names, n_estimators, tree_max_depth, seed,
            )
        elif src == "et":
            rules = _rules_from_et(
                X, y, task, feature_names, n_estimators, tree_max_depth, seed,
            )
        else:
            raise ValueError(f"Unknown rule source: {src!r}")

        for r in rules:
            r.tree_idx += source_offset
        source_offset += n_estimators
        all_rules.extend(rules)

    return RuleEnsemble(
        rules=all_rules,
        n_features=len(feature_names),
        feature_names=feature_names,
    )


def _rules_from_gb(X, y, task, feature_names, n_estimators, max_depth, seed):
    """Extract rules from gradient boosting."""
    if _HAS_LGBM:
        params = dict(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.15,
            subsample=0.7, colsample_bytree=0.8, reg_alpha=0.1,
            random_state=seed, verbose=-1, n_jobs=-1,
        )
        if task == "classification":
            model = lgb.LGBMClassifier(**params)
        else:
            model = lgb.LGBMRegressor(**params)
        model.fit(X, y)
        return _extract_rules_from_lgbm(model, feature_names).rules
    else:
        if task == "classification":
            from sklearn.ensemble import GradientBoostingClassifier
            model = GradientBoostingClassifier(
                n_estimators=n_estimators, max_depth=max_depth,
                learning_rate=0.15, subsample=0.5, max_features="sqrt",
                random_state=seed,
            )
        else:
            from sklearn.ensemble import GradientBoostingRegressor
            model = GradientBoostingRegressor(
                n_estimators=n_estimators, max_depth=max_depth,
                learning_rate=0.15, subsample=0.5, max_features="sqrt",
                random_state=seed,
            )
        model.fit(X, y)
        return extract_rules_from_ensemble(model, feature_names).rules


def _rules_from_rf(X, y, task, feature_names, n_estimators, max_depth, seed):
    if task == "classification":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            max_features="sqrt", random_state=seed, n_jobs=-1,
        )
    else:
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            max_features="sqrt", random_state=seed, n_jobs=-1,
        )
    model.fit(X, y)
    return extract_rules_from_ensemble(model, feature_names).rules


def _rules_from_et(X, y, task, feature_names, n_estimators, max_depth, seed):
    if task == "classification":
        from sklearn.ensemble import ExtraTreesClassifier
        model = ExtraTreesClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            max_features="sqrt", random_state=seed, n_jobs=-1,
        )
    else:
        from sklearn.ensemble import ExtraTreesRegressor
        model = ExtraTreesRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            max_features="sqrt", random_state=seed, n_jobs=-1,
        )
    model.fit(X, y)
    return extract_rules_from_ensemble(model, feature_names).rules


# ---------------------------------------------------------------------------
# Rule transforms (soft / hard)
# ---------------------------------------------------------------------------

def _compile_rules(rules):
    """Compile rules into flat numpy arrays for vectorized evaluation.

    Returns dict with keys: feat_idx, thresholds, is_le, n_conds, max_conds.
    """
    n = len(rules)
    if n == 0:
        return dict(
            feat_idx=np.zeros((0, 0), dtype=np.intp),
            thresholds=np.zeros((0, 0)),
            is_le=np.zeros((0, 0), dtype=bool),
            n_conds=np.zeros(0, dtype=np.int32),
            max_conds=0,
        )

    max_c = max(len(r.conditions) for r in rules)
    fi = np.zeros((n, max_c), dtype=np.intp)
    th = np.zeros((n, max_c), dtype=np.float64)
    le = np.zeros((n, max_c), dtype=bool)
    nc = np.zeros(n, dtype=np.int32)

    for j, rule in enumerate(rules):
        nc[j] = len(rule.conditions)
        for k, c in enumerate(rule.conditions):
            fi[j, k] = c.feature_idx
            th[j, k] = c.threshold
            le[j, k] = c.operator == Operator.LE

    return dict(feat_idx=fi, thresholds=th, is_le=le, n_conds=nc, max_conds=max_c)


def _hard_transform(X, compiled):
    """Binary 0/1 rule evaluation (same as classic RuleFit)."""
    n_samples = X.shape[0]
    n_rules = len(compiled["n_conds"])
    if n_rules == 0:
        return np.zeros((n_samples, 0), dtype=np.float32)

    result = np.ones((n_samples, n_rules), dtype=np.float32)
    for k in range(compiled["max_conds"]):
        active = np.where(compiled["n_conds"] > k)[0]
        if len(active) == 0:
            break
        fi = compiled["feat_idx"][active, k]
        th = compiled["thresholds"][active, k]
        le = compiled["is_le"][active, k]
        vals = X[:, fi]
        cond = np.where(le, vals <= th, vals > th)
        result[:, active] *= cond.astype(np.float32)
    return result


def _soft_transform(X, compiled, sharpness=10.0):
    """Sigmoid-based soft rule evaluation.

    For LE conditions: sigmoid(sharpness * (threshold - value))
    For GT conditions: sigmoid(sharpness * (value - threshold))
    """
    n_samples = X.shape[0]
    n_rules = len(compiled["n_conds"])
    if n_rules == 0:
        return np.zeros((n_samples, 0), dtype=np.float32)

    result = np.ones((n_samples, n_rules), dtype=np.float32)
    half_s = sharpness * 0.5
    for k in range(compiled["max_conds"]):
        active = np.where(compiled["n_conds"] > k)[0]
        if len(active) == 0:
            break
        fi = compiled["feat_idx"][active, k]
        th = compiled["thresholds"][active, k]
        le = compiled["is_le"][active, k]
        vals = X[:, fi]
        diff = np.where(le, th - vals, vals - th)
        # tanh form of sigmoid: 0.5*(1 + tanh(x/2)), avoids slow exp()
        soft = (0.5 * (1.0 + np.tanh(np.clip(half_s * diff, -15, 15)))).astype(np.float32)
        result[:, active] *= soft
    return result


# ---------------------------------------------------------------------------
# Rule merging
# ---------------------------------------------------------------------------

def _merge_similar_rules(rules, tolerance=0.05):
    """Merge rules with same condition structure and similar thresholds.

    Groups by (feature_indices, operators), then within each group keeps
    the rule with highest support when threshold vectors are within tolerance.
    """
    groups = defaultdict(list)
    for rule in rules:
        key = tuple(
            sorted((c.feature_idx, c.operator.value) for c in rule.conditions)
        )
        groups[key].append(rule)

    merged = []
    for _key, group in groups.items():
        if len(group) == 1:
            merged.append(group[0])
            continue

        group.sort(key=lambda r: r.support, reverse=True)
        kept = []
        for rule in group:
            tv = np.array([c.threshold for c in rule.conditions])
            is_dup = False
            for kept_rule in kept:
                ktv = np.array([c.threshold for c in kept_rule.conditions])
                if len(tv) == len(ktv):
                    denom = np.maximum(np.abs(ktv), 1e-10)
                    if np.all(np.abs(tv - ktv) / denom < tolerance):
                        is_dup = True
                        break
            if not is_dup:
                kept.append(rule)
        merged.extend(kept)
    return merged


# ---------------------------------------------------------------------------
# Rule interactions
# ---------------------------------------------------------------------------

def _create_interaction_features(X_rules, max_interaction_rules=30):
    """Create pairwise product features from the top rules by variance.

    Returns (X_interactions, pair_indices) where pair_indices is a list of
    (i, j) tuples indicating which rule columns were multiplied.
    """
    n_rules = X_rules.shape[1]
    if n_rules < 2:
        return np.zeros((X_rules.shape[0], 0), dtype=np.float32), []

    var = np.var(X_rules, axis=0)
    top_k = min(max_interaction_rules, n_rules)
    top_idx = np.argsort(var)[-top_k:]

    pairs = []
    cols = []
    for a_pos in range(len(top_idx)):
        for b_pos in range(a_pos + 1, len(top_idx)):
            i, j = int(top_idx[a_pos]), int(top_idx[b_pos])
            cols.append(X_rules[:, i] * X_rules[:, j])
            pairs.append((i, j))

    if not cols:
        return np.zeros((X_rules.shape[0], 0), dtype=np.float32), []
    return np.column_stack(cols).astype(np.float32), pairs


def _apply_interaction_pairs(X_rules, pairs):
    """Apply pre-computed interaction pairs (from training) to new data.

    Unlike _create_interaction_features, this does NOT re-select top rules
    by variance — it uses the exact pairs stored during fit.
    """
    if not pairs:
        return np.zeros((X_rules.shape[0], 0), dtype=np.float32)

    cols = []
    for i, j in pairs:
        cols.append(X_rules[:, i] * X_rules[:, j])
    return np.column_stack(cols).astype(np.float32)


# ---------------------------------------------------------------------------
# Selection methods
# ---------------------------------------------------------------------------

def _fit_elasticnet_regression(X, y, l1_ratio, cv, n_jobs, random_state):
    from sklearn.linear_model import ElasticNetCV, LassoCV

    # Phase 1: fast Lasso screen to reduce dimensionality
    screen = LassoCV(cv=3, alphas=20, n_jobs=n_jobs,
                     random_state=random_state, max_iter=1000)
    screen.fit(X, y)
    mask = np.abs(screen.coef_) > 1e-10
    n_kept = int(mask.sum())

    if n_kept == 0:
        return screen.coef_, screen.intercept_, screen.alpha_

    # Phase 2: elasticnet on surviving features
    X_reduced = X[:, mask]
    model = ElasticNetCV(
        l1_ratio=l1_ratio, cv=cv, alphas=30,
        n_jobs=n_jobs, random_state=random_state, max_iter=2000,
    )
    model.fit(X_reduced, y)

    full_coef = np.zeros(X.shape[1])
    full_coef[mask] = model.coef_
    return full_coef, model.intercept_, model.alpha_


def _fit_l1_regression(X, y, cv, n_jobs, random_state):
    from sklearn.linear_model import LassoCV

    model = LassoCV(
        cv=cv, alphas=30, n_jobs=n_jobs,
        random_state=random_state, max_iter=2000,
    )
    model.fit(X, y)
    return model.coef_, model.intercept_, model.alpha_


def _fit_boosted_regression(
    X, y, n_rounds=200, learning_rate=0.1, l2_reg=1.0,
):
    """Gradient-boosted matching pursuit for regression."""
    n_samples, n_features = X.shape
    intercept = float(np.mean(y))
    predictions = np.full(n_samples, intercept, dtype=np.float64)
    coef = np.zeros(n_features, dtype=np.float64)

    for _ in range(n_rounds):
        residuals = y - predictions
        corr = X.T @ residuals
        best = int(np.argmax(np.abs(corr)))
        col = X[:, best]
        denom = float(col @ col) + l2_reg
        if denom < 1e-12:
            break
        step = learning_rate * corr[best] / denom
        if abs(step) < 1e-10:
            break
        coef[best] += step
        predictions += step * col

    return coef, intercept, 0.0


def _pick_solver(n_samples, threshold=5000):
    """Choose solver based on dataset size.

    liblinear's coordinate descent yields better L1 sparsity on small data;
    saga's stochastic updates are much faster when n is large.
    """
    return "liblinear" if n_samples < threshold else "saga"


def _fit_elasticnet_classification(X, y, l1_ratio, cv, n_jobs, random_state,
                                   class_weight):
    """Elastic net classification with adaptive solver.

    Expects decorrelated input (reasonable condition number).
    Uses liblinear for small n (better sparsity), saga for large n (faster).
    """
    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

    n_classes = len(np.unique(y))
    solver = _pick_solver(X.shape[0])

    if n_classes > 2:
        model = LogisticRegression(
            penalty="l1", solver=solver, C=1.0,
            max_iter=500, tol=1e-3,
            class_weight=class_weight, random_state=random_state,
        )
        model.fit(X, y)
        model.C_ = np.array([1.0])
        model.Cs_ = np.array([1.0])
        return model

    if solver == "liblinear":
        # L1 screen then elasticnet refit for best sparsity
        screen = LogisticRegressionCV(
            penalty="l1", solver="liblinear", Cs=5, cv=cv,
            max_iter=300, n_jobs=n_jobs,
            class_weight=class_weight, random_state=random_state,
        )
        screen.fit(X, y)

        mask = np.abs(screen.coef_.ravel()) > 1e-10
        n_kept = int(mask.sum())
        if n_kept == 0 or n_kept >= X.shape[1] * 0.5:
            return screen

        X_reduced = X[:, mask]
        model = LogisticRegressionCV(
            penalty="elasticnet", solver="saga",
            l1_ratios=[l1_ratio], Cs=3, cv=cv,
            max_iter=500, tol=1e-3, n_jobs=n_jobs,
            class_weight=class_weight, random_state=random_state,
        )
        model.fit(X_reduced, y)

        if model.coef_.ndim == 1:
            full_coef = np.zeros(X.shape[1])
            full_coef[mask] = model.coef_
            model.coef_ = full_coef.reshape(1, -1)
        else:
            full_coef = np.zeros((model.coef_.shape[0], X.shape[1]))
            full_coef[:, mask] = model.coef_
            model.coef_ = full_coef
        return model
    else:
        model = LogisticRegressionCV(
            penalty="elasticnet", solver="saga",
            l1_ratios=[l1_ratio], Cs=3, cv=cv,
            max_iter=500, tol=1e-3, n_jobs=n_jobs,
            class_weight=class_weight, random_state=random_state,
        )
        model.fit(X, y)
        return model


def _fit_l1_classification(X, y, cv, n_jobs, random_state, class_weight):
    """L1 classification with adaptive solver.

    Expects decorrelated input for fast convergence.
    """
    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

    n_classes = len(np.unique(y))
    solver = _pick_solver(X.shape[0])

    if n_classes > 2:
        model = LogisticRegression(
            penalty="l1", solver=solver, C=1.0,
            max_iter=500, tol=1e-3,
            class_weight=class_weight, random_state=random_state,
        )
        model.fit(X, y)
        model.C_ = np.array([1.0])
        return model

    mi = 300 if solver == "liblinear" else 500
    kw = dict(
        penalty="l1", solver=solver, cv=cv, Cs=5,
        max_iter=mi, n_jobs=n_jobs,
        class_weight=class_weight, random_state=random_state,
    )
    if solver == "saga":
        kw["tol"] = 1e-3
    model = LogisticRegressionCV(**kw)
    model.fit(X, y)
    return model


def _fit_boosted_classification(X, y, n_classes, n_rounds=200,
                                learning_rate=0.1, l2_reg=1.0):
    """Gradient-boosted matching pursuit on logistic loss."""
    n_samples, n_features = X.shape

    if n_classes == 2:
        p_mean = np.clip(np.mean(y), 1e-7, 1 - 1e-7)
        intercept = np.log(p_mean / (1 - p_mean))
        raw = np.full(n_samples, intercept, dtype=np.float64)
        coef = np.zeros(n_features, dtype=np.float64)

        for _ in range(n_rounds):
            p = 1.0 / (1.0 + np.exp(-np.clip(raw, -30, 30)))
            grad = y - p
            corr = X.T @ grad
            best = int(np.argmax(np.abs(corr)))
            col = X[:, best]
            denom = float(col @ col) + l2_reg
            if denom < 1e-12:
                break
            step = learning_rate * corr[best] / denom
            if abs(step) < 1e-10:
                break
            coef[best] += step
            raw += step * col

        return coef, np.array([intercept])
    else:
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(
            penalty="l1", solver="liblinear", C=1.0, max_iter=1000,
        )
        model.fit(X, y)
        return model.coef_, model.intercept_


# ---------------------------------------------------------------------------
# Threshold refinement
# ---------------------------------------------------------------------------

def _refine_thresholds(X, y, compiled, coef, intercept, sharpness,
                       n_linear, is_classification, lr=0.01, steps=50):
    """Fine-tune rule thresholds via gradient descent (Adam).

    Only adjusts thresholds while keeping coef/intercept frozen.
    """
    thresholds = compiled["thresholds"].copy()
    n_rules = len(compiled["n_conds"])
    if n_rules == 0:
        return compiled

    m = np.zeros_like(thresholds)
    v = np.zeros_like(thresholds)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    for t in range(1, steps + 1):
        X_rules = _soft_transform_with_thresholds(
            X, compiled, thresholds, sharpness
        )
        if n_linear > 0:
            pred = X[:, :n_linear] @ coef[:n_linear] + X_rules @ coef[n_linear:] + intercept
        else:
            pred = X_rules @ coef + intercept

        if is_classification:
            p = 1.0 / (1.0 + np.exp(-np.clip(pred, -30, 30)))
            dloss_dpred = p - y
        else:
            dloss_dpred = pred - y

        grad = np.zeros_like(thresholds)
        for k in range(compiled["max_conds"]):
            active = np.where(compiled["n_conds"] > k)[0]
            if len(active) == 0:
                break
            fi = compiled["feat_idx"][active, k]
            th_k = thresholds[active, k]
            le = compiled["is_le"][active, k]

            vals = X[:, fi]
            diff = np.where(le, th_k - vals, vals - th_k)
            diff_clipped = np.clip(sharpness * diff, -30, 30)
            sig = 1.0 / (1.0 + np.exp(-diff_clipped))

            rule_vals = _soft_transform_with_thresholds(
                X, compiled, thresholds, sharpness
            )
            safe_sig = np.maximum(sig, 1e-10)
            dsig_dt = sharpness * sig * (1 - sig)
            dsig_dt_signed = np.where(le, dsig_dt, -dsig_dt)

            drule_dt = rule_vals[:, active] / safe_sig * dsig_dt_signed

            rule_coefs = coef[n_linear:][active] if n_linear > 0 else coef[active]
            g = np.sum(dloss_dpred[:, None] * drule_dt * rule_coefs[None, :], axis=0)
            grad[active, k] = g / X.shape[0]

        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        thresholds -= lr * m_hat / (np.sqrt(v_hat) + eps)

    refined = dict(compiled)
    refined["thresholds"] = thresholds
    return refined


def _soft_transform_with_thresholds(X, compiled, thresholds, sharpness):
    """Soft transform using externally provided thresholds."""
    n_samples = X.shape[0]
    n_rules = len(compiled["n_conds"])
    result = np.ones((n_samples, n_rules), dtype=np.float64)
    for k in range(compiled["max_conds"]):
        active = np.where(compiled["n_conds"] > k)[0]
        if len(active) == 0:
            break
        fi = compiled["feat_idx"][active, k]
        th = thresholds[active, k]
        le = compiled["is_le"][active, k]
        vals = X[:, fi]
        diff = np.where(le, th - vals, vals - th)
        diff_clipped = np.clip(sharpness * diff, -30, 30)
        result[:, active] *= 1.0 / (1.0 + np.exp(-diff_clipped))
    return result.astype(np.float32)


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def _decorrelate_features(X, threshold=0.85):
    """Remove features whose pairwise |correlation| exceeds *threshold*.

    Among each correlated pair, the lower-variance feature is dropped.
    Returns (X_reduced, keep_mask) where keep_mask is a boolean array of
    shape (n_features,).
    """
    n = X.shape[1]
    if n <= 1:
        return X, np.ones(n, dtype=bool)

    # Sample-based correlation for large feature counts to avoid O(p³)
    if X.shape[0] > 500 and n > 200:
        idx = np.random.RandomState(0).choice(X.shape[0], 500, replace=False)
        corr = np.corrcoef(X[idx].T)
    else:
        corr = np.corrcoef(X.T)
    np.fill_diagonal(corr, 0)
    var = np.var(X, axis=0)
    keep = np.ones(n, dtype=bool)

    # Find all correlated pairs at once (vectorized)
    abs_corr = np.abs(corr)
    rows, cols = np.where(np.triu(abs_corr > threshold, k=1))
    # Process pairs in order, dropping lower-variance feature
    for i, j in zip(rows, cols):
        if not keep[i] or not keep[j]:
            continue
        if var[i] >= var[j]:
            keep[j] = False
        else:
            keep[i] = False

    return X[:, keep], keep


def _preprocess_linear(X, fit=True, params=None, winsorize=0.025):
    """Winsorize + standardize linear features."""
    params = params.copy() if params else {}
    Xp = X.copy()

    if winsorize and winsorize > 0:
        if fit:
            lo = np.percentile(X, winsorize * 100, axis=0)
            hi = np.percentile(X, (1 - winsorize) * 100, axis=0)
            params["lo"], params["hi"] = lo, hi
        Xp = np.clip(Xp, params["lo"], params["hi"])

    if fit:
        mu = np.mean(Xp, axis=0)
        sd = np.std(Xp, axis=0)
        sd[sd < 1e-10] = 1.0
        params["mu"], params["sd"] = mu, sd
    Xp = (Xp - params["mu"]) / params["sd"]
    return Xp, params


# ---------------------------------------------------------------------------
# RuleFitPlusRegressor
# ---------------------------------------------------------------------------

class RuleFitPlusRegressor(BaseEstimator, RegressorMixin):
    """RuleFit++ regressor — delegates to RuleFitRegressor.

    RuleFit++ regression consistently underperforms vanilla RuleFit
    (9W/20L on benchmarks), so this class now delegates entirely to
    ``RuleFitRegressor`` while preserving the original ``__init__``
    signature for backwards compatibility.

    Parameters
    ----------
    n_estimators : int
        Trees per source ensemble.
    tree_max_depth : int
        Maximum tree depth for rule extraction.
    rule_sources : tuple of str
        Ignored (kept for API compat).
    max_rules : int or None
        Cap on total rules after dedup/filter.
    min_support, max_support : float
        Support bounds for rule filtering.
    soft_rules : bool
        Ignored (kept for API compat).
    sharpness : float
        Ignored (kept for API compat).
    include_linear : bool
        Include original features as linear terms.
    rule_interactions : bool
        Ignored (kept for API compat).
    max_interaction_rules : int
        Ignored (kept for API compat).
    selection : str
        Ignored (kept for API compat).
    alpha : float or None
        Fixed regularization strength. If None, selected via CV.
    l1_ratio : float
        Ignored (kept for API compat).
    cv : int
        CV folds for regularization selection.
    n_boosting_rounds : int
        Ignored (kept for API compat).
    boosting_lr : float
        Ignored (kept for API compat).
    refine_thresholds : bool
        Ignored (kept for API compat).
    refine_steps : int
        Ignored (kept for API compat).
    refine_lr : float
        Ignored (kept for API compat).
    merge_similar : bool
        Ignored (kept for API compat).
    merge_tolerance : float
        Ignored (kept for API compat).
    random_state : int or None
        Random seed.
    n_jobs : int or None
        Parallel jobs for CV.
    """

    def __init__(
        self,
        n_estimators: int = 30,
        tree_max_depth: int = 3,
        rule_sources: tuple = ("gb", "rf"),
        max_rules: int | None = 200,
        min_support: float = 0.01,
        max_support: float = 0.99,
        soft_rules: bool = True,
        sharpness: float = 10.0,
        include_linear: bool = True,
        rule_interactions: bool = False,
        max_interaction_rules: int = 30,
        selection: str = "l1",
        alpha: float | None = None,
        l1_ratio: float = 0.8,
        cv: int = 5,
        n_boosting_rounds: int = 200,
        boosting_lr: float = 0.1,
        refine_thresholds: bool = False,
        refine_steps: int = 50,
        refine_lr: float = 0.01,
        merge_similar: bool = True,
        merge_tolerance: float = 0.05,
        random_state: int | None = None,
        n_jobs: int | None = None,
    ):
        self.n_estimators = n_estimators
        self.tree_max_depth = tree_max_depth
        self.rule_sources = rule_sources
        self.max_rules = max_rules
        self.min_support = min_support
        self.max_support = max_support
        self.soft_rules = soft_rules
        self.sharpness = sharpness
        self.include_linear = include_linear
        self.rule_interactions = rule_interactions
        self.max_interaction_rules = max_interaction_rules
        self.selection = selection
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.cv = cv
        self.n_boosting_rounds = n_boosting_rounds
        self.boosting_lr = boosting_lr
        self.refine_thresholds = refine_thresholds
        self.refine_steps = refine_steps
        self.refine_lr = refine_lr
        self.merge_similar = merge_similar
        self.merge_tolerance = merge_tolerance
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y, feature_names=None, sample_weight=None):
        from endgame.models.rules.rulefit import RuleFitRegressor

        self._delegate = RuleFitRegressor(
            n_estimators=self.n_estimators,
            tree_max_depth=self.tree_max_depth,
            max_rules=self.max_rules,
            min_support=self.min_support,
            max_support=self.max_support,
            alpha=self.alpha,
            cv=self.cv,
            include_linear=self.include_linear,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self._delegate.fit(X, y, feature_names=feature_names,
                           sample_weight=sample_weight)

        # Copy fitted attributes for direct access
        for attr in (
            "coef_", "intercept_", "rule_ensemble_", "n_rules_",
            "n_rules_selected_", "rules_", "feature_importances_",
            "linear_coef_", "rule_coef_", "alpha_", "n_features_in_",
            "feature_names_in_", "tree_generator_", "cv_results_",
        ):
            if hasattr(self._delegate, attr):
                setattr(self, attr, getattr(self._delegate, attr))

        return self

    def predict(self, X):
        check_is_fitted(self, "_delegate")
        return self._delegate.predict(X)

    def transform(self, X):
        check_is_fitted(self, "_delegate")
        return self._delegate.transform(X)

    def get_rules(self, exclude_zero_coef=True, sort_by="importance"):
        check_is_fitted(self, "_delegate")
        return self._delegate.get_rules(exclude_zero_coef=exclude_zero_coef,
                                        sort_by=sort_by)

    def summary(self):
        check_is_fitted(self, "_delegate")
        return self._delegate.summary()

    def get_equation(self, precision: int = 4):
        check_is_fitted(self, "_delegate")
        return self._delegate.get_equation(precision=precision)

    def visualize_rule(self, rule_idx: int):
        check_is_fitted(self, "_delegate")
        return self._delegate.visualize_rule(rule_idx)


# ---------------------------------------------------------------------------
# RuleFitPlusClassifier
# ---------------------------------------------------------------------------

class RuleFitPlusClassifier(ClassifierMixin, BaseEstimator):
    """RuleFit++ classifier: enhanced rule ensemble with soft rules.

    Parameters
    ----------
    n_estimators : int
        Trees per source ensemble.
    tree_max_depth : int
        Maximum tree depth for rule extraction.
    rule_sources : tuple of str
        Ensemble types: 'gb', 'rf', 'et'.
    max_rules : int or None
        Cap on total rules.
    min_support, max_support : float
        Support bounds.
    soft_rules : bool
        Sigmoid activation.
    sharpness : float
        Sigmoid steepness.
    include_linear : bool
        Include original features.
    rule_interactions : bool
        Pairwise interaction features.
    max_interaction_rules : int
        Top rules for interaction pairs.
    selection : str
        'elasticnet', 'boosted', or 'l1'.
    alpha : float or None
        Fixed regularization.
    l1_ratio : float
        Elastic net mixing.
    cv : int
        CV folds.
    n_boosting_rounds : int
        Boosted selection rounds.
    boosting_lr : float
        Boosted selection learning rate.
    decorrelation_threshold : float
        Pairwise |correlation| above which duplicate features are
        pruned before fitting.
    refine_thresholds : bool
        Gradient-based threshold refinement.
    refine_steps : int
        Refinement iterations.
    refine_lr : float
        Refinement learning rate.
    calibrate : bool
        Apply isotonic calibration.
    merge_similar : bool
        Merge near-duplicate rules.
    merge_tolerance : float
        Threshold tolerance for merging.
    random_state : int or None
        Random seed.
    n_jobs : int or None
        Parallel jobs.
    class_weight : dict, 'balanced', or None
        Class weights.
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        n_estimators: int = 50,
        tree_max_depth: int = 3,
        rule_sources: tuple = ("gb", "rf"),
        max_rules: int | None = 300,
        min_support: float = 0.01,
        max_support: float = 0.99,
        soft_rules: bool = True,
        sharpness: float = 10.0,
        include_linear: bool = True,
        rule_interactions: bool = False,
        max_interaction_rules: int = 30,
        selection: str = "l1",
        alpha: float | None = None,
        l1_ratio: float = 0.8,
        cv: int = 3,
        n_boosting_rounds: int = 200,
        boosting_lr: float = 0.1,
        decorrelation_threshold: float = 0.95,
        refine_thresholds: bool = False,
        refine_steps: int = 50,
        refine_lr: float = 0.01,
        calibrate: bool = False,
        merge_similar: bool = True,
        merge_tolerance: float = 0.05,
        random_state: int | None = None,
        n_jobs: int | None = None,
        class_weight: dict | str | None = None,
    ):
        self.n_estimators = n_estimators
        self.tree_max_depth = tree_max_depth
        self.rule_sources = rule_sources
        self.max_rules = max_rules
        self.min_support = min_support
        self.max_support = max_support
        self.soft_rules = soft_rules
        self.sharpness = sharpness
        self.include_linear = include_linear
        self.rule_interactions = rule_interactions
        self.max_interaction_rules = max_interaction_rules
        self.selection = selection
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.cv = cv
        self.n_boosting_rounds = n_boosting_rounds
        self.boosting_lr = boosting_lr
        self.decorrelation_threshold = decorrelation_threshold
        self.refine_thresholds = refine_thresholds
        self.refine_steps = refine_steps
        self.refine_lr = refine_lr
        self.calibrate = calibrate
        self.merge_similar = merge_similar
        self.merge_tolerance = merge_tolerance
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.class_weight = class_weight

    def fit(self, X, y, feature_names=None, sample_weight=None):
        X, y = check_X_y(X, y, dtype=np.float64)

        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(y)
        self.classes_ = self._label_encoder.classes_
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = X.shape[1]

        if feature_names is not None:
            self.feature_names_in_ = np.array(feature_names)
        elif hasattr(X, "columns"):
            self.feature_names_in_ = np.array(X.columns)
        else:
            self.feature_names_in_ = np.array(
                [f"x{i}" for i in range(self.n_features_in_)]
            )

        # Step 1: generate rules
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rule_ensemble = _generate_rules(
                X, y_encoded, "classification",
                list(self.feature_names_in_),
                self.rule_sources, self.n_estimators, self.tree_max_depth,
                self.random_state,
            )

        # Step 2: dedup, merge, filter
        rule_ensemble = rule_ensemble.deduplicate()
        if self.merge_similar:
            rule_ensemble = RuleEnsemble(
                rules=_merge_similar_rules(
                    rule_ensemble.rules, self.merge_tolerance
                ),
                n_features=rule_ensemble.n_features,
                feature_names=rule_ensemble.feature_names,
            )
        rule_ensemble = rule_ensemble.filter_by_support(
            self.min_support, self.max_support
        )
        if self.max_rules is not None:
            rule_ensemble = rule_ensemble.limit_rules(self.max_rules)

        self.rule_ensemble_ = rule_ensemble
        self.n_rules_ = len(rule_ensemble)

        # Step 3: compile and transform
        self._compiled = _compile_rules(rule_ensemble.rules)
        xform = _soft_transform if self.soft_rules else _hard_transform
        xform_args = (X, self._compiled, self.sharpness) if self.soft_rules else (X, self._compiled)
        X_rules = xform(*xform_args)

        # Step 4: build combined features
        self._n_linear = 0
        if self.include_linear:
            X_linear, self._linear_params = _preprocess_linear(X, fit=True)
            self._n_linear = X_linear.shape[1]
            X_combined = np.hstack([X_linear, X_rules])
        else:
            X_combined = X_rules
            self._linear_params = {}

        self._interaction_pairs = []
        if self.rule_interactions and X_rules.shape[1] >= 2:
            X_inter, self._interaction_pairs = _create_interaction_features(
                X_rules, self.max_interaction_rules
            )
            if X_inter.shape[1] > 0:
                X_combined = np.hstack([X_combined, X_inter])

        # Step 5: standardize + decorrelate for faster solver convergence
        from sklearn.preprocessing import StandardScaler
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_combined)
        X_decorr, self._keep_mask = _decorrelate_features(
            X_scaled, threshold=self.decorrelation_threshold,
        )

        # Step 6: fit selection model on decorrelated features
        n_full = X_scaled.shape[1]

        if self.selection == "boosted":
            coef_r, intercept_r = _fit_boosted_classification(
                X_decorr, y_encoded, self.n_classes_,
                self.n_boosting_rounds, self.boosting_lr,
            )
            self._logistic_model = None
            self.alpha_ = 0.0

            # Expand + unscale coefficients
            scale = self._scaler.scale_
            mean = self._scaler.mean_
            if coef_r.ndim == 1:
                coef_s = np.zeros(n_full)
                coef_s[self._keep_mask] = coef_r
                int_val = intercept_r[0] if isinstance(intercept_r, np.ndarray) else intercept_r
                self.coef_ = coef_s / scale
                self.intercept_ = int_val - np.dot(coef_s, mean / scale)
            else:
                coef_s = np.zeros((coef_r.shape[0], n_full))
                coef_s[:, self._keep_mask] = coef_r
                self.coef_ = coef_s / scale[None, :]
                self.intercept_ = intercept_r - (coef_s / scale[None, :]) @ mean
        else:
            if self.selection == "elasticnet":
                model = _fit_elasticnet_classification(
                    X_decorr, y_encoded, self.l1_ratio, self.cv,
                    self.n_jobs, self.random_state, self.class_weight,
                )
            elif self.selection == "l1":
                model = _fit_l1_classification(
                    X_decorr, y_encoded, self.cv, self.n_jobs,
                    self.random_state, self.class_weight,
                )
            else:
                raise ValueError(f"Unknown selection: {self.selection!r}")

            self._logistic_model = model
            self.alpha_ = 1.0 / np.mean(model.C_)

            # Expand + unscale coefficients from decorrelated→standardized→original
            scale = self._scaler.scale_
            mean = self._scaler.mean_
            if model.coef_.ndim == 1:
                coef_s = np.zeros(n_full)
                coef_s[self._keep_mask] = model.coef_
                self.coef_ = coef_s / scale
                self.intercept_ = model.intercept_ - np.dot(coef_s, mean / scale)
            else:
                coef_s = np.zeros((model.coef_.shape[0], n_full))
                coef_s[:, self._keep_mask] = model.coef_
                self.coef_ = coef_s / scale[None, :]
                self.intercept_ = model.intercept_ - (coef_s / scale[None, :]) @ mean

        # Step 6: optional threshold refinement (binary only for now)
        if (self.refine_thresholds and self.soft_rules
                and self.n_rules_ > 0 and self.n_classes_ == 2):
            coef_flat = self.coef_.ravel() if self.coef_.ndim > 1 else self.coef_
            intercept_scalar = (
                self.intercept_[0] if isinstance(self.intercept_, np.ndarray)
                else self.intercept_
            )
            self._compiled = _refine_thresholds(
                X, y_encoded, self._compiled, coef_flat, intercept_scalar,
                self.sharpness, self._n_linear, True,
                lr=self.refine_lr, steps=self.refine_steps,
            )

        # Step 7: optional calibration
        self._calibrator = None
        if self.calibrate and self._logistic_model is not None:
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.linear_model import LogisticRegression

            cal_base = LogisticRegression(
                penalty="l1", solver="saga",
                C=1.0 / max(self.alpha_, 1e-10), max_iter=1000, tol=1e-3,
            )
            cal_base.fit(X_combined, y_encoded)
            self._calibrator = CalibratedClassifierCV(
                cal_base, method="isotonic", cv=3,
            )
            self._calibrator.fit(X_combined, y_encoded)

        # Step 8: bookkeeping
        self._assign_coefficients()
        self._compute_feature_importances()
        return self

    def _assign_coefficients(self):
        n_lin = self._n_linear
        n_rules = self.n_rules_
        coef = self.coef_

        if self.include_linear:
            if coef.ndim == 1:
                self.linear_coef_ = coef[:n_lin]
                self.rule_coef_ = coef[n_lin:n_lin + n_rules]
            else:
                self.linear_coef_ = np.mean(np.abs(coef[:, :n_lin]), axis=0)
                self.rule_coef_ = np.mean(
                    np.abs(coef[:, n_lin:n_lin + n_rules]), axis=0
                )
        else:
            self.linear_coef_ = np.zeros(self.n_features_in_)
            if coef.ndim == 1:
                self.rule_coef_ = coef[:n_rules]
            else:
                self.rule_coef_ = np.mean(np.abs(coef[:, :n_rules]), axis=0)

        for i, rule in enumerate(self.rule_ensemble_.rules):
            if i < len(self.rule_coef_):
                rule.coefficient = self.rule_coef_[i]

        self.rules_ = [
            r for r in self.rule_ensemble_.rules if abs(r.coefficient) > 1e-10
        ]
        self.n_rules_selected_ = len(self.rules_)

    def _compute_feature_importances(self):
        importances = np.zeros(self.n_features_in_)
        if self.include_linear:
            importances += np.abs(self.linear_coef_)
        for rule in self.rule_ensemble_.rules:
            if abs(rule.coefficient) > 1e-10:
                fi = rule.feature_indices
                if fi:
                    imp = abs(rule.coefficient) / len(fi)
                    for idx in fi:
                        importances[idx] += imp
        total = np.sum(importances)
        if total > 0:
            importances /= total
        self.feature_importances_ = importances

    def _transform_combined(self, X):
        xform = _soft_transform if self.soft_rules else _hard_transform
        xform_args = (X, self._compiled, self.sharpness) if self.soft_rules else (X, self._compiled)
        X_rules = xform(*xform_args)

        parts = []
        if self.include_linear:
            X_linear, _ = _preprocess_linear(
                X, fit=False, params=self._linear_params
            )
            parts.append(X_linear)
        parts.append(X_rules)

        if self._interaction_pairs:
            X_inter = _apply_interaction_pairs(X_rules, self._interaction_pairs)
            if X_inter.shape[1] > 0:
                parts.append(X_inter)

        return np.hstack(parts)

    def _raw_decision(self, X_combined):
        """Compute raw decision values using stored coef_ and intercept_."""
        coef = self.coef_
        intercept = self.intercept_
        if coef.ndim == 1:
            return X_combined @ coef + intercept
        else:
            return X_combined @ coef.T + intercept

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X, dtype=np.float64)
        X_combined = self._transform_combined(X)
        raw = self._raw_decision(X_combined)

        if self.coef_.ndim == 1 or self.coef_.shape[0] == 1:
            r = raw.ravel()
            y_pred = (r > 0).astype(int)
        else:
            y_pred = np.argmax(raw, axis=1)

        return self._label_encoder.inverse_transform(y_pred)

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X, dtype=np.float64)
        X_combined = self._transform_combined(X)

        if self._calibrator is not None:
            return self._calibrator.predict_proba(X_combined)

        raw = self._raw_decision(X_combined)
        if self.coef_.ndim == 1 or self.coef_.shape[0] == 1:
            r = raw.ravel()
            p1 = 1.0 / (1.0 + np.exp(-np.clip(r, -30, 30)))
            return np.column_stack([1 - p1, p1])
        else:
            exp_raw = np.exp(raw - np.max(raw, axis=1, keepdims=True))
            return exp_raw / exp_raw.sum(axis=1, keepdims=True)

    def predict_log_proba(self, X):
        return np.log(np.maximum(self.predict_proba(X), 1e-15))

    def transform(self, X):
        check_is_fitted(self)
        X = check_array(X, dtype=np.float64)
        xform = _soft_transform if self.soft_rules else _hard_transform
        xform_args = (X, self._compiled, self.sharpness) if self.soft_rules else (X, self._compiled)
        return xform(*xform_args)

    def get_rules(self, exclude_zero_coef=True, sort_by="importance"):
        check_is_fitted(self)
        if exclude_zero_coef:
            rules = [r for r in self.rule_ensemble_.rules if abs(r.coefficient) > 1e-10]
        else:
            rules = list(self.rule_ensemble_.rules)

        key_map = {
            "importance": lambda r: r.importance,
            "support": lambda r: r.support,
            "coefficient": lambda r: r.coefficient,
            "length": lambda r: r.length,
        }
        reverse = sort_by != "length"
        rules = sorted(rules, key=key_map.get(sort_by, key_map["importance"]),
                        reverse=reverse)
        return [r.to_dict() for r in rules]

    def summary(self):
        check_is_fitted(self)
        lines = [
            "=" * 60,
            "RuleFit++ Classifier Summary",
            "=" * 60, "",
            "Model Statistics:",
            "-" * 40,
            f"  Number of classes:         {self.n_classes_}",
            f"  Rule sources:              {', '.join(self.rule_sources)}",
            f"  Soft rules:                {self.soft_rules}",
            f"  Selection method:           {self.selection}",
            f"  Total rules extracted:      {self.n_rules_}",
            f"  Rules with non-zero coef:   {self.n_rules_selected_}",
            f"  Alpha (regularization):     {self.alpha_:.6f}",
            "",
        ]
        lines.append("Top Rules by Importance:")
        lines.append("-" * 40)
        for i, rd in enumerate(self.get_rules()[:20]):
            lines.append(f"  [{i+1}] {rd['rule']}")
            lines.append(
                f"      Coef: {rd['coefficient']:+.4f}, "
                f"Support: {rd['support']:.3f}"
            )
            lines.append("")

        lines.append("Feature Importances:")
        lines.append("-" * 40)
        pairs = sorted(
            zip(self.feature_names_in_, self.feature_importances_),
            key=lambda x: x[1], reverse=True,
        )
        for name, imp in pairs[:10]:
            lines.append(f"  {name:30s} {imp:.4f}")
        lines.extend(["", "=" * 60])
        return "\n".join(lines)
