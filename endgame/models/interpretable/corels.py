"""CORELS: Certifiably Optimal Rule Lists.

Supports two backends:

* **cli** — Calls the CORELS C++ branch-and-bound binary for
  *certifiably optimal* rule lists.  The binary is built from source
  automatically on first use (requires ``g++``; GMP is optional).
* **greedy** — Pure-Python greedy sequential covering with beam
  search.  Fast and dependency-free, but not provably optimal.

``backend="auto"`` (the default) tries the CLI backend first and
falls back to greedy if the binary cannot be found or built.

References
----------
- Angelino et al. "Learning Certifiably Optimal Rule Lists for
  Categorical Data" (JMLR 2018)
- https://github.com/corels/corels

Example
-------
>>> from endgame.models.interpretable import CORELSClassifier
>>> clf = CORELSClassifier(max_card=2, c=0.001)
>>> clf.fit(X_train, y_train)
>>> print(clf.summary())
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from endgame.core.glassbox import GlassboxMixin
from typing import Any

logger = logging.getLogger(__name__)

_CORELS_CACHE_DIR = Path.home() / ".cache" / "endgame" / "corels"

# CLI output pattern:  if ({...}) then ({...})  /  else if  /  else
_RE_IF = re.compile(
    r"(?:else\s+)?if\s+\((.+?)\)\s+then\s+\((.+?)\)", re.IGNORECASE,
)
_RE_ELSE = re.compile(r"else\s+\((.+?)\)", re.IGNORECASE)


# ======================================================================
# Helper: locate / build the CORELS binary
# ======================================================================

def _find_corels_binary(hint: str | None = None) -> str | None:
    """Return the path to a usable ``corels`` binary, or *None*."""
    if hint and os.path.isfile(hint) and os.access(hint, os.X_OK):
        return hint

    # Check workspace-local cache (set by endgame build)
    workspace_bin = Path(__file__).resolve().parents[3] / ".cache" / "corels" / "repo" / "src" / "corels"
    if workspace_bin.is_file() and os.access(str(workspace_bin), os.X_OK):
        return str(workspace_bin)

    # Check user-level cache
    cached = _CORELS_CACHE_DIR / "repo" / "src" / "corels"
    if cached.is_file() and os.access(str(cached), os.X_OK):
        return str(cached)

    # Check system PATH
    on_path = shutil.which("corels")
    if on_path:
        return on_path

    return None


def _build_corels_binary() -> str:
    """Clone the CORELS repo and compile the binary.  Returns the path."""
    repo_dir = _CORELS_CACHE_DIR / "repo"
    src_dir = repo_dir / "src"
    binary = src_dir / "corels"

    if binary.is_file() and os.access(str(binary), os.X_OK):
        return str(binary)

    if not shutil.which("g++"):
        raise RuntimeError("g++ is required to build CORELS from source")

    repo_dir.mkdir(parents=True, exist_ok=True)

    if not (src_dir / "main.cc").is_file():
        logger.info("Cloning CORELS repository …")
        subprocess.check_call(
            ["git", "clone", "--depth", "1",
             "https://github.com/corels/corels.git", str(repo_dir)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    logger.info("Building CORELS binary …")
    env = os.environ.copy()
    # Build without GMP if the header isn't available
    has_gmp_h = (
        Path("/usr/include/gmp.h").is_file()
        or Path("/usr/local/include/gmp.h").is_file()
    )
    if not has_gmp_h:
        env["NGMP"] = "1"

    subprocess.check_call(
        ["make", "corels"], cwd=str(src_dir), env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    if not binary.is_file():
        raise RuntimeError("CORELS binary was not produced by make")

    return str(binary)


# ======================================================================
# Main classifier
# ======================================================================

class CORELSClassifier(GlassboxMixin, ClassifierMixin, BaseEstimator):
    """Certifiably Optimal Rule List Classifier.

    Parameters
    ----------
    c : float, default=0.001
        Regularization.  Higher → simpler rule lists.
    n_iter : int, default=100000
        Max trie nodes (CLI) or candidate evaluations (greedy).
    map_type : str, default="prefix"
        Symmetry-aware map: ``"none"``, ``"prefix"``, ``"captured"``.
    policy : str, default="lower_bound"
        Search policy: ``"bfs"``, ``"curious"``, ``"lower_bound"``,
        ``"objective"``, ``"dfs"``.
    max_card : int, default=2
        Maximum conjunction cardinality in a single rule.
    min_support : float, default=0.01
        Minimum fraction of samples a rule must cover.
    beam_width : int, default=50
        Beam width for the greedy backend.
    auto_discretize : bool, default=True
        Bin continuous features into binary indicators.
    n_bins : int, default=5
        Number of bins when ``auto_discretize=True``.
    discretize_strategy : str, default="quantile"
        ``"uniform"``, ``"quantile"``, or ``"kmeans"``.
    backend : str, default="auto"
        ``"auto"`` (try CLI, fallback greedy), ``"cli"`` (require CLI),
        or ``"greedy"`` (pure Python only).
    corels_binary : str or None
        Explicit path to the ``corels`` executable.
    random_state : int or None
        Random seed.

    Attributes
    ----------
    rules_ : list[dict]
        Learned rules (antecedent, consequent, support, accuracy).
    rule_list_ : str
        Human-readable rule list.
    backend_used_ : str
        ``"cli"`` or ``"greedy"``—whichever actually ran.
    optimal_ : bool
        *True* when the CLI backend certified optimality.
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        c: float = 0.001,
        n_iter: int = 100_000,
        map_type: str = "prefix",
        policy: str = "lower_bound",
        verbosity: list[str] | None = None,
        ablation: int = 0,
        max_card: int = 2,
        min_support: float = 0.01,
        beam_width: int = 50,
        auto_discretize: bool = True,
        n_bins: int = 5,
        discretize_strategy: str = "quantile",
        backend: str = "auto",
        corels_binary: str | None = None,
        random_state: int | None = None,
    ):
        self.c = c
        self.n_iter = n_iter
        self.map_type = map_type
        self.policy = policy
        self.verbosity = verbosity
        self.ablation = ablation
        self.max_card = max_card
        self.min_support = min_support
        self.beam_width = beam_width
        self.auto_discretize = auto_discretize
        self.n_bins = n_bins
        self.discretize_strategy = discretize_strategy
        self.backend = backend
        self.corels_binary = corels_binary
        self.random_state = random_state

    # ------------------------------------------------------------------
    # fit / predict
    # ------------------------------------------------------------------

    def fit(self, X, y, feature_names=None, sample_weight=None):
        """Fit the rule list classifier."""
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]

        self._label_encoder = LabelEncoder()
        y_enc = self._label_encoder.fit_transform(y)
        self.classes_ = self._label_encoder.classes_
        if len(self.classes_) != 2:
            raise ValueError(
                f"CORELSClassifier only supports binary classification. "
                f"Got {len(self.classes_)} classes."
            )

        if feature_names is not None:
            self._original_feature_names = list(feature_names)
        elif hasattr(X, "columns"):
            self._original_feature_names = list(X.columns)
        else:
            self._original_feature_names = [
                f"x{i}" for i in range(self.n_features_in_)
            ]

        if self.auto_discretize:
            X_bin, self.feature_names_ = self._discretize(X)
        else:
            X_bin = (X > 0.5).astype(np.int8)
            self.feature_names_ = self._original_feature_names
            self.discretizer_ = None

        # --- Choose backend ---
        if self.backend in ("auto", "cli"):
            try:
                self._fit_cli(X_bin, y_enc)
                return self
            except Exception as exc:
                if self.backend == "cli":
                    raise RuntimeError(f"CORELS CLI failed: {exc}") from exc
                logger.info("CLI backend unavailable (%s), using greedy", exc)

        self._fit_greedy(X_bin, y_enc)
        return self

    def predict(self, X):
        """Predict class labels."""
        check_is_fitted(self, "rules_")
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}."
            )

        if self.auto_discretize and self.discretizer_ is not None:
            X_bin = self.discretizer_.transform(X).astype(np.int8)
        else:
            X_bin = (X > 0.5).astype(np.int8)

        y_enc = self._apply_rules(X_bin)
        return self._label_encoder.inverse_transform(y_enc)

    def predict_proba(self, X):
        """Class probabilities (hard 0/1)."""
        y_pred = self.predict(X)
        y_enc = self._label_encoder.transform(y_pred)
        proba = np.zeros((len(y_pred), 2))
        proba[np.arange(len(y_pred)), y_enc] = 1.0
        return proba

    # ------------------------------------------------------------------
    # Discretization
    # ------------------------------------------------------------------

    def _discretize(self, X):
        self.discretizer_ = KBinsDiscretizer(
            n_bins=self.n_bins, encode="onehot-dense",
            strategy=self.discretize_strategy,
            random_state=self.random_state,
        )
        X_binned = self.discretizer_.fit_transform(X)
        names: list[str] = []
        for fi, edges in enumerate(self.discretizer_.bin_edges_):
            fname = self._original_feature_names[fi]
            for bi in range(len(edges) - 1):
                if bi == 0:
                    names.append(f"{fname}<={edges[bi+1]:.3g}")
                elif bi == len(edges) - 2:
                    names.append(f"{fname}>{edges[bi]:.3g}")
                else:
                    names.append(f"{edges[bi]:.3g}<{fname}<={edges[bi+1]:.3g}")
        return X_binned.astype(np.int8), names

    # ==================================================================
    # CLI backend — calls the CORELS C++ binary
    # ==================================================================

    def _fit_cli(self, X_bin, y_enc):
        binary = _find_corels_binary(self.corels_binary)
        if binary is None:
            binary = _build_corels_binary()

        with tempfile.TemporaryDirectory(prefix="corels_") as tmpdir:
            out_path, label_path = self._write_corels_files(
                X_bin, y_enc, tmpdir,
            )
            stdout = self._run_corels(binary, out_path, label_path, tmpdir)
            self._parse_corels_output(stdout, y_enc)

        self.backend_used_ = "cli"

    # --- Data file writing -------------------------------------------

    def _write_corels_files(self, X_bin, y_enc, tmpdir):
        """Write ``.out`` and ``.label`` files in CORELS format.

        The ``.out`` file contains one line per mined antecedent.
        We mine all single features and all conjunctions up to
        *max_card* that meet the *min_support* threshold.
        """
        n_samples, n_feats = X_bin.shape
        min_cap = max(1, int(self.min_support * n_samples))

        # Sanitise feature names (no spaces allowed)
        safe_names = [n.replace(" ", "_") for n in self.feature_names_]

        # Mine antecedents
        antecedents: list[tuple[str, np.ndarray]] = []
        # Cardinality 1
        for fi in range(n_feats):
            col = X_bin[:, fi].astype(np.int8)
            if col.sum() >= min_cap and (n_samples - col.sum()) >= 0:
                antecedents.append((safe_names[fi], col))

        # Higher cardinalities
        for card in range(2, self.max_card + 1):
            base_indices = [
                fi for fi in range(n_feats)
                if X_bin[:, fi].sum() >= min_cap
            ]
            for combo in combinations(base_indices, card):
                mask = np.ones(n_samples, dtype=np.int8)
                for fi in combo:
                    mask &= X_bin[:, fi]
                if mask.sum() >= min_cap:
                    name = ",".join(safe_names[fi] for fi in combo)
                    antecedents.append((name, mask))

        # Write .out file
        out_path = os.path.join(tmpdir, "data.out")
        with open(out_path, "w") as f:
            for name, bitvec in antecedents:
                bits = " ".join(str(int(b)) for b in bitvec)
                f.write(f"{{{name}}} {bits}\n")

        # Build reverse mapping: sanitized_name → original_name
        self._sanitized_to_original = {}
        for name, _ in antecedents:
            # For single features, map the sanitized name back
            parts = name.split(",")
            for part in parts:
                # Map each sanitized feature name back to original
                for orig_name in self.feature_names_:
                    if orig_name.replace(" ", "_") == part:
                        self._sanitized_to_original[part] = orig_name
                        break
            # Also map the full conjunction
            full_orig = ",".join(
                self._sanitized_to_original.get(p, p) for p in parts
            )
            self._sanitized_to_original[name] = full_orig

        # Write .label file
        label_path = os.path.join(tmpdir, "data.label")
        neg_label = str(self.classes_[0]).replace(" ", "_")
        pos_label = str(self.classes_[1]).replace(" ", "_")
        neg_bits = " ".join(str(1 - int(b)) for b in y_enc)
        pos_bits = " ".join(str(int(b)) for b in y_enc)
        with open(label_path, "w") as f:
            f.write(f"{{label={neg_label}}} {neg_bits}\n")
            f.write(f"{{label={pos_label}}} {pos_bits}\n")

        return out_path, label_path

    # --- Binary invocation -------------------------------------------

    _POLICY_MAP = {
        "curious": "-c 1",
        "lower_bound": "-c 2",
        "objective": "-c 3",
        "dfs": "-c 4",
        "bfs": "-b",
    }
    _MAP_TYPE = {"none": "0", "prefix": "1", "captured": "2"}

    def _run_corels(self, binary, out_path, label_path, tmpdir):
        cmd = [binary]
        pol = self._POLICY_MAP.get(self.policy, "-c 2")
        cmd.extend(pol.split())
        cmd.extend(["-p", self._MAP_TYPE.get(self.map_type, "1")])
        cmd.extend(["-r", str(self.c)])
        cmd.extend(["-n", str(self.n_iter)])
        cmd.extend(["-v", "silent"])
        cmd.extend(["-a", str(self.ablation)])
        cmd.extend([out_path, label_path])

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"CORELS exited with code {result.returncode}: "
                f"{result.stderr.strip()}"
            )
        return result.stdout

    # --- Output parsing -----------------------------------------------

    def _parse_corels_output(self, stdout: str, y_enc: np.ndarray):
        """Parse the text output of the CORELS binary into ``rules_``."""
        lines = stdout.strip().splitlines()

        self.rules_ = []
        self.optimal_ = False
        n_samples = len(y_enc)

        for line in lines:
            if "OPTIMAL RULE LIST" in line:
                self.optimal_ = True
                continue

            m_if = _RE_IF.search(line)
            if m_if:
                antecedent_raw = m_if.group(1).strip()
                label_raw = m_if.group(2).strip()
                label_enc = self._label_from_string(label_raw)
                antecedent = self._clean_antecedent(antecedent_raw)
                self.rules_.append({
                    "antecedent": antecedent,
                    "consequent": int(label_enc),
                    "support": 0.0,
                    "accuracy": 0.0,
                })
                continue

            m_else = _RE_ELSE.search(line)
            if m_else:
                label_raw = m_else.group(1).strip()
                label_enc = self._label_from_string(label_raw)
                self.rules_.append({
                    "antecedent": "default",
                    "consequent": int(label_enc),
                    "support": 0.0,
                    "accuracy": 0.0,
                })
                continue

        if not self.rules_:
            raise RuntimeError(
                "Could not parse any rules from CORELS output:\n" + stdout
            )

        self._build_rule_list_str()

    def _label_from_string(self, label_str: str) -> int:
        """Map a label string from CORELS output to encoded int."""
        clean = label_str.strip().strip("{}")
        clean = clean.replace("label=", "").replace("_", " ").strip()
        for idx, cls in enumerate(self.classes_):
            if str(cls).strip() == clean:
                return idx
        return 1 if "yes" in clean.lower() or "1" in clean else 0

    def _clean_antecedent(self, raw: str) -> str:
        """Clean a raw antecedent string from CORELS output.

        Strips braces, splits conjunctions on ``,``, and maps each part
        back to its original (un-sanitized) feature name.
        """
        s = raw.strip().strip("{}")
        parts = [p.strip() for p in s.split(",")]
        clean_parts = []
        for part in parts:
            clean_parts.append(self._sanitized_to_original.get(part, part))
        return " AND ".join(clean_parts)

    # ==================================================================
    # Greedy backend — pure Python
    # ==================================================================

    def _fit_greedy(self, X_bin, y_enc):
        n_samples = len(y_enc)
        min_cap = max(1, int(self.min_support * n_samples))
        remaining = np.ones(n_samples, dtype=bool)
        self.rules_ = []
        n_feats = X_bin.shape[1]
        iters_used = 0

        while remaining.sum() > 0 and iters_used < self.n_iter:
            best = self._greedy_find_best(
                X_bin, y_enc, remaining, n_feats, min_cap,
            )
            iters_used += best.get("iters", 0)
            if best["feature_indices"] is None:
                break
            captured = remaining & best["mask"]
            n_cap = captured.sum()
            if n_cap < min_cap:
                break

            label = best["label"]
            parts = [self.feature_names_[fi] for fi in best["feature_indices"]]
            self.rules_.append({
                "antecedent": " AND ".join(parts),
                "consequent": int(label),
                "support": float(n_cap / n_samples),
                "accuracy": float((y_enc[captured] == label).sum() / n_cap),
            })
            remaining[captured] = False

        default_label = (
            int(np.round(y_enc[remaining].mean()))
            if remaining.any() else int(np.round(y_enc.mean()))
        )
        default_acc = (
            float((y_enc[remaining] == default_label).mean())
            if remaining.any() else 1.0
        )
        self.rules_.append({
            "antecedent": "default",
            "consequent": default_label,
            "support": float(remaining.sum() / n_samples),
            "accuracy": default_acc,
        })

        self._build_rule_list_str()
        self.backend_used_ = "greedy"
        self.optimal_ = False

    def _greedy_find_best(self, X, y, remaining, n_feats, min_cap):
        y_rem, X_rem = y[remaining], X[remaining]
        n_rem = int(remaining.sum())
        best_score = -np.inf
        best: dict = {"feature_indices": None, "mask": None, "label": 0, "iters": 0}
        iters = 0

        # (score, feat_tuple, label, mask_on_remaining)
        beam: list[tuple[float, tuple[int, ...], int, np.ndarray]] = []

        for fi in range(n_feats):
            mask_loc = X_rem[:, fi] == 1
            n_cap = int(mask_loc.sum())
            iters += 1
            if n_cap < min_cap:
                continue
            label, score = self._score(y_rem[mask_loc], n_rem)
            beam.append((score, (fi,), label, mask_loc))
            if score > best_score:
                best_score = score
                full = np.zeros(len(y), dtype=bool)
                full[remaining] = mask_loc
                best = {"feature_indices": (fi,), "mask": full, "label": label}

        for _card in range(2, self.max_card + 1):
            if iters >= self.n_iter or not beam:
                break
            beam.sort(key=lambda t: -t[0])
            parents = beam[: self.beam_width]
            beam = []
            for _ps, pfeats, _pl, pmask in parents:
                for fi in range(pfeats[-1] + 1, n_feats):
                    child = pmask & (X_rem[:, fi] == 1)
                    n_cap = int(child.sum())
                    iters += 1
                    if n_cap < min_cap:
                        continue
                    label, score = self._score(y_rem[child], n_rem)
                    cfeats = pfeats + (fi,)
                    beam.append((score, cfeats, label, child))
                    if score > best_score:
                        best_score = score
                        full = np.zeros(len(y), dtype=bool)
                        full[remaining] = child
                        best = {"feature_indices": cfeats, "mask": full, "label": label}
                    if iters >= self.n_iter:
                        break
                if iters >= self.n_iter:
                    break

        best["iters"] = iters
        return best

    @staticmethod
    def _score(y_cap, n_total):
        n = len(y_cap)
        if n == 0:
            return 0, -np.inf
        n_pos = int(y_cap.sum())
        n_neg = n - n_pos
        label = 1 if n_pos >= n_neg else 0
        correct = max(n_pos, n_neg)
        return label, correct / n_total

    # ==================================================================
    # Inference
    # ==================================================================

    def _apply_rules(self, X_bin):
        n = X_bin.shape[0]
        preds = np.full(n, -1, dtype=int)
        remaining = np.ones(n, dtype=bool)
        for rule in self.rules_:
            if rule["antecedent"] == "default":
                preds[remaining] = rule["consequent"]
                break
            mask = self._eval_antecedent(X_bin, rule["antecedent"])
            captured = remaining & mask
            preds[captured] = rule["consequent"]
            remaining[captured] = False
        preds[preds == -1] = self.rules_[-1]["consequent"]
        return preds

    def _eval_antecedent(self, X, antecedent):
        parts = [p.strip() for p in antecedent.split(" AND ")]
        mask = np.ones(X.shape[0], dtype=bool)
        for part in parts:
            if part in self.feature_names_:
                mask &= X[:, self.feature_names_.index(part)] == 1
        return mask

    # ==================================================================
    # Display
    # ==================================================================

    def _build_rule_list_str(self):
        lines = []
        for i, r in enumerate(self.rules_):
            if r["antecedent"] == "default":
                lines.append(f"ELSE predict {self.classes_[r['consequent']]}")
            else:
                pfx = "IF" if i == 0 else "ELSE IF"
                lines.append(
                    f"{pfx} {r['antecedent']} "
                    f"THEN predict {self.classes_[r['consequent']]}"
                )
        self.rule_list_ = "\n".join(lines)

    def get_rules(self):
        """Return the learned rules."""
        check_is_fitted(self, "rules_")
        return self.rules_

    def summary(self):
        """Human-readable summary."""
        check_is_fitted(self, "rules_")
        tag = "OPTIMAL" if getattr(self, "optimal_", False) else "GREEDY"
        lines = ["=" * 60, f"CORELS Rule List ({tag})", "=" * 60, ""]
        for i, r in enumerate(self.rules_):
            if r["antecedent"] == "default":
                lines.append(f"ELSE predict {self.classes_[r['consequent']]}")
            else:
                pfx = "IF" if i == 0 else "ELSE IF"
                lines.append(
                    f"{pfx} {r['antecedent']} "
                    f"THEN predict {self.classes_[r['consequent']]}"
                )
        lines += [
            "",
            f"Regularization (c): {self.c}",
            f"Number of rules: {len(self.rules_)}",
            f"Backend: {getattr(self, 'backend_used_', 'unknown')}",
            "=" * 60,
        ]
        return "\n".join(lines)

    def __repr__(self):
        return (
            f"CORELSClassifier(c={self.c}, max_card={self.max_card}, "
            f"backend='{self.backend}')"
        )

    _structure_type = "rules"

    def _structure_content(self) -> dict[str, Any]:
        check_is_fitted(self, "rules_")
        rule_dicts = []
        default = None
        for i, r in enumerate(self.rules_):
            consequent = r["consequent"]
            class_label = self.classes_[consequent]
            entry = {
                "antecedent": str(r["antecedent"]),
                "consequent_class": class_label.item() if hasattr(class_label, "item") else class_label,
                "position": i,
            }
            if r["antecedent"] == "default":
                default = entry
            else:
                rule_dicts.append(entry)
        return {
            "rules": rule_dicts,
            "default": default,
            "n_rules": len(rule_dicts),
            "regularization": float(self.c),
            "max_cardinality": int(self.max_card),
            "backend": getattr(self, "backend_used_", "unknown"),
            "optimal": bool(getattr(self, "optimal_", False)),
        }
