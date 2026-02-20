"""Pareto frontier tracking for symbolic regression.

Maintains the best equation at each complexity level across all
iterations, enabling the loss-vs-complexity trade-off analysis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from endgame.models.symbolic._expression import (
    Node,
    complexity,
    to_string,
)
from endgame.models.symbolic._population import clone_tree


@dataclass
class ParetoEntry:
    """Single entry on the Pareto frontier."""
    tree: Node
    loss: float
    complexity: int
    equation: str


class ParetoFrontier:
    """Track best equation at each complexity level.

    After the GP run, call ``to_dataframe()`` to get a pandas DataFrame
    compatible with the old PySR ``equations_`` API.
    """

    def __init__(self):
        self._best: dict[int, ParetoEntry] = {}

    def update(self, tree: Node, loss: float,
               feature_names: list[str] | None = None) -> bool:
        """Try to insert *tree* into the frontier.

        Returns True if the entry was added/updated.
        """
        c = complexity(tree)
        if c in self._best and self._best[c].loss <= loss:
            return False
        eq_str = to_string(tree, feature_names)
        self._best[c] = ParetoEntry(
            tree=clone_tree(tree),
            loss=loss,
            complexity=c,
            equation=eq_str,
        )
        return True

    def get_best(self, model_selection: str = "best") -> ParetoEntry | None:
        """Return the best entry according to *model_selection*.

        Parameters
        ----------
        model_selection : str
            "best"  – lowest loss regardless of complexity.
            "score" – best loss / complexity trade-off (knee point).
        """
        if not self._best:
            return None
        entries = sorted(self._best.values(), key=lambda e: e.loss)
        if model_selection == "score":
            # Simple knee-point heuristic: minimize loss * log(complexity + 1)
            scored = [(e.loss * np.log(e.complexity + 1), e) for e in entries]
            return min(scored, key=lambda t: t[0])[1]
        return entries[0]

    def get_at_complexity(self, c: int) -> ParetoEntry | None:
        return self._best.get(c)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert frontier to a DataFrame mimicking old PySR format."""
        if not self._best:
            return pd.DataFrame(columns=["equation", "loss", "complexity"])
        rows = []
        for c in sorted(self._best.keys()):
            e = self._best[c]
            rows.append({
                "equation": e.equation,
                "loss": e.loss,
                "complexity": e.complexity,
            })
        return pd.DataFrame(rows)

    def get_pareto_optimal(self) -> pd.DataFrame:
        """Return only the Pareto-optimal subset of the frontier."""
        df = self.to_dataframe()
        if df.empty:
            return df
        # Already keyed by complexity (one per), just need to remove dominated
        mask = []
        for i, row in df.iterrows():
            dominated = False
            for j, other in df.iterrows():
                if i == j:
                    continue
                if (other["loss"] <= row["loss"] and
                    other["complexity"] <= row["complexity"] and
                    (other["loss"] < row["loss"] or other["complexity"] < row["complexity"])):
                    dominated = True
                    break
            mask.append(not dominated)
        return df[mask].reset_index(drop=True)
