from __future__ import annotations

"""Pipeline persistence executor for AutoML.

Auto-saves the best model, ensemble, and pipeline metadata at the end
of the AutoML run so results survive process exits.
"""

import logging
import time
from pathlib import Path
from typing import Any

from endgame.automl.orchestrator import BaseStageExecutor, StageResult

logger = logging.getLogger(__name__)


class PersistenceExecutor(BaseStageExecutor):
    """Save the best pipeline, ensemble, and metadata to disk.

    Parameters
    ----------
    output_dir : str or Path, optional
        Directory to save artifacts.  If ``None``, persistence is skipped
        (the executor becomes a no-op).
    save_ensemble : bool, default=True
        Whether to save the ensemble alongside the best single model.
    save_top_k : int, default=3
        Save the top-k individual models (by score).
    """

    def __init__(
        self,
        output_dir: str | Path | None = None,
        save_ensemble: bool = True,
        save_top_k: int = 3,
    ):
        self.output_dir = Path(output_dir) if output_dir else None
        self.save_ensemble = save_ensemble
        self.save_top_k = save_top_k

    def execute(
        self,
        context: dict[str, Any],
        time_budget: float,
    ) -> StageResult:
        """Persist best models and ensemble to disk.

        Reads ``trained_models``, ``results``, ``ensemble``, and
        ``preprocessor`` from context.
        """
        start = time.time()

        if self.output_dir is None:
            return StageResult(
                stage_name="persistence",
                success=True,
                duration=time.time() - start,
                output={"saved_path": None},
                metadata={"reason": "no output_dir configured"},
            )

        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)

            trained_models = context.get("trained_models", {})
            results = context.get("results", [])
            ensemble = context.get("ensemble")
            preprocessor = context.get("preprocessor")

            saved_artifacts: list[str] = []

            # Rank models by score
            successful = sorted(
                [r for r in results if r.success],
                key=lambda r: r.score,
                reverse=True,
            )

            # Save top-k individual models
            models_dir = self.output_dir / "models"
            models_dir.mkdir(exist_ok=True)
            saved_models = 0

            for r in successful[:self.save_top_k]:
                name = r.config.model_name
                model = trained_models.get(name)
                if model is None:
                    continue

                model_path = models_dir / name
                model_path.mkdir(exist_ok=True)

                try:
                    from endgame.persistence import save as eg_save
                    eg_save(model, model_path / "model.egm")
                    saved_artifacts.append(f"models/{name}/model.egm")
                except Exception:
                    # Fallback to pickle
                    import pickle
                    with open(model_path / "model.pkl", "wb") as f:
                        pickle.dump(model, f)
                    saved_artifacts.append(f"models/{name}/model.pkl")

                # Save metadata
                import json
                meta = {
                    "model_name": name,
                    "score": r.score,
                    "fit_time": r.fit_time,
                    "config": r.config.to_dict(),
                }
                with open(model_path / "meta.json", "w") as f:
                    json.dump(meta, f, indent=2, default=str)
                saved_artifacts.append(f"models/{name}/meta.json")
                saved_models += 1

            # Save ensemble
            if self.save_ensemble and ensemble is not None:
                try:
                    from endgame.persistence import save as eg_save
                    eg_save(ensemble, self.output_dir / "ensemble.egm")
                    saved_artifacts.append("ensemble.egm")
                except Exception:
                    import pickle
                    with open(self.output_dir / "ensemble.pkl", "wb") as f:
                        pickle.dump(ensemble, f)
                    saved_artifacts.append("ensemble.pkl")

            # Save preprocessor
            if preprocessor is not None:
                try:
                    from endgame.persistence import save as eg_save
                    eg_save(preprocessor, self.output_dir / "preprocessor.egm")
                    saved_artifacts.append("preprocessor.egm")
                except Exception:
                    import pickle
                    with open(self.output_dir / "preprocessor.pkl", "wb") as f:
                        pickle.dump(preprocessor, f)
                    saved_artifacts.append("preprocessor.pkl")

            # Save HTML report if report data is available
            report = context.get("report")
            if report is not None and hasattr(report, "save_html"):
                try:
                    report.save_html(str(self.output_dir / "report.html"))
                    saved_artifacts.append("report.html")
                except Exception as e:
                    logger.debug(f"HTML report generation failed: {e}")

            # Save pipeline summary
            import json
            summary = {
                "n_models_saved": saved_models,
                "ensemble_saved": ensemble is not None and self.save_ensemble,
                "best_model": successful[0].config.model_name if successful else None,
                "best_score": successful[0].score if successful else None,
                "artifacts": saved_artifacts,
            }
            with open(self.output_dir / "pipeline_summary.json", "w") as f:
                json.dump(summary, f, indent=2, default=str)
            saved_artifacts.append("pipeline_summary.json")

            duration = time.time() - start
            logger.info(
                f"Pipeline saved to {self.output_dir}: "
                f"{len(saved_artifacts)} artifacts"
            )

            return StageResult(
                stage_name="persistence",
                success=True,
                duration=duration,
                output={
                    "saved_path": str(self.output_dir),
                    "saved_artifacts": saved_artifacts,
                },
                metadata={"n_artifacts": len(saved_artifacts)},
            )

        except Exception as e:
            logger.warning(f"Pipeline persistence failed: {e}")
            return StageResult(
                stage_name="persistence",
                success=False,
                duration=time.time() - start,
                error=str(e),
            )
