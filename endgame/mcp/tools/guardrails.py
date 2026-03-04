"""Guardrails tool: check_data_quality."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from endgame.mcp.server import capture_stdout, error_response, ok_response
from endgame.mcp.session import SessionManager


def register(mcp: FastMCP, session: SessionManager) -> None:

    @mcp.tool()
    def check_data_quality(
        dataset_id: str,
        checks: str = "default",
        fix: bool = False,
    ) -> str:
        """Run data quality and leakage detection on a loaded dataset.

        Returns a JSON report of detected issues. If fix=True, also removes
        flagged features from the dataset in-place.

        checks can be 'default', 'all', or a comma-separated list of check names.
        """
        try:
            ds = session.get_dataset(dataset_id)

            with capture_stdout():
                import json

                from endgame.guardrails import LeakageDetector

                if ds.target_column and ds.target_column in ds.df.columns:
                    X = ds.df.drop(columns=[ds.target_column])
                    y = ds.df[ds.target_column]
                else:
                    X = ds.df
                    y = None

                check_list: str | list[str] = checks
                if checks not in ("default", "all"):
                    check_list = [c.strip() for c in checks.split(",")]

                mode = "fix" if fix else "detect"
                detector = LeakageDetector(
                    mode=mode,
                    checks=check_list,
                    time_budget=30,
                    verbose=False,
                )
                detector.fit(X, y)
                report = detector.report_

                if fix and report.features_to_drop:
                    X_clean = detector.transform(X)
                    if ds.target_column:
                        import pandas as pd
                        ds.df = pd.concat([X_clean, y], axis=1)
                    else:
                        ds.df = X_clean

                result = report.to_dict()
                if fix:
                    result["features_removed"] = report.features_to_drop

                return ok_response(result)

        except KeyError:
            return error_response("not_found", f"Dataset '{dataset_id}' not found")
        except Exception as e:
            return error_response("guardrails_error", str(e))
