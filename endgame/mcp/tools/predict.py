"""Prediction tool: predict."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from endgame.mcp.server import capture_stdout, error_response, ok_response
from endgame.mcp.session import SessionManager


def register(mcp: FastMCP, session: SessionManager) -> None:

    @mcp.tool()
    def predict(
        model_id: str,
        dataset_id: str,
        output_path: str | None = None,
        include_probabilities: bool = False,
    ) -> str:
        """Generate predictions for a dataset using a trained model. Optionally saves to CSV."""
        try:
            model_art = session.get_model(model_id)
            ds = session.get_dataset(dataset_id)

            with capture_stdout():
                import pandas as pd

                X = ds.df.copy()
                # Drop target if present
                if ds.target_column and ds.target_column in X.columns:
                    X = X.drop(columns=[ds.target_column])

                from endgame.mcp.tools._encoding import apply_encoders
                X, _ = apply_encoders(X, None, model_art)

                preds = model_art.estimator.predict(X)

                result_data = {"predictions_shape": list(preds.shape)}
                out_df = pd.DataFrame({"prediction": preds})

                if include_probabilities and hasattr(model_art.estimator, "predict_proba"):
                    try:
                        proba = model_art.estimator.predict_proba(X)
                        for i in range(proba.shape[1]):
                            out_df[f"probability_class_{i}"] = proba[:, i]
                        result_data["includes_probabilities"] = True
                    except Exception:
                        result_data["includes_probabilities"] = False

                if output_path:
                    out_df.to_csv(output_path, index=False)
                    result_data["saved_to"] = output_path
                else:
                    # Return first few predictions
                    result_data["head"] = out_df.head(10).to_dict(orient="records")

                result_data["n_predictions"] = len(preds)
                return ok_response(result_data)

        except KeyError as e:
            return error_response("not_found", str(e))
        except Exception as e:
            return error_response("internal", str(e))
