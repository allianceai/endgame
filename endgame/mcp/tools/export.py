"""Export tools: export_script, save_model."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from endgame.mcp.server import capture_stdout, error_response, ok_response
from endgame.mcp.session import SessionManager


def register(mcp: FastMCP, session: SessionManager) -> None:

    @mcp.tool()
    def export_script(
        model_id: str,
        dataset_path: str | None = None,
        include_preprocessing: bool = True,
    ) -> str:
        """Generate a standalone Python script that reproduces the trained model pipeline."""
        try:
            model_art = session.get_model(model_id)

            with capture_stdout():
                from endgame.mcp.codegen import generate_script

                ds = session.get_dataset(model_art.dataset_id)
                script = generate_script(
                    model_art=model_art,
                    dataset_art=ds,
                    dataset_path=dataset_path,
                    include_preprocessing=include_preprocessing,
                )

                # Save to working dir
                out_path = str(session.working_dir / f"pipeline_{model_art.name}.py")
                with open(out_path, "w") as f:
                    f.write(script)

                return ok_response({
                    "script_path": out_path,
                    "script": script,
                })

        except KeyError as e:
            return error_response("not_found", str(e))
        except Exception as e:
            return error_response("internal", str(e))

    @mcp.tool()
    def save_model(
        model_id: str,
        path: str | None = None,
    ) -> str:
        """Save a trained model to disk using endgame persistence."""
        try:
            model_art = session.get_model(model_id)

            with capture_stdout():
                if path is None:
                    path = str(session.working_dir / f"model_{model_art.name}")

                from endgame.persistence import save

                saved_path = save(model_art.estimator, path)

                return ok_response({
                    "model_id": model_id,
                    "saved_path": saved_path,
                    "model_name": model_art.name,
                })

        except KeyError as e:
            return error_response("not_found", str(e))
        except Exception as e:
            return error_response("internal", str(e))
