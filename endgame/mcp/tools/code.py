"""Code tool: run_python, Python in the session with every Endgame module (disable: ENDGAME_MCP_ALLOW_CODE=0)."""

from __future__ import annotations

import ast
import contextlib
import io
import os
import traceback
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from endgame.mcp.server import error_response, ok_response
from endgame.mcp.session import SessionManager
from endgame.mcp.tools._timeout import DEFAULT_TIMEOUT, MCPTimeoutError, timeout_guard

_MAX_OUTPUT = 20000


@contextlib.contextmanager
def _output_to(buffer: io.StringIO):
    """Python output to ``buffer``; C-level writes to fd 1 (the JSON-RPC stream) go to stderr instead."""
    saved = os.dup(1)
    os.dup2(2, 1)
    try:
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            yield
    finally:
        os.dup2(saved, 1)
        os.close(saved)


def _namespace(session: SessionManager) -> dict[str, Any]:
    ns = getattr(session, "code_namespace", None)
    if ns is not None:
        return ns

    import numpy as np
    import pandas as pd
    import polars as pl

    import endgame as eg

    def dataset(dataset_id: str) -> pd.DataFrame:
        return session.get_dataset(dataset_id).df.copy()

    def add_dataset(df, name: str, target: str | None = None, task_type: str | None = None) -> str:
        if isinstance(df, pl.DataFrame):
            df = df.to_pandas()
        if target and task_type is None:
            from endgame.automl.utils.data_loader import infer_task_type
            task_type = infer_task_type(df[target])
        return session.add_dataset(df=df.reset_index(drop=True), name=name, source="run_python",
                                   target_column=target, task_type=task_type).id

    def model(model_id: str):
        return session.get_model(model_id).estimator

    def add_model(estimator, dataset_id: str, name: str, oof_predictions=None, metrics: dict | None = None,
                  task_type: str | None = None) -> str:
        ds = session.get_dataset(dataset_id)
        features = list(getattr(estimator, "feature_names_in_", None)
                        or [c for c in ds.df.columns if c != ds.target_column])
        return session.add_model(estimator=estimator, name=name, model_type=type(estimator).__name__,
                                 dataset_id=dataset_id, task_type=task_type or ds.task_type or "classification",
                                 metrics=metrics or {}, feature_names=features,
                                 oof_predictions=None if oof_predictions is None else np.asarray(oof_predictions)).id

    ns = {"eg": eg, "np": np, "pd": pd, "pl": pl, "session": session, "dataset": dataset,
          "add_dataset": add_dataset, "model": model, "add_model": add_model, "__name__": "__endgame_session__"}
    session.code_namespace = ns
    return ns


def _clip(text: str) -> str:
    return text if len(text) <= _MAX_OUTPUT else text[:_MAX_OUTPUT // 2] + "\n...[truncated]...\n" + text[-_MAX_OUTPUT // 2:]


def register(mcp: FastMCP, session: SessionManager) -> None:
    if os.environ.get("ENDGAME_MCP_ALLOW_CODE", "1").strip().lower() in ("0", "false", "no", "off"):
        return

    @mcp.tool(annotations=ToolAnnotations(title="Run Python with Endgame", readOnlyHint=False,
                                          destructiveHint=True, openWorldHint=True))
    def run_python(code: str, timeout_seconds: int = DEFAULT_TIMEOUT) -> str:
        """Run Python in this session to use any Endgame module the dedicated tools don't cover (survival,
        calibration, fairness, NLP, vision, custom CV, tuning, ...). Preloaded: eg (endgame), np, pd, pl and
        dataset(id) -> DataFrame, add_dataset(df, name, target=None) -> id, model(id) -> estimator,
        add_model(fitted_estimator, dataset_id, name, oof_predictions=None, metrics=None) -> id.
        Variables persist between calls. Returns printed output and the value of a final expression.
        It runs with this machine's permissions. guide("code") has details."""
        ns = _namespace(session)
        before = (set(session.datasets), set(session.models))
        buffer = io.StringIO()
        try:
            tree = ast.parse(code, mode="exec")
            last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
            value = None
            with _output_to(buffer), timeout_guard(timeout_seconds):
                exec(compile(tree, "<run_python>", "exec"), ns)
                if last is not None:
                    value = eval(compile(ast.Expression(last.value), "<run_python>", "eval"), ns)
        except MCPTimeoutError as e:
            return error_response("timeout", str(e), hint=f"Output so far:\n{_clip(buffer.getvalue())}")
        except Exception:
            tb = traceback.format_exc().split('File "<run_python>"', 1)
            message = 'File "<run_python>"' + tb[1] if len(tb) == 2 else tb[0]
            return error_response("execution_error", _clip(message),
                                  hint=f"Output before the error:\n{_clip(buffer.getvalue())}" if buffer.getvalue() else "")

        result: dict[str, Any] = {"stdout": _clip(buffer.getvalue())}
        if value is not None:
            result["result"] = _clip(repr(value))
        new_ds = [d for d in session.datasets if d not in before[0]]
        new_models = [m for m in session.models if m not in before[1]]
        if new_ds:
            result["new_datasets"] = {d: list(session.datasets[d].df.shape) for d in new_ds}
        if new_models:
            result["new_models"] = new_models
        return ok_response(result)
