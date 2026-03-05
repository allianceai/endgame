"""FastMCP server — tool and resource registration.

This is the central file that creates the MCP server instance, registers
all tools from the ``tools/`` sub-package, wires up MCP resources, and
exposes the ``capture_stdout`` helper used by every tool function.
"""

from __future__ import annotations

import contextlib
import json
import sys
from typing import Any

from mcp.server.fastmcp import FastMCP

from endgame.mcp.session import SessionManager

# ---------------------------------------------------------------------------
# Stdout protection — CRITICAL for stdio transport
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def capture_stdout():
    """Redirect stdout → stderr while calling Endgame code.

    Many Endgame modules print progress to stdout; if the MCP server
    uses stdio transport, corrupted stdout breaks the JSON-RPC protocol.
    """
    old = sys.stdout
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        sys.stdout = old


# ---------------------------------------------------------------------------
# Structured error responses
# ---------------------------------------------------------------------------

def error_response(
    error_type: str,
    message: str,
    hint: str = "",
) -> str:
    payload: dict[str, Any] = {
        "status": "error",
        "error_type": error_type,
        "message": message,
    }
    if hint:
        payload["hint"] = hint
    return json.dumps(payload, default=str)


def ok_response(data: Any) -> str:
    payload = {"status": "ok", **data} if isinstance(data, dict) else {"status": "ok", "result": data}
    return json.dumps(payload, default=str)


# ---------------------------------------------------------------------------
# Server factory
# ---------------------------------------------------------------------------

def create_server() -> FastMCP:
    """Build and return a fully-configured ``FastMCP`` server."""

    mcp = FastMCP(
        "endgame",
        instructions=(
            "Endgame ML toolkit — build ML pipelines through natural language. "
            "Use discovery resources (endgame://catalog/*) for zero-cost browsing "
            "of models, presets, metrics, and visualizers. Use tools to load data, "
            "train models, evaluate, visualize, and export pipelines."
        ),
    )

    # Shared session (one per server process)
    session = SessionManager()

    # ----- register tools ------------------------------------------------
    from endgame.mcp.tools.advanced import register as reg_advanced
    from endgame.mcp.tools.data import register as reg_data
    from endgame.mcp.tools.discover import register as reg_discover
    from endgame.mcp.tools.evaluate import register as reg_evaluate
    from endgame.mcp.tools.export import register as reg_export
    from endgame.mcp.tools.guardrails import register as reg_guardrails
    from endgame.mcp.tools.predict import register as reg_predict
    from endgame.mcp.tools.preprocess import register as reg_preprocess
    from endgame.mcp.tools.train import register as reg_train
    from endgame.mcp.tools.visualize import register as reg_visualize

    for reg_fn in (
        reg_data,
        reg_discover,
        reg_train,
        reg_evaluate,
        reg_predict,
        reg_preprocess,
        reg_visualize,
        reg_export,
        reg_advanced,
        reg_guardrails,
    ):
        reg_fn(mcp, session)

    # ----- register resources --------------------------------------------
    from endgame.mcp.resources import register as reg_resources
    reg_resources(mcp, session)

    return mcp
