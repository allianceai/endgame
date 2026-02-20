"""Endgame MCP Server — LLM-powered ML pipeline building.

Exposes ~20 meta-tools over the Model Context Protocol so that any
MCP-compatible LLM host (Claude Code, Claude Desktop, VS Code, etc.)
can build ML pipelines through natural language.

Usage
-----
    python -m endgame.mcp          # stdio transport (default)
    python -m endgame.mcp --sse    # SSE transport
"""

__all__ = ["create_server"]


def create_server():
    """Create and return the configured MCP server."""
    from endgame.mcp.server import create_server as _create
    return _create()
