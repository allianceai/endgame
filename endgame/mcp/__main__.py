from __future__ import annotations

"""Entry point: python -m endgame.mcp"""

import sys


def main():
    import os

    # stdin/stdout carry the MCP protocol: tabpfn's licence flow would open a browser and poll stdin for an API key.
    # Raise its licence error (which says how to accept) instead.
    os.environ.setdefault("TABPFN_NO_BROWSER", "1")
    transport = "stdio"
    if "--sse" in sys.argv:
        transport = "sse"

    from endgame.mcp.server import create_server
    server = create_server()

    if transport == "sse":
        server.run(transport="sse")
    else:
        server.run(transport="stdio")


if __name__ == "__main__":
    main()
