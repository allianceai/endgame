"""Entry point: python -m endgame.mcp"""

import sys


def main():
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
