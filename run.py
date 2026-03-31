"""Entry point for the PMA-RAG API server.

Passes a SelectorEventLoop factory to uvicorn so psycopg async works on Windows.

Usage:
    python run_api.py
    python run_api.py --port 8000 --host 0.0.0.0
"""

import asyncio
import selectors
import sys

import uvicorn


def _selector_event_loop() -> asyncio.AbstractEventLoop:
    selector = selectors.SelectSelector()
    return asyncio.SelectorEventLoop(selector)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the PMA-RAG API server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    loop_factory = _selector_event_loop if sys.platform == "win32" else "auto"

    uvicorn.run(
        "src.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        loop=loop_factory,
    )
