"""Timeout protection for long-running MCP operations.

Uses ``signal.alarm`` on Unix (reliable, thread-safe for the main thread)
with a ``threading.Timer`` fallback for non-main threads or Windows.
"""

from __future__ import annotations

import contextlib
import os
import signal
import threading
from functools import wraps
from typing import Callable

DEFAULT_TIMEOUT = int(os.environ.get("ENDGAME_MCP_TIMEOUT", "600"))  # 10 min


class MCPTimeoutError(Exception):
    pass


@contextlib.contextmanager
def timeout_guard(seconds: int = DEFAULT_TIMEOUT):
    """Context manager that raises ``MCPTimeoutError`` after *seconds*.

    Uses ``signal.SIGALRM`` on Unix main thread, otherwise a no-op
    (non-main threads cannot use signals reliably).
    """
    use_signal = (
        hasattr(signal, "SIGALRM")
        and threading.current_thread() is threading.main_thread()
    )

    if use_signal:
        def _handler(signum, frame):
            raise MCPTimeoutError(f"Operation timed out after {seconds}s")

        old_handler = signal.signal(signal.SIGALRM, _handler)
        old_alarm = signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
            if old_alarm > 0:
                signal.alarm(old_alarm)
    else:
        yield


def with_timeout(seconds: int = DEFAULT_TIMEOUT) -> Callable:
    """Decorator that aborts a function after *seconds* and returns an error response."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                with timeout_guard(seconds):
                    return func(*args, **kwargs)
            except MCPTimeoutError:
                from endgame.mcp.server import error_response
                return error_response(
                    "timeout",
                    f"Operation timed out after {seconds}s",
                    hint="Try a simpler model or smaller dataset.",
                )

        return wrapper
    return decorator
