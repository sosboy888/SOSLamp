"""Runs an asyncio event loop on a background thread for GUI apps.

Qt has its own event loop and bleak needs asyncio, and the two can't share one
thread. This module gives the GUI a simple way to hand coroutines to a
separate asyncio thread without blocking the window.
"""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import Future
from typing import Awaitable, Callable, TypeVar

T = TypeVar("T")


class AsyncExecutor:
    """Owns an asyncio loop that lives on a daemon thread."""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        # daemon=True so a forgotten stop() can't keep the app alive after the window closes.
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        # Runs on the background thread: bind the loop to this thread and spin forever.
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def submit(self, coro: Awaitable[T]) -> Future:
        """Schedule a coroutine from any thread; returns a concurrent Future for the result."""
        # run_coroutine_threadsafe is the thread-safe way to push work into another thread's loop.
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def call_soon(self, func: Callable[..., T], *args, **kwargs) -> None:
        """Run a plain (non-async) function on the loop thread."""
        self._loop.call_soon_threadsafe(func, *args, **kwargs)

    def stop(self) -> None:
        """Stop the loop and wait briefly for the thread to finish."""
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=1)
