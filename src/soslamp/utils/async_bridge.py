"""Runs an asyncio event loop on a background thread for GUI apps."""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import Future
from typing import Awaitable, Callable, TypeVar

T = TypeVar("T")


class AsyncExecutor:
    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def submit(self, coro: Awaitable[T]) -> Future:
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def call_soon(self, func: Callable[..., T], *args, **kwargs) -> None:
        self._loop.call_soon_threadsafe(func, *args, **kwargs)

    def stop(self) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=1)
