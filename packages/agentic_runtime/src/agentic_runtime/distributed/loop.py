"""One asyncio loop on a background thread, so a synchronous caller can await.

``laser_sdk`` is async the whole way down; the distributed runtime around it is
not. ``DistributedService`` runs a blocking consume loop with a heartbeat
thread beside it, and ``DistributedChatClient.ask`` blocks until the answer
comes back. Making that seam async would change every caller of a transport
that is otherwise unchanged, so the transport keeps the synchronous protocol
Redis Streams gave it and owns a loop instead.

**Work is submitted as a callable, never as a coroutine.** That is the one
thing about this module worth reading. `laser_sdk` is a pyo3 extension, and its
awaitables are built against the *calling* thread's running loop — so
`loop.run(client.connect(...))` fails with "no running event loop" before it
ever reaches the loop it was meant for. Taking a zero-argument callable moves
construction to the loop thread as well, which is where it has to happen.

Every Laser object is therefore created and awaited on this one loop. That is
also what lets the heartbeat thread publish while the consume thread sits in a
poll: both hand a callable to the same loop, which interleaves them.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from concurrent.futures import Future
import inspect
import threading

type Work[T] = Callable[[], Awaitable[T] | T]

# One turn of the loop, so a cancellation lands before it stops.
_CANCELLATION_TICK_SECONDS = 0.05


class LoopClosedError(RuntimeError):
    """Raised when work is submitted to a loop that has already stopped."""


class BackgroundLoop:
    """An event loop running on a daemon thread, driven from synchronous code."""

    def __init__(self, *, name: str = "laser-loop") -> None:
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._closed = False
        self._thread = threading.Thread(target=self._serve, name=name, daemon=True)
        self._thread.start()
        self._ready.wait()

    @property
    def closed(self) -> bool:
        return self._closed

    def run[T](self, work: Work[T], *, timeout: float | None = None) -> T:
        """Call *work* on the loop thread and block until its result is in."""
        return self.submit(work).result(timeout)

    def submit[T](self, work: Work[T]) -> Future[T]:
        """Schedule *work* without waiting. Cancelling the future cancels the task."""
        if self._closed:
            raise LoopClosedError("The transport loop is closed.")
        return asyncio.run_coroutine_threadsafe(_invoke(work), self._loop)

    def close(self, *, timeout: float = 5.0) -> None:
        """Stop the loop and join its thread. Idempotent."""
        if self._closed:
            return
        self._closed = True
        self._loop.call_soon_threadsafe(self._stop)
        self._thread.join(timeout=timeout)
        if not self._thread.is_alive():
            self._loop.close()

    def _stop(self) -> None:
        """Cancel what is still in flight, then stop. Runs on the loop thread.

        Stopping outright leaves anything pending — a connect to a port that
        turned out not to be a broker, say — to be garbage-collected as a task
        destroyed while pending, which is a warning printed at nobody.
        """
        for task in asyncio.all_tasks(self._loop):
            task.cancel()
        self._loop.call_later(_CANCELLATION_TICK_SECONDS, self._loop.stop)

    def _serve(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.call_soon(self._ready.set)
        self._loop.run_forever()


async def _invoke[T](work: Work[T]) -> T:
    result = work()
    if inspect.isawaitable(result):
        return await result
    return result
