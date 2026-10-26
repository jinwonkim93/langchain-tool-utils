"""Consumer watchdog for async streams, separate from HTTP SSE read timeout."""

from __future__ import annotations

import asyncio
import math
from contextlib import suppress


class StreamIdleTimeout(TimeoutError):
    pass


async def watch_stream(
    source, *, idle_timeout: float, total_timeout: float | None = None
):
    """Close/cancel a stalled async stream; never synthesize a finished message.

    Events are provisional until the model node commits. A timeout does not prove
    a tool stopped. Check operation outcomes before resuming a checkpoint that can rerun tools.
    The deadline includes consumer backpressure. Sources must cooperate with
    cancellation; Python cannot forcibly stop an arbitrary blocked worker.
    """
    if not math.isfinite(idle_timeout) or idle_timeout <= 0:
        raise ValueError("idle_timeout must be finite and positive")
    if total_timeout is not None and (
        not math.isfinite(total_timeout) or total_timeout <= 0
    ):
        raise ValueError("total_timeout must be finite and positive")
    iterator = aiter(source)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + total_timeout if total_timeout is not None else math.inf
    try:
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise StreamIdleTimeout("Agent stream exceeded its total deadline")
            try:
                event = await asyncio.wait_for(
                    anext(iterator), min(idle_timeout, remaining)
                )
            except StopAsyncIteration:
                return
            except TimeoutError as exc:
                raise StreamIdleTimeout(
                    "Agent stream stalled or exceeded its deadline"
                ) from exc
            yield event
    finally:
        close = getattr(iterator, "aclose", None)
        if close:
            with suppress(Exception):
                await close()
