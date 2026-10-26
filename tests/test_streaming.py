import asyncio

import pytest

from langchain_tool_utils import StreamIdleTimeout, watch_stream


async def test_normal_stream_and_cleanup():
    closed = []

    async def events():
        try:
            yield 1
            yield 2
        finally:
            closed.append(True)

    assert [value async for value in watch_stream(events(), idle_timeout=1)] == [1, 2]
    assert closed == [True]


async def test_total_deadline_stops_endless_heartbeats():
    async def events():
        while True:
            yield "heartbeat"
            await asyncio.sleep(0.001)

    with pytest.raises(StreamIdleTimeout):
        async for _ in watch_stream(events(), idle_timeout=1, total_timeout=0.02):
            pass


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
async def test_invalid_timeout(timeout):
    async def events():
        yield 1

    with pytest.raises(ValueError):
        async for _ in watch_stream(events(), idle_timeout=timeout):
            pass
