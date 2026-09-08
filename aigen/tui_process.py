from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
import os
from pathlib import Path
import signal


TERMINATION_GRACE_SECONDS = 5.0


async def command_lines(stream: asyncio.StreamReader) -> AsyncIterator[str]:
    """Frame stdout in linear time, including records larger than a pipe buffer."""
    pending = bytearray()
    while chunk := await stream.read(64 * 1024):
        search_from = len(pending)
        pending.extend(chunk)
        start = 0
        while (end := pending.find(b"\n", search_from)) != -1:
            yield pending[start:end].decode("utf-8", errors="replace")
            start = search_from = end + 1
        if start:
            del pending[:start]
    if pending:
        yield pending.decode("utf-8", errors="replace")


@asynccontextmanager
async def command_process(command: Sequence[str], *, cwd: Path,
                          env: Mapping[str, str]) -> AsyncIterator[asyncio.subprocess.Process]:
    """Own a CLI process group from spawn through pipe draining and final reap."""
    spawn = asyncio.create_task(asyncio.create_subprocess_exec(
        *command, cwd=cwd, env=env, stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        start_new_session=True,
    ))
    process = None
    try:
        process = await asyncio.shield(spawn)
        yield process
    finally:
        # Cancellation can arrive before subprocess creation has returned its handle.
        if process is None and not spawn.cancelled():
            process = await spawn
        if process is not None:
            cleanup = asyncio.create_task(_terminate_process(process))
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                await cleanup
                raise


async def _terminate_process(process: asyncio.subprocess.Process) -> None:
    async def drain_and_wait() -> None:
        assert process.stdout is not None
        while await process.stdout.read(64 * 1024):
            pass
        await process.wait()

    _signal_group(process.pid, signal.SIGTERM)
    drain = asyncio.create_task(drain_and_wait())
    try:
        await asyncio.wait_for(asyncio.shield(drain), TERMINATION_GRACE_SECONDS)
    except TimeoutError:
        pass
    finally:
        # Reap descendants as well, including a worker whose parent exited first.
        _signal_group(process.pid, signal.SIGKILL)
        await drain


def _signal_group(pid: int, signum: signal.Signals) -> None:
    try:
        os.killpg(pid, signum)
    except ProcessLookupError:
        # The process group may have exited between reading and signalling it.
        pass
