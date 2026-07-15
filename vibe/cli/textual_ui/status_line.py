"""External status line support: payload builder and command runner.

The status line feature runs a user-configured shell command
(``status_line_command`` in ``config.toml``) and renders the first line
of its stdout as a one-line row below the input bar. The command
receives a JSON payload on stdin describing the current session; the
key shape intentionally matches Claude Code's ``statusLine`` stdin
contract so status line scripts written for one runtime port to the
other unchanged.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from vibe.core.logger import logger
from vibe.core.utils import kill_async_subprocess
from vibe.core.utils.io import decode_safe

DEFAULT_COMMAND_TIMEOUT = 2.0


def build_status_line_payload(
    *,
    session_id: str,
    transcript_path: str,
    cwd: str,
    model_id: str | None,
    context_size: int | None,
    input_tokens: int,
) -> dict[str, Any]:
    """Build the stdin JSON payload for the status line command.

    Pure function; all session state is passed in. The returned key
    shape is a cross-runtime wire contract (Claude Code's statusLine
    payload) — do not rename keys.
    """
    size = context_size if context_size is not None and context_size > 0 else 0
    used_percentage = (input_tokens / size) * 100 if size else 0.0
    return {
        "session_id": session_id,
        "transcript_path": transcript_path,
        "cwd": cwd,
        "model": {"id": model_id or ""},
        "context_window": {
            "size": size,
            "used_percentage": used_percentage,
            "current_usage": {"input_tokens": input_tokens},
        },
    }


class StatusLineRunner:
    """Runs the status line command and keeps the last good output line.

    The command is executed through the shell (like hook commands) with
    the JSON payload on stdin. On timeout, non-zero exit, or empty
    stdout the previous good line is kept — errors are logged at debug
    level and never rendered into the UI. Overlapping invocations are
    skipped: a refresh requested while one is in flight returns the
    current last good line without spawning a second process.
    """

    def __init__(self, command: str, timeout: float = DEFAULT_COMMAND_TIMEOUT) -> None:
        self._command = command
        self._timeout = timeout
        self._last_line: str | None = None
        self._in_flight = False

    @property
    def last_line(self) -> str | None:
        return self._last_line

    async def refresh(self, payload: dict[str, Any]) -> str | None:
        """Run the command once and return the freshest good line."""
        if self._in_flight:
            return self._last_line
        self._in_flight = True
        try:
            line = await self._run_once(payload)
        finally:
            self._in_flight = False
        if line is not None:
            self._last_line = line
        return self._last_line

    async def _run_once(self, payload: dict[str, Any]) -> str | None:
        stdin_data = json.dumps(payload).encode()
        try:
            process = await asyncio.create_subprocess_shell(
                self._command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as e:
            logger.debug("Status line command failed to start: %s", e)
            return None
        try:
            stdout_bytes, _ = await asyncio.wait_for(
                process.communicate(stdin_data), timeout=self._timeout
            )
        except TimeoutError:
            await kill_async_subprocess(process)
            logger.debug("Status line command timed out after %.1fs", self._timeout)
            return None
        except BaseException:
            if process.returncode is None:
                await kill_async_subprocess(process)
            raise
        if process.returncode != 0:
            logger.debug("Status line command exited with code %s", process.returncode)
            return None
        text = decode_safe(stdout_bytes, from_subprocess=True).text
        first_line = next(
            (line.strip() for line in text.splitlines() if line.strip()), ""
        )
        return first_line or None
