from __future__ import annotations

from collections.abc import Callable
import time
from typing import Any

from rich.text import Text
from textual.timer import Timer

from vibe.cli.textual_ui.status_line import StatusLineRunner
from vibe.cli.textual_ui.widgets.no_markup_static import NoMarkupStatic

DEBOUNCE_SECONDS = 1.0


class StatusLine(NoMarkupStatic):
    """One-line row rendering the configured status line command's output.

    Refresh triggers:
    - widget mount
    - the periodic interval timer (``status_line_interval`` > 0)
    - :meth:`trigger_refresh` (e.g. end of an agent turn)
    - :meth:`trigger_refresh_debounced` (e.g. context token updates),
      coalesced to at most one run per ``DEBOUNCE_SECONDS``.

    Output is rendered via ``rich.text.Text.from_ansi`` so ANSI colours
    in the command's stdout survive.
    """

    def __init__(
        self,
        *,
        command: str,
        interval: float = 5.0,
        payload_provider: Callable[[], dict[str, Any]],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._runner = StatusLineRunner(command)
        self._interval = interval
        self._payload_provider = payload_provider
        self._debounce_timer: Timer | None = None
        self._last_trigger_at: float | None = None

    def on_mount(self) -> None:
        if self._interval > 0:
            self.set_interval(self._interval, self.trigger_refresh)
        # Defer the first run: is_mounted is still False inside on_mount (the
        # trigger_refresh guard would drop it), so kick it once the widget is
        # fully mounted on the next refresh cycle.
        self.call_after_refresh(self.trigger_refresh)

    def trigger_refresh(self) -> None:
        """Run the command now (skipped internally if one is in flight)."""
        if not self.is_mounted:
            return
        self._last_trigger_at = time.monotonic()
        self.run_worker(self._do_refresh(), exclusive=False)

    def trigger_refresh_debounced(self) -> None:
        """Run the command, coalescing rapid calls into one trailing run."""
        if not self.is_mounted:
            return
        now = time.monotonic()
        if (
            self._last_trigger_at is None
            or now - self._last_trigger_at >= DEBOUNCE_SECONDS
        ):
            self.trigger_refresh()
            return
        if self._debounce_timer is None:
            remaining = DEBOUNCE_SECONDS - (now - self._last_trigger_at)
            self._debounce_timer = self.set_timer(remaining, self._fire_debounced)

    def _fire_debounced(self) -> None:
        self._debounce_timer = None
        self.trigger_refresh()

    async def _do_refresh(self) -> None:
        line = await self._runner.refresh(self._payload_provider())
        if line is not None:
            self.update(Text.from_ansi(line))
