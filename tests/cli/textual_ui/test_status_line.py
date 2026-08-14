from __future__ import annotations

import asyncio
from collections.abc import Callable
import json
from pathlib import Path
import time

import pytest

from tests.conftest import ConfigBuilder, build_test_vibe_app, build_test_vibe_config
from vibe.cli.textual_ui.status_line import StatusLineRunner, build_status_line_payload
from vibe.cli.textual_ui.widgets.status_line import StatusLine

# ---------------------------------------------------------------------------
# Payload builder
# ---------------------------------------------------------------------------


def test_payload_matches_wire_contract_exactly() -> None:
    # Cross-runtime wire contract (Claude Code statusLine stdin shape).
    # Assert the full structure literally — key renames are breaking.
    payload = build_status_line_payload(
        session_id="sess-1",
        transcript_path="/tmp/sessions/abc/messages.jsonl",
        cwd="/work/project",
        model_id="devstral-medium-2507",
        context_size=200_000,
        input_tokens=85_000,
    )

    assert payload == {
        "session_id": "sess-1",
        "transcript_path": "/tmp/sessions/abc/messages.jsonl",
        "cwd": "/work/project",
        "model": {"id": "devstral-medium-2507"},
        "context_window": {
            "size": 200000,
            "used_percentage": 42.5,
            "current_usage": {"input_tokens": 85000},
        },
    }


def test_payload_percentage_is_derived_from_tokens_and_size() -> None:
    payload = build_status_line_payload(
        session_id="s",
        transcript_path="",
        cwd="/",
        model_id="m",
        context_size=100_000,
        input_tokens=25_000,
    )

    assert payload["context_window"]["used_percentage"] == 25.0


def test_payload_is_graceful_without_model_or_threshold() -> None:
    payload = build_status_line_payload(
        session_id="s",
        transcript_path="",
        cwd="/",
        model_id=None,
        context_size=None,
        input_tokens=10,
    )

    assert payload["model"] == {"id": ""}
    assert payload["context_window"]["size"] == 0
    assert payload["context_window"]["used_percentage"] == 0.0
    assert payload["context_window"]["current_usage"] == {"input_tokens": 10}
    json.dumps(payload)  # remains JSON-serializable


def test_payload_treats_zero_size_as_unknown() -> None:
    payload = build_status_line_payload(
        session_id="s",
        transcript_path="",
        cwd="/",
        model_id="m",
        context_size=0,
        input_tokens=10,
    )

    assert payload["context_window"]["size"] == 0
    assert payload["context_window"]["used_percentage"] == 0.0


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_runner_returns_first_stdout_line() -> None:
    runner = StatusLineRunner("printf 'hello\\nworld'")

    assert await runner.refresh({}) == "hello"


@pytest.mark.asyncio
async def test_runner_passes_payload_on_stdin() -> None:
    runner = StatusLineRunner("cat")
    payload = {"session_id": "abc", "cwd": "/tmp"}

    line = await runner.refresh(payload)

    assert line is not None
    assert json.loads(line) == payload


@pytest.mark.asyncio
async def test_runner_timeout_returns_none() -> None:
    runner = StatusLineRunner("sleep 5", timeout=0.2)

    assert await runner.refresh({}) is None


@pytest.mark.asyncio
async def test_runner_nonzero_exit_returns_none() -> None:
    runner = StatusLineRunner("printf 'nope'; exit 3")

    assert await runner.refresh({}) is None


@pytest.mark.asyncio
async def test_runner_empty_stdout_returns_none() -> None:
    runner = StatusLineRunner("true")

    assert await runner.refresh({}) is None


@pytest.mark.asyncio
async def test_runner_keeps_last_good_line_on_failure(tmp_path: Path) -> None:
    flag = tmp_path / "ok"
    runner = StatusLineRunner(f"if [ -e {flag} ]; then printf 'good'; else exit 1; fi")

    flag.touch()
    assert await runner.refresh({}) == "good"

    flag.unlink()
    assert await runner.refresh({}) == "good"  # retained across the failure
    assert runner.last_line == "good"


@pytest.mark.asyncio
async def test_runner_keeps_last_good_line_on_timeout(tmp_path: Path) -> None:
    flag = tmp_path / "fast"
    runner = StatusLineRunner(
        f"if [ -e {flag} ]; then printf 'fast'; else sleep 5; fi", timeout=0.2
    )

    flag.touch()
    assert await runner.refresh({}) == "fast"

    flag.unlink()
    assert await runner.refresh({}) == "fast"  # retained across the timeout


@pytest.mark.asyncio
async def test_runner_skips_overlapping_invocations(tmp_path: Path) -> None:
    counter = tmp_path / "count"
    runner = StatusLineRunner(f"echo run >> {counter}; sleep 0.2; printf 'done'")

    first, second = await asyncio.gather(runner.refresh({}), runner.refresh({}))

    assert counter.read_text().count("run") == 1  # second call never spawned
    assert first == "done"
    assert second is None  # in-flight skip returned the then-empty last line


# ---------------------------------------------------------------------------
# Config defaults (both config classes via the parametrized builder)
# ---------------------------------------------------------------------------


def test_status_line_config_defaults(build_config: ConfigBuilder) -> None:
    config = build_config()

    assert config.status_line_command is None
    assert config.status_line_interval == 5.0


# ---------------------------------------------------------------------------
# Widget / app integration
# ---------------------------------------------------------------------------


async def _wait_until(
    pilot, predicate: Callable[[], bool], timeout: float = 4.0
) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await pilot.pause(0.05)
    return False


@pytest.mark.asyncio
async def test_status_line_not_mounted_when_config_unset() -> None:
    app = build_test_vibe_app()

    async with app.run_test():
        assert list(app.query(StatusLine)) == []


@pytest.mark.asyncio
async def test_status_line_mounted_and_populated_when_configured() -> None:
    config = build_test_vibe_config(status_line_command="printf 'hello status'")
    app = build_test_vibe_app(config=config)

    async with app.run_test() as pilot:
        widget = app.query_one(StatusLine)
        assert await _wait_until(pilot, lambda: str(widget.render()) == "hello status")


@pytest.mark.asyncio
async def test_status_line_renders_ansi_output() -> None:
    config = build_test_vibe_config(
        status_line_command="printf '\\033[32mgreen\\033[0m status'"
    )
    app = build_test_vibe_app(config=config)

    async with app.run_test() as pilot:
        widget = app.query_one(StatusLine)
        # ANSI colours are parsed (not shown raw, no markup errors).
        assert await _wait_until(pilot, lambda: str(widget.render()) == "green status")


@pytest.mark.asyncio
async def test_status_line_debounce_coalesces_rapid_triggers(tmp_path: Path) -> None:
    counter = tmp_path / "count"
    config = build_test_vibe_config(
        status_line_command=f"echo run >> {counter}; printf 'ok'",
        status_line_interval=0,  # timer off: only mount + explicit triggers
    )
    app = build_test_vibe_app(config=config)

    async with app.run_test() as pilot:
        widget = app.query_one(StatusLine)
        assert await _wait_until(
            pilot, lambda: counter.exists() and counter.read_text().count("run") >= 1
        )
        base = counter.read_text().count("run")

        for _ in range(5):
            widget.trigger_refresh_debounced()

        # The burst coalesces into exactly one trailing run.
        assert await _wait_until(
            pilot, lambda: counter.read_text().count("run") == base + 1
        )
        await pilot.pause(0.3)
        assert counter.read_text().count("run") == base + 1


@pytest.mark.asyncio
async def test_status_line_interval_timer_refreshes(tmp_path: Path) -> None:
    counter = tmp_path / "count"
    config = build_test_vibe_config(
        status_line_command=f"echo run >> {counter}; printf 'ok'",
        status_line_interval=0.05,
    )
    app = build_test_vibe_app(config=config)

    async with app.run_test() as pilot:
        assert await _wait_until(
            pilot, lambda: counter.exists() and counter.read_text().count("run") >= 3
        )


@pytest.mark.asyncio
async def test_status_line_payload_provider_reflects_session() -> None:
    config = build_test_vibe_config(status_line_command="cat")
    app = build_test_vibe_app(config=config)

    async with app.run_test():
        payload = app._build_status_line_payload()

        runtime = app.app_server.resources.runtime
        assert payload["session_id"] == app.app_server.session_id
        assert payload["cwd"] == str(Path.cwd().resolve())
        assert payload["model"]["id"] == config.get_active_model().name
        assert payload["context_window"]["size"] == runtime.context_window
        assert payload["context_window"]["current_usage"] == {
            "input_tokens": runtime.stats.context_tokens
        }
