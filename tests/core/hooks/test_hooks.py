from __future__ import annotations

import json
from pathlib import Path
import shlex
import sys
from typing import Any

import pytest
import tomli_w

from tests.conftest import (
    build_test_agent_loop,
    build_test_vibe_config,
    make_test_models,
)
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend
from tests.stubs.fake_tool import FakeTool, FakeToolArgs
from vibe.core.agents.models import BuiltinAgentName
from vibe.core.config import SessionLoggingConfig
from vibe.core.hooks._handler import HookOutputError, _parse_structured_response
from vibe.core.hooks.config import (
    HookConfigResult,
    _load_hooks_file,
    load_hooks_from_fs,
)
from vibe.core.hooks.executor import HookExecutor
from vibe.core.hooks.manager import HooksManager
from vibe.core.hooks.models import (
    HookConfig,
    HookContextInjection,
    HookEndEvent,
    HookEvent,
    HookMessageSeverity,
    HookPermissionDecision,
    HookPromptDenial,
    HookRunEndEvent,
    HookRunStartEvent,
    HookSessionContext,
    HookStartEvent,
    HookStructuredResponse,
    HookTextReplacement,
    HookToolDenial,
    HookToolInputRewrite,
    HookType,
    HookUserMessage,
    NotificationInvocation,
    PermissionRequestHookResponse,
    PermissionRequestInvocation,
    PostAgentInvocation,
    PostCompactInvocation,
    PostToolInvocation,
    PreCompactInvocation,
    SessionEndInvocation,
    SessionStartInvocation,
    StopFailureInvocation,
    SubagentStartInvocation,
    SubagentStopInvocation,
    UserPromptSubmitInvocation,
    WorktreeCreateInvocation,
    build_invocation,
)
from vibe.core.llm.exceptions import BackendError, PayloadSummary
from vibe.core.types import (
    ApprovalRequestEvent,
    ApprovalResponse,
    AssistantEvent,
    ContextTooLongError,
    FunctionCall,
    RateLimitError,
    ResponseTooLongError,
    ToolCall,
    ToolResultEvent,
)

_AnyHookYield = (
    HookEvent
    | HookUserMessage
    | HookToolDenial
    | HookToolInputRewrite
    | HookTextReplacement
)


def _run(
    handler: HooksManager, hook_type: HookType, ctx: HookSessionContext, **kwargs: Any
) -> Any:
    """Test convenience: build the right invocation subclass and pipe it
    to ``HooksManager.run``. The manager itself only knows about
    invocations — the per-type kwargs (``tool_name`` / ``tool_input`` /
    ``initial_text`` mapped to ``tool_output_text`` / …) are flattened
    here so tests stay readable.
    """
    initial_text = kwargs.pop("initial_text", "")
    return handler.run(
        build_invocation(hook_type, ctx, tool_output_text=initial_text, **kwargs)
    )


async def _drain_post_tool_chain(
    handler: HooksManager, ctx: HookSessionContext, **kwargs: object
) -> tuple[str, list[_AnyHookYield]]:
    final_text = str(kwargs.get("initial_text", ""))
    events: list[_AnyHookYield] = []
    async for ev in _run(handler, HookType.POST_TOOL, ctx, **kwargs):
        if isinstance(ev, HookTextReplacement):
            final_text = ev.text
        else:
            events.append(ev)
    return final_text, events


@pytest.fixture
def sample_invocation() -> PostAgentInvocation:
    return PostAgentInvocation(
        session_id="test-session", transcript_path="", cwd=str(Path.cwd())
    )


@pytest.fixture
def ctx() -> HookSessionContext:
    return HookSessionContext(
        session_id="sess", transcript_path="", cwd=str(Path.cwd())
    )


def _write_hooks_toml(path: Path, hooks: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        tomli_w.dump({"hooks": hooks}, f)


def _make_hook(
    name: str = "test-hook",
    command: str = "echo ok",
    timeout: float = 60.0,
    type: HookType = HookType.POST_AGENT,
    match: str | None = None,
) -> HookConfig:
    return HookConfig(
        name=name, type=type, command=command, timeout=timeout, match=match
    )


def _make_tool_hook(
    name: str,
    command: str,
    *,
    type: HookType,
    match: str | None = None,
    match_status: str | None = None,
    timeout: float | None = None,
    strict: bool = False,
) -> HookConfig:
    return HookConfig(
        name=name,
        type=type,
        command=command,
        match=match,
        match_status=match_status,  # type: ignore[arg-type]
        timeout=timeout,
        strict=strict,
    )


def _emit_cmd(payload: dict[str, Any]) -> str:
    """Build a shell command that prints ``payload`` as JSON to stdout.

    Uses Python (over ``printf``) so embedded quotes / shell metacharacters
    in field values are not a footgun, and ``shlex.quote`` to give the
    shell a single safe argument to pass to ``-c``.
    """
    body = f"import sys; sys.stdout.write({json.dumps(payload)!r})"
    return f"{sys.executable} -c {shlex.quote(body)}"


def _deny_cmd(reason: str = "") -> str:
    return _emit_cmd({"decision": "deny", "reason": reason})


def _context_cmd(context: str) -> str:
    """Allowing hook that injects ``context`` via the structured response."""
    return _emit_cmd({"hook_specific_output": {"additional_context": context}})


def _capture_cmd(out: Path) -> str:
    """Hook command that writes its stdin (the JSON invocation) to *out*."""
    body = f"import sys; open({str(out)!r}, 'w').write(sys.stdin.read())"
    return f"{sys.executable} -c {shlex.quote(body)}"


def _append_cmd(log: Path, tag: str) -> str:
    """Hook command that appends *tag* as a line to *log* (ordering probe)."""
    body = f"open({str(log)!r}, 'a').write({tag!r} + chr(10))"
    return f"{sys.executable} -c {shlex.quote(body)}"


class TestConfigLoading:
    def test_load_from_global_file(self, config_dir: Path) -> None:
        _write_hooks_toml(
            config_dir / "hooks.toml",
            [{"name": "lint", "type": HookType.POST_AGENT, "command": "echo lint"}],
        )
        result = load_hooks_from_fs()
        assert len(result.hooks) == 1
        assert result.hooks[0].name == "lint"
        assert result.issues == []

    def test_load_from_both_global_and_project(
        self, config_dir: Path, tmp_working_directory: Path
    ) -> None:
        _write_hooks_toml(
            config_dir / "hooks.toml",
            [{"name": "global-hook", "type": "post_agent", "command": "echo global"}],
        )
        project_vibe = tmp_working_directory / ".vibe"
        _write_hooks_toml(
            project_vibe / "hooks.toml",
            [{"name": "project-hook", "type": "post_agent", "command": "echo project"}],
        )
        from vibe.core.trusted_folders import trusted_folders_manager

        trusted_folders_manager.add_trusted(tmp_working_directory)

        result = load_hooks_from_fs()
        assert len(result.hooks) == 2
        names = [h.name for h in result.hooks]
        # Project hooks are loaded first.
        assert names == ["project-hook", "global-hook"]

    def test_project_file_skipped_when_untrusted(
        self, tmp_working_directory: Path
    ) -> None:
        project_vibe = tmp_working_directory / ".vibe"
        _write_hooks_toml(
            project_vibe / "hooks.toml",
            [{"name": "sneaky-hook", "type": "post_agent", "command": "echo sneaky"}],
        )
        result = load_hooks_from_fs()
        assert not any(h.name == "sneaky-hook" for h in result.hooks)

    def test_duplicate_hook_name_project_wins(
        self, config_dir: Path, tmp_working_directory: Path
    ) -> None:
        # Project file loads first; the user-global duplicate is flagged.
        _write_hooks_toml(
            config_dir / "hooks.toml",
            [{"name": "dup-hook", "type": "post_agent", "command": "echo global"}],
        )
        project_vibe = tmp_working_directory / ".vibe"
        _write_hooks_toml(
            project_vibe / "hooks.toml",
            [{"name": "dup-hook", "type": "post_agent", "command": "echo project"}],
        )
        from vibe.core.trusted_folders import trusted_folders_manager

        trusted_folders_manager.add_trusted(tmp_working_directory)

        result = load_hooks_from_fs()
        assert len(result.hooks) == 1
        assert result.hooks[0].command == "echo project"
        assert any("Duplicate" in i.message for i in result.issues)

    def test_toml_parse_error_reported(self, config_dir: Path) -> None:
        hooks_file = config_dir / "hooks.toml"
        hooks_file.write_text("this is not valid toml [[[", encoding="utf-8")
        result = load_hooks_from_fs()
        assert result.hooks == []
        assert len(result.issues) == 1
        assert (
            "parse" in result.issues[0].message.lower()
            or "Failed" in result.issues[0].message
        )

    def test_validation_error_reported(self, config_dir: Path) -> None:
        _write_hooks_toml(
            config_dir / "hooks.toml",
            [{"name": "bad", "type": "InvalidType", "command": "echo"}],
        )
        result = load_hooks_from_fs()
        assert result.hooks == []
        assert len(result.issues) == 1

    def test_missing_command_reported(self, config_dir: Path) -> None:
        _write_hooks_toml(
            config_dir / "hooks.toml", [{"name": "no-cmd", "type": "post_agent"}]
        )
        result = load_hooks_from_fs()
        assert result.hooks == []
        assert len(result.issues) == 1

    def test_empty_command_reported(self, config_dir: Path) -> None:
        _write_hooks_toml(
            config_dir / "hooks.toml",
            [{"name": "empty-cmd", "type": "post_agent", "command": "   "}],
        )
        result = load_hooks_from_fs()
        assert result.hooks == []
        assert len(result.issues) == 1

    def test_default_timeout_is_uniform(self, config_dir: Path) -> None:
        _write_hooks_toml(
            config_dir / "hooks.toml",
            [
                {"name": "p", "type": "post_agent", "command": "echo ok"},
                {"name": "b", "type": "pre_tool", "command": "echo ok"},
                {"name": "a", "type": "post_tool", "command": "echo ok"},
            ],
        )
        result = load_hooks_from_fs()
        assert all(h.timeout == 60.0 for h in result.hooks)

    def test_explicit_timeout_overrides_default(self, config_dir: Path) -> None:
        _write_hooks_toml(
            config_dir / "hooks.toml",
            [{"name": "b", "type": "pre_tool", "command": "echo ok", "timeout": 12.5}],
        )
        result = load_hooks_from_fs()
        assert result.hooks[0].timeout == 12.5

    def test_match_field_on_tool_hooks(self, config_dir: Path) -> None:
        _write_hooks_toml(
            config_dir / "hooks.toml",
            [
                {
                    "name": "b",
                    "type": "pre_tool",
                    "command": "echo ok",
                    "match": "bash",
                },
                {
                    "name": "a",
                    "type": "post_tool",
                    "command": "echo ok",
                    "match": "re:read_.*",
                },
            ],
        )
        result = load_hooks_from_fs()
        assert result.hooks[0].match == "bash"
        assert result.hooks[1].match == "re:read_.*"

    def test_match_field_rejected_on_post_agent(self, config_dir: Path) -> None:
        _write_hooks_toml(
            config_dir / "hooks.toml",
            [
                {
                    "name": "h",
                    "type": "post_agent",
                    "command": "echo ok",
                    "match": "bash",
                }
            ],
        )
        result = load_hooks_from_fs()
        assert result.hooks == []
        assert any("match" in i.message for i in result.issues)

    def test_empty_match_rejected(self, config_dir: Path) -> None:
        _write_hooks_toml(
            config_dir / "hooks.toml",
            [{"name": "h", "type": "pre_tool", "command": "echo ok", "match": "   "}],
        )
        result = load_hooks_from_fs()
        assert result.hooks == []

    def test_nonexistent_file_returns_empty(self, tmp_path: Path) -> None:
        result = _load_hooks_file(tmp_path / "missing.toml")
        assert result.hooks == []
        assert result.issues == []


class TestHookExecutor:
    @pytest.mark.asyncio
    async def test_exit_0_success(self, sample_invocation: PostAgentInvocation) -> None:
        hook = _make_hook(command="echo success")
        result = await HookExecutor().run(hook, sample_invocation)
        assert result.exit_code == 0
        assert result.stdout == "success"
        assert not result.timed_out

    @pytest.mark.asyncio
    async def test_nonzero_exit_passes_through(
        self, sample_invocation: PostAgentInvocation
    ) -> None:
        # The executor is a thin wrapper around the subprocess — it does not
        # interpret exit codes itself. Any non-zero value is forwarded so the
        # manager can decide what to do.
        hook = _make_hook(command="echo 'oops'; exit 1")
        result = await HookExecutor().run(hook, sample_invocation)
        assert result.exit_code == 1
        assert "oops" in result.stdout

    @pytest.mark.asyncio
    async def test_timeout(self, sample_invocation: PostAgentInvocation) -> None:
        hook = _make_hook(command="sleep 60", timeout=0.5)
        result = await HookExecutor().run(hook, sample_invocation)
        assert result.timed_out
        assert result.exit_code is None

    @pytest.mark.asyncio
    async def test_timeout_after_stdio_closed(
        self, sample_invocation: PostAgentInvocation
    ) -> None:
        # A hook that closes stdout/stderr but keeps running must still be
        # killed by the timeout. Before the fix, process.wait() had no
        # timeout so the session would hang indefinitely.
        script = (
            f'{sys.executable} -c "'
            "import sys, time; "
            "sys.stdout.close(); sys.stderr.close(); "
            "time.sleep(60)"
            '"'
        )
        hook = _make_hook(command=script, timeout=0.5)
        result = await HookExecutor().run(hook, sample_invocation)
        assert result.timed_out
        assert result.exit_code is None

    @pytest.mark.asyncio
    async def test_stderr_captured_separately(
        self, sample_invocation: PostAgentInvocation
    ) -> None:
        hook = _make_hook(command="echo out; echo err >&2")
        result = await HookExecutor().run(hook, sample_invocation)
        assert result.exit_code == 0
        assert result.stdout == "out"
        assert result.stderr == "err"

    @pytest.mark.asyncio
    async def test_stdin_json_received(
        self, sample_invocation: PostAgentInvocation
    ) -> None:
        hook = _make_hook(
            command=f"{sys.executable} -c \"import sys,json; d=json.load(sys.stdin); print(d['session_id'])\""
        )
        result = await HookExecutor().run(hook, sample_invocation)
        assert result.exit_code == 0
        assert result.stdout == "test-session"

    @pytest.mark.asyncio
    async def test_large_stdin_when_child_closes_pipe_does_not_crash(self) -> None:
        command = f"{sys.executable} -c \"import sys; sys.stdin.close(); print('ok')\""
        hook = _make_hook(command=command, type=HookType.POST_TOOL)
        invocation = PostToolInvocation(
            session_id="test-session",
            transcript_path="",
            cwd=str(Path.cwd()),
            tool_name="tool",
            tool_call_id="call-1",
            tool_input={},
            tool_status="success",
            tool_output=None,
            tool_output_text="x" * 500_000,
            tool_error=None,
            duration_ms=1.0,
        )
        result = await HookExecutor().run(hook, invocation)
        assert result.exit_code == 0
        assert result.stdout == "ok"
        assert not result.timed_out

    @pytest.mark.asyncio
    async def test_spawn_failure_message_goes_to_stderr(
        self, sample_invocation: PostAgentInvocation, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def _raise(*_args: Any, **_kwargs: Any) -> Any:
            raise OSError("nope")

        monkeypatch.setattr("asyncio.create_subprocess_shell", _raise)
        result = await HookExecutor().run(_make_hook(), sample_invocation)
        assert result.exit_code == 1
        assert result.stdout == ""
        assert "Failed to start" in result.stderr
        assert "nope" in result.stderr


class TestPostAgentHook:
    @pytest.mark.asyncio
    async def test_exit_0_emits_start_and_end(self, ctx: HookSessionContext) -> None:
        handler = HooksManager([_make_hook(command="echo ok")])
        events: list[_AnyHookYield] = []
        async for ev in _run(handler, HookType.POST_AGENT, ctx):
            events.append(ev)

        event_types = [type(e).__name__ for e in events]
        assert "HookRunStartEvent" in event_types
        assert "HookStartEvent" in event_types
        assert "HookEndEvent" in event_types
        assert "HookRunEndEvent" in event_types
        assert not any(isinstance(e, HookUserMessage) for e in events)

    @pytest.mark.asyncio
    async def test_decision_deny_injects_user_message(
        self, ctx: HookSessionContext
    ) -> None:
        handler = HooksManager([_make_hook(command=_deny_cmd("fix it"))])
        events: list[_AnyHookYield] = []
        async for ev in _run(handler, HookType.POST_AGENT, ctx):
            events.append(ev)

        retry_msgs = [e for e in events if isinstance(e, HookUserMessage)]
        assert len(retry_msgs) == 1
        assert retry_msgs[0].content == "fix it"

        end_msgs = [
            e for e in events if isinstance(e, HookEndEvent) and e.content is not None
        ]
        assert any("retrying" in m.content.lower() for m in end_msgs if m.content)
        assert not any("fix it" in (m.content or "") for m in end_msgs)

    @pytest.mark.asyncio
    async def test_decision_deny_with_no_reason_injects_empty(
        self, ctx: HookSessionContext
    ) -> None:
        # decision=deny with reason missing/empty injects an empty user
        # message — the hook explicitly asked for a retry with no guidance.
        handler = HooksManager([_make_hook(command=_deny_cmd())])
        events: list[_AnyHookYield] = []
        async for ev in _run(handler, HookType.POST_AGENT, ctx):
            events.append(ev)

        retry_msgs = [e for e in events if isinstance(e, HookUserMessage)]
        assert len(retry_msgs) == 1
        assert retry_msgs[0].content == ""

    @pytest.mark.asyncio
    async def test_max_retry_limit(self, ctx: HookSessionContext) -> None:
        # Cap matches Claude Code's Stop-hook block cap of 8.
        handler = HooksManager([_make_hook(command=_deny_cmd("retry"))])

        for _ in range(8):
            events = [ev async for ev in _run(handler, HookType.POST_AGENT, ctx)]
            assert any(isinstance(e, HookUserMessage) for e in events)

        events = [ev async for ev in _run(handler, HookType.POST_AGENT, ctx)]
        assert not any(isinstance(e, HookUserMessage) for e in events)
        error_events = [
            e
            for e in events
            if isinstance(e, HookEndEvent)
            and e.content
            and "exhausted" in e.content.lower()
        ]
        assert len(error_events) == 1

    @pytest.mark.asyncio
    async def test_stop_hook_active_false_then_true_on_retry(
        self, ctx: HookSessionContext, tmp_path: Path
    ) -> None:
        # Mirrors Claude Code's Stop contract: the invocation carries
        # stop_hook_active=False on the first fire and True when the run
        # is a retry caused by this hook's previous deny.
        log = tmp_path / "invocations.jsonl"
        capture = (
            f"import sys; open({str(log)!r}, 'a').write(sys.stdin.read().strip()"
            f" + chr(10)); sys.stdout.write"
            f"('{{\"decision\": \"deny\", \"reason\": \"again\"}}')"
        )
        command = f"{sys.executable} -c {shlex.quote(capture)}"
        handler = HooksManager([_make_hook(name="loop-guard", command=command)])

        for _ in range(3):
            _ = [ev async for ev in _run(handler, HookType.POST_AGENT, ctx)]

        payloads = [json.loads(line) for line in log.read_text().splitlines()]
        assert [p["stop_hook_active"] for p in payloads] == [False, True, True]

    @pytest.mark.asyncio
    async def test_stop_hook_active_resets_after_allow(
        self, ctx: HookSessionContext, tmp_path: Path
    ) -> None:
        # deny → retry → allow → the next fire is a fresh stop again.
        log = tmp_path / "invocations.jsonl"
        counter = tmp_path / "count"
        capture = (
            f"import sys; from pathlib import Path; "
            f"log = open({str(log)!r}, 'a'); "
            f"log.write(sys.stdin.read().strip() + chr(10)); "
            f"p = Path({str(counter)!r}); "
            f"c = int(p.read_text()) if p.exists() else 0; "
            f"p.write_text(str(c + 1)); "
            f"sys.stdout.write("
            f"'{{\"decision\": \"deny\", \"reason\": \"again\"}}' if c == 0 else '')"
        )
        command = f"{sys.executable} -c {shlex.quote(capture)}"
        handler = HooksManager([_make_hook(name="loop-guard", command=command)])

        for _ in range(3):
            _ = [ev async for ev in _run(handler, HookType.POST_AGENT, ctx)]

        payloads = [json.loads(line) for line in log.read_text().splitlines()]
        # Fire 1: fresh (denies). Fire 2: retry (allows, clearing the
        # count). Fire 3: fresh again.
        assert [p["stop_hook_active"] for p in payloads] == [False, True, False]

    @pytest.mark.asyncio
    async def test_warning_prefers_stderr_on_nonzero_exit(
        self, ctx: HookSessionContext
    ) -> None:
        # Stderr is the conventional channel for shell diagnostics, and is
        # now preferred over stdout (which is reserved for the JSON response
        # and may be empty / garbage on a crash).
        handler = HooksManager([
            _make_hook(command="echo stdout-msg; echo stderr-msg >&2; exit 1")
        ])
        events = [ev async for ev in _run(handler, HookType.POST_AGENT, ctx)]

        warnings = [
            e
            for e in events
            if isinstance(e, HookEndEvent) and e.status == HookMessageSeverity.WARNING
        ]
        assert len(warnings) == 1
        assert warnings[0].content == "stderr-msg"

    @pytest.mark.asyncio
    async def test_warning_falls_back_to_stdout(self, ctx: HookSessionContext) -> None:
        # When stderr is empty, stdout is still used as a fallback.
        handler = HooksManager([_make_hook(command="echo only-stdout; exit 1")])
        events = [ev async for ev in _run(handler, HookType.POST_AGENT, ctx)]

        warnings = [
            e
            for e in events
            if isinstance(e, HookEndEvent) and e.status == HookMessageSeverity.WARNING
        ]
        assert len(warnings) == 1
        assert warnings[0].content == "only-stdout"

    @pytest.mark.asyncio
    async def test_timeout_emits_warning(self, ctx: HookSessionContext) -> None:
        handler = HooksManager([_make_hook(command="sleep 60", timeout=0.5)])
        events = [ev async for ev in _run(handler, HookType.POST_AGENT, ctx)]

        warnings = [
            e
            for e in events
            if isinstance(e, HookEndEvent) and e.status == HookMessageSeverity.WARNING
        ]
        assert len(warnings) == 1
        assert warnings[0].content and "Timed out" in warnings[0].content


class TestPreToolHook:
    @pytest.mark.asyncio
    async def test_no_hooks_no_events(self, ctx: HookSessionContext) -> None:
        handler = HooksManager([])
        events = [
            ev
            async for ev in _run(
                handler,
                HookType.PRE_TOOL,
                ctx,
                tool_name="bash",
                tool_call_id="tc1",
                tool_input={},
            )
        ]
        assert events == []

    @pytest.mark.asyncio
    async def test_matcher_filters_non_matching(self, ctx: HookSessionContext) -> None:
        handler = HooksManager([
            _make_tool_hook(
                "guard", _deny_cmd("nope"), type=HookType.PRE_TOOL, match="bash"
            )
        ])
        events = [
            ev
            async for ev in _run(
                handler,
                HookType.PRE_TOOL,
                ctx,
                tool_name="read_file",
                tool_call_id="tc1",
                tool_input={},
            )
        ]
        # Non-matching tool: no hooks, no events at all.
        assert events == []

    @pytest.mark.asyncio
    async def test_exit_0_allows_tool(self, ctx: HookSessionContext) -> None:
        handler = HooksManager([
            _make_tool_hook("audit", "echo ok", type=HookType.PRE_TOOL)
        ])
        events = [
            ev
            async for ev in _run(
                handler,
                HookType.PRE_TOOL,
                ctx,
                tool_name="bash",
                tool_call_id="tc1",
                tool_input={"command": "ls"},
            )
        ]
        assert not any(isinstance(e, HookToolDenial) for e in events)
        assert any(isinstance(e, HookRunStartEvent) for e in events)
        assert any(isinstance(e, HookRunEndEvent) for e in events)

    @pytest.mark.asyncio
    async def test_decision_deny_with_reason(self, ctx: HookSessionContext) -> None:
        handler = HooksManager([
            _make_tool_hook("guard", _deny_cmd("no rm -rf"), type=HookType.PRE_TOOL)
        ])
        events = [
            ev
            async for ev in _run(
                handler,
                HookType.PRE_TOOL,
                ctx,
                tool_name="bash",
                tool_call_id="tc1",
                tool_input={"command": "rm -rf /"},
            )
        ]
        denials = [e for e in events if isinstance(e, HookToolDenial)]
        assert len(denials) == 1
        assert denials[0].hook_name == "guard"
        assert denials[0].content == "no rm -rf"

    @pytest.mark.asyncio
    async def test_decision_deny_with_no_reason(self, ctx: HookSessionContext) -> None:
        handler = HooksManager([
            _make_tool_hook("guard", _deny_cmd(), type=HookType.PRE_TOOL)
        ])
        events = [
            ev
            async for ev in _run(
                handler,
                HookType.PRE_TOOL,
                ctx,
                tool_name="bash",
                tool_call_id="tc1",
                tool_input={},
            )
        ]
        denials = [e for e in events if isinstance(e, HookToolDenial)]
        assert len(denials) == 1
        assert denials[0].content == ""

    @pytest.mark.asyncio
    async def test_first_deny_wins(self, ctx: HookSessionContext) -> None:
        # Two hooks both match; the first denies, the second must not run.
        handler = HooksManager([
            _make_tool_hook("first", _deny_cmd("first deny"), type=HookType.PRE_TOOL),
            _make_tool_hook(
                "second", _deny_cmd("second should not run"), type=HookType.PRE_TOOL
            ),
        ])
        events = [
            ev
            async for ev in _run(
                handler,
                HookType.PRE_TOOL,
                ctx,
                tool_name="bash",
                tool_call_id="tc1",
                tool_input={},
            )
        ]
        denials = [e for e in events if isinstance(e, HookToolDenial)]
        assert len(denials) == 1
        assert denials[0].hook_name == "first"
        start_events = [e for e in events if isinstance(e, HookStartEvent)]
        assert [e.hook_name for e in start_events] == ["first"]

    @pytest.mark.asyncio
    async def test_spawn_failure_is_fail_open(self, ctx: HookSessionContext) -> None:
        handler = HooksManager([
            _make_tool_hook(
                "broken", "/nonexistent/hook/binary", type=HookType.PRE_TOOL
            )
        ])
        events = [
            ev
            async for ev in _run(
                handler,
                HookType.PRE_TOOL,
                ctx,
                tool_name="bash",
                tool_call_id="tc1",
                tool_input={},
            )
        ]
        denials = [e for e in events if isinstance(e, HookToolDenial)]
        assert denials == []  # fail-open: no deny on spawn failure
        warnings = [
            e
            for e in events
            if isinstance(e, HookEndEvent) and e.status == HookMessageSeverity.WARNING
        ]
        assert len(warnings) == 1

    @pytest.mark.asyncio
    async def test_strict_failure_denies(self, ctx: HookSessionContext) -> None:
        handler = HooksManager([
            _make_tool_hook("guard", "exit 1", type=HookType.PRE_TOOL, strict=True),
            _make_tool_hook("second", "echo ok", type=HookType.PRE_TOOL),
        ])
        events = [
            ev
            async for ev in _run(
                handler,
                HookType.PRE_TOOL,
                ctx,
                tool_name="bash",
                tool_call_id="tc1",
                tool_input={},
            )
        ]
        denials = [e for e in events if isinstance(e, HookToolDenial)]
        assert len(denials) == 1
        assert denials[0].hook_name == "guard"
        errors = [
            e
            for e in events
            if isinstance(e, HookEndEvent) and e.status == HookMessageSeverity.ERROR
        ]
        assert any("strict" in (e.content or "") for e in errors)
        # Second hook must not have started
        starts = [e for e in events if isinstance(e, HookStartEvent)]
        assert [e.hook_name for e in starts] == ["guard"]

    @pytest.mark.asyncio
    async def test_strict_timeout_denies(self, ctx: HookSessionContext) -> None:
        handler = HooksManager([
            _make_tool_hook(
                "slow", "sleep 10", type=HookType.PRE_TOOL, timeout=0.1, strict=True
            )
        ])
        events = [
            ev
            async for ev in _run(
                handler,
                HookType.PRE_TOOL,
                ctx,
                tool_name="bash",
                tool_call_id="tc1",
                tool_input={},
            )
        ]
        denials = [e for e in events if isinstance(e, HookToolDenial)]
        assert len(denials) == 1
        assert denials[0].hook_name == "slow"


class TestPostToolHook:
    @pytest.mark.asyncio
    async def test_no_hooks_returns_initial_text(self, ctx: HookSessionContext) -> None:
        handler = HooksManager([])
        final_text, events = await _drain_post_tool_chain(
            handler,
            ctx,
            tool_name="bash",
            tool_call_id="tc1",
            tool_input={},
            tool_status="success",
            tool_output={"result": "ok"},
            tool_error=None,
            duration_ms=10.0,
            initial_text="ok",
        )
        assert final_text == "ok"
        assert events == []

    @pytest.mark.asyncio
    async def test_exit_0_passthrough(self, ctx: HookSessionContext) -> None:
        handler = HooksManager([
            _make_tool_hook("audit", "echo ok", type=HookType.POST_TOOL)
        ])
        final_text, events = await _drain_post_tool_chain(
            handler,
            ctx,
            tool_name="bash",
            tool_call_id="tc1",
            tool_input={},
            tool_status="success",
            tool_output={"r": 1},
            tool_error=None,
            duration_ms=10.0,
            initial_text="original",
        )
        assert final_text == "original"
        assert any(isinstance(e, HookRunStartEvent) for e in events)

    @pytest.mark.asyncio
    async def test_decision_deny_replaces_text(self, ctx: HookSessionContext) -> None:
        handler = HooksManager([
            _make_tool_hook("redact", _deny_cmd("REDACTED"), type=HookType.POST_TOOL)
        ])
        final_text, _events = await _drain_post_tool_chain(
            handler,
            ctx,
            tool_name="bash",
            tool_call_id="tc1",
            tool_input={},
            tool_status="success",
            tool_output={"r": 1},
            tool_error=None,
            duration_ms=10.0,
            initial_text="sensitive data",
        )
        assert final_text == "REDACTED"

    @pytest.mark.asyncio
    async def test_decision_deny_no_reason_replaces_with_empty(
        self, ctx: HookSessionContext
    ) -> None:
        handler = HooksManager([
            _make_tool_hook("silence", _deny_cmd(), type=HookType.POST_TOOL)
        ])
        final_text, _events = await _drain_post_tool_chain(
            handler,
            ctx,
            tool_name="bash",
            tool_call_id="tc1",
            tool_input={},
            tool_status="success",
            tool_output={"r": 1},
            tool_error=None,
            duration_ms=10.0,
            initial_text="something",
        )
        assert final_text == ""

    @pytest.mark.asyncio
    async def test_additional_context_appends_to_text(
        self, ctx: HookSessionContext
    ) -> None:
        # hook_specific_output.additional_context is appended (not replaced)
        # to the current tool_output_text.
        payload = {"hook_specific_output": {"additional_context": "[redacted 1 key]"}}
        handler = HooksManager([
            _make_tool_hook("audit", _emit_cmd(payload), type=HookType.POST_TOOL)
        ])
        final_text, _events = await _drain_post_tool_chain(
            handler,
            ctx,
            tool_name="bash",
            tool_call_id="tc1",
            tool_input={},
            tool_status="success",
            tool_output={"r": 1},
            tool_error=None,
            duration_ms=10.0,
            initial_text="original output",
        )
        assert final_text == "original output\n[redacted 1 key]"

    @pytest.mark.asyncio
    async def test_deny_plus_additional_context_combines(
        self, ctx: HookSessionContext
    ) -> None:
        # decision=deny replaces with reason, then additional_context appends
        # to the replacement (Gemini-style combined semantics).
        payload = {
            "decision": "deny",
            "reason": "REDACTED",
            "hook_specific_output": {"additional_context": "(2 secrets stripped)"},
        }
        handler = HooksManager([
            _make_tool_hook("redact", _emit_cmd(payload), type=HookType.POST_TOOL)
        ])
        final_text, _events = await _drain_post_tool_chain(
            handler,
            ctx,
            tool_name="bash",
            tool_call_id="tc1",
            tool_input={},
            tool_status="success",
            tool_output={"r": 1},
            tool_error=None,
            duration_ms=10.0,
            initial_text="sensitive data",
        )
        assert final_text == "REDACTED\n(2 secrets stripped)"

    @pytest.mark.asyncio
    async def test_pipeline_composes_left_to_right(
        self, ctx: HookSessionContext
    ) -> None:
        # First hook replaces with "piped". Second hook reads stdin and emits
        # a JSON deny whose reason is the prior text uppercased.
        upper_script = (
            f'{sys.executable} -c "'
            "import sys,json; "
            "d=json.load(sys.stdin); "
            "sys.stdout.write(json.dumps("
            "{'decision':'deny','reason': d['tool_output_text'].upper()}"
            "))"
            '"'
        )
        handler = HooksManager([
            _make_tool_hook("first", _deny_cmd("piped"), type=HookType.POST_TOOL),
            _make_tool_hook("second", upper_script, type=HookType.POST_TOOL),
        ])
        final_text, _events = await _drain_post_tool_chain(
            handler,
            ctx,
            tool_name="bash",
            tool_call_id="tc1",
            tool_input={},
            tool_status="success",
            tool_output={"r": 1},
            tool_error=None,
            duration_ms=10.0,
            initial_text="initial",
        )
        assert final_text == "PIPED"

    @pytest.mark.asyncio
    async def test_decision_deny_on_failure_status_still_replaces(
        self, ctx: HookSessionContext
    ) -> None:
        handler = HooksManager([
            _make_tool_hook(
                "rescue", _deny_cmd("synthetic recovery"), type=HookType.POST_TOOL
            )
        ])
        final_text, _events = await _drain_post_tool_chain(
            handler,
            ctx,
            tool_name="bash",
            tool_call_id="tc1",
            tool_input={},
            tool_status="failure",
            tool_output=None,
            tool_error="boom",
            duration_ms=0.0,
            initial_text="<tool_error>boom</tool_error>",
        )
        assert final_text == "synthetic recovery"

    @pytest.mark.asyncio
    async def test_invocation_includes_status_and_output(
        self, ctx: HookSessionContext
    ) -> None:
        # Hook script asserts the invocation payload contains the expected
        # fields and writes a JSON deny whose reason confirms success.
        script = (
            f'{sys.executable} -c "'
            "import sys,json; "
            "d=json.load(sys.stdin); "
            "assert d['hook_event_name'] == 'post_tool'; "
            "assert d['tool_status'] == 'success'; "
            "assert d['tool_output'] == {'r': 1}; "
            "assert d['tool_name'] == 'bash'; "
            "assert d['tool_call_id'] == 'tc1'; "
            "sys.stdout.write(json.dumps("
            "{'decision':'deny','reason':'asserts passed'}"
            "))"
            '"'
        )
        handler = HooksManager([
            _make_tool_hook("inspect", script, type=HookType.POST_TOOL)
        ])
        final_text, _events = await _drain_post_tool_chain(
            handler,
            ctx,
            tool_name="bash",
            tool_call_id="tc1",
            tool_input={"command": "ls"},
            tool_status="success",
            tool_output={"r": 1},
            tool_error=None,
            duration_ms=10.0,
            initial_text="ignored",
        )
        assert final_text == "asserts passed"

    @pytest.mark.asyncio
    async def test_strict_failure_empties_text(self, ctx: HookSessionContext) -> None:
        handler = HooksManager([
            _make_tool_hook("guard", "exit 1", type=HookType.POST_TOOL, strict=True),
            _make_tool_hook("second", _deny_cmd("replaced"), type=HookType.POST_TOOL),
        ])
        final_text, events = await _drain_post_tool_chain(
            handler,
            ctx,
            tool_name="bash",
            tool_call_id="tc1",
            tool_input={},
            tool_status="success",
            tool_output={"r": 1},
            tool_error=None,
            duration_ms=10.0,
            initial_text="sensitive data",
        )
        assert final_text == ""
        errors = [
            e
            for e in events
            if isinstance(e, HookEndEvent) and e.status == HookMessageSeverity.ERROR
        ]
        assert any("strict" in (e.content or "") for e in errors)
        # Second hook must not have started
        starts = [e for e in events if isinstance(e, HookStartEvent)]
        assert [e.hook_name for e in starts] == ["guard"]

    @pytest.mark.asyncio
    async def test_strict_timeout_empties_text(self, ctx: HookSessionContext) -> None:
        handler = HooksManager([
            _make_tool_hook(
                "slow", "sleep 10", type=HookType.POST_TOOL, timeout=0.1, strict=True
            )
        ])
        final_text, _events = await _drain_post_tool_chain(
            handler,
            ctx,
            tool_name="bash",
            tool_call_id="tc1",
            tool_input={},
            tool_status="success",
            tool_output={"r": 1},
            tool_error=None,
            duration_ms=10.0,
            initial_text="sensitive data",
        )
        assert final_text == ""


class TestAfterToolStatusMatching:
    """`match_status` filters after_tool hooks by `tool_status` (unset =
    all statuses), covering Claude Code's PostToolUseFailure without a
    separate event.
    """

    async def _statuses_fired(
        self,
        ctx: HookSessionContext,
        hook: HookConfig,
        statuses: tuple[str, ...] = ("success", "failure", "cancelled"),
        tool_name: str = "bash",
    ) -> list[str]:
        handler = HooksManager([hook])
        fired: list[str] = []
        for status in statuses:
            events = [
                ev
                async for ev in _run(
                    handler,
                    HookType.POST_TOOL,
                    ctx,
                    tool_name=tool_name,
                    tool_call_id="tc1",
                    tool_input={},
                    tool_status=status,
                    tool_output=None,
                    tool_error=None,
                    duration_ms=1.0,
                )
            ]
            if any(isinstance(e, HookStartEvent) for e in events):
                fired.append(status)
        return fired

    @pytest.mark.asyncio
    async def test_match_status_failure_fires_only_on_failure(
        self, ctx: HookSessionContext
    ) -> None:
        hook = _make_tool_hook(
            "on-fail", "echo ok", type=HookType.POST_TOOL, match_status="failure"
        )
        assert await self._statuses_fired(ctx, hook) == ["failure"]

    @pytest.mark.asyncio
    async def test_match_status_unset_fires_on_all_statuses(
        self, ctx: HookSessionContext
    ) -> None:
        hook = _make_tool_hook("always", "echo ok", type=HookType.POST_TOOL)
        assert await self._statuses_fired(ctx, hook) == [
            "success",
            "failure",
            "cancelled",
        ]

    @pytest.mark.asyncio
    async def test_match_status_combines_with_name_match(
        self, ctx: HookSessionContext
    ) -> None:
        hook = _make_tool_hook(
            "bash-fail",
            "echo ok",
            type=HookType.POST_TOOL,
            match="bash",
            match_status="failure",
        )
        # Right name, right status.
        assert await self._statuses_fired(ctx, hook, tool_name="bash") == ["failure"]
        # Wrong name never fires, regardless of status.
        assert await self._statuses_fired(ctx, hook, tool_name="grep") == []

    def test_match_status_parses_from_config(self) -> None:
        hook = HookConfig(
            name="on-fail",
            type=HookType.POST_TOOL,
            command="echo ok",
            match_status="failure",
        )
        assert hook.match_status == "failure"

    def test_match_status_rejects_unknown_value(self) -> None:
        with pytest.raises(ValueError):
            HookConfig(
                name="bad",
                type=HookType.POST_TOOL,
                command="echo ok",
                match_status="exploded",  # type: ignore[arg-type]
            )

    @pytest.mark.parametrize(
        "hook_type",
        [
            HookType.POST_AGENT,
            HookType.PRE_TOOL,
            HookType.PERMISSION_REQUEST,
            HookType.SESSION_START,
        ],
    )
    def test_match_status_forbidden_on_other_types(self, hook_type: HookType) -> None:
        with pytest.raises(
            ValueError, match="match_status is only valid for post_tool"
        ):
            HookConfig(
                name="bad",
                type=hook_type,
                command="echo ok",
                match_status="failure",
            )


class TestStrictValidation:
    def test_strict_forbidden_on_post_agent(self) -> None:
        with pytest.raises(ValueError, match="strict is only valid for tool hooks"):
            HookConfig(
                name="bad", type=HookType.POST_AGENT, command="echo ok", strict=True
            )

    def test_strict_allowed_on_pre_tool(self) -> None:
        hook = HookConfig(
            name="guard", type=HookType.PRE_TOOL, command="echo ok", strict=True
        )
        assert hook.strict is True

    def test_strict_allowed_on_post_tool(self) -> None:
        hook = HookConfig(
            name="redact", type=HookType.POST_TOOL, command="echo ok", strict=True
        )
        assert hook.strict is True


def _stub_tool_call(call_id: str = "call_1", arguments: str = "{}") -> ToolCall:
    return ToolCall(
        id=call_id,
        index=0,
        function=FunctionCall(name="stub_tool", arguments=arguments),
    )


class TestStructuredResponseParsing:
    def test_empty_stdout_returns_none(self) -> None:
        # Empty stdout is the only legitimate "passthrough" signal — the
        # hook explicitly chose to do nothing.
        assert _parse_structured_response("") is None

    def test_non_json_stdout_raises(self) -> None:
        # Any non-empty stdout MUST be a structured response. Free-form
        # text (debug logs, accidental prints) is a contract violation;
        # diagnostics belong on stderr.
        with pytest.raises(HookOutputError, match="not valid JSON"):
            _parse_structured_response("hello world")

    def test_truncated_json_raises(self) -> None:
        with pytest.raises(HookOutputError, match="not valid JSON"):
            _parse_structured_response('{"decision": "deny", "reason": "no"')

    def test_json_array_raises(self) -> None:
        with pytest.raises(HookOutputError, match="expected an object"):
            _parse_structured_response("[1, 2, 3]")

    def test_json_scalar_raises(self) -> None:
        with pytest.raises(HookOutputError, match="expected an object"):
            _parse_structured_response('"just a string"')

    def test_schema_mismatch_raises(self) -> None:
        # "maybe" is not a valid Literal value for `decision`.
        with pytest.raises(HookOutputError, match="schema"):
            _parse_structured_response('{"decision": "maybe"}')

    def test_empty_object_parses_to_passthrough(self) -> None:
        # {} is valid: no rewrite, no system_message, just an explicit OK.
        result = _parse_structured_response("{}")
        assert result is not None
        assert result.hook_specific_output.tool_input is None
        assert result.system_message is None

    def test_unknown_fields_ignored(self) -> None:
        # Forward-compat: reserved fields we may grow into are tolerated.
        result = _parse_structured_response(
            '{"decision": "allow", "continue": false, "future_field": 42}'
        )
        assert result is not None
        assert result.hook_specific_output.tool_input is None

    def test_unknown_nested_fields_ignored(self) -> None:
        result = _parse_structured_response(
            '{"hook_specific_output": {"future_subfield": "x"}}'
        )
        assert result is not None
        assert result.hook_specific_output.tool_input is None

    def test_tool_input_parses(self) -> None:
        result = _parse_structured_response(
            '{"hook_specific_output": {"tool_input": {"command": "ls -la"}}}'
        )
        assert result is not None
        assert result.hook_specific_output.tool_input == {"command": "ls -la"}

    def test_top_level_tool_input_ignored(self) -> None:
        # Backwards-incompatible safeguard: a flat tool_input at the top
        # level (the v1 shape before nesting) is silently ignored.
        result = _parse_structured_response('{"tool_input": {"command": "x"}}')
        assert result is not None
        assert result.hook_specific_output.tool_input is None

    def test_system_message_parses(self) -> None:
        result = _parse_structured_response('{"system_message": "audited"}')
        assert result is not None
        assert result.system_message == "audited"

    def test_default_construct_defaults(self) -> None:
        # The model's defaults are stable so manager logic can rely on them.
        m = HookStructuredResponse()
        assert m.system_message is None
        assert m.hook_specific_output.tool_input is None


class TestPreToolRewrite:
    @pytest.mark.asyncio
    async def test_single_hook_rewrites_tool_input(
        self, ctx: HookSessionContext
    ) -> None:
        script = (
            f'{sys.executable} -c "'
            "import json,sys; "
            "json.dump({'hook_specific_output': "
            "{'tool_input': {'command': 'echo rewritten'}}}, sys.stdout)"
            '"'
        )
        handler = HooksManager([
            _make_tool_hook("rewriter", script, type=HookType.PRE_TOOL, match="bash")
        ])
        events = [
            ev
            async for ev in _run(
                handler,
                HookType.PRE_TOOL,
                ctx,
                tool_name="bash",
                tool_call_id="tc1",
                tool_input={"command": "echo original"},
            )
        ]
        rewrites = [e for e in events if isinstance(e, HookToolInputRewrite)]
        assert len(rewrites) == 1
        assert rewrites[0].hook_name == "rewriter"
        assert rewrites[0].tool_input == {"command": "echo rewritten"}
        # No denial, no after-tool replacement event
        assert not any(isinstance(e, HookToolDenial) for e in events)

    @pytest.mark.asyncio
    async def test_rewrite_pipeline_composes_left_to_right(
        self, ctx: HookSessionContext
    ) -> None:
        # First hook prepends "echo "; second hook reads its piped input and
        # uppercases the command. The second hook must see the FIRST hook's
        # rewrite, proving manager threads tool_input through the chain.
        first = (
            f'{sys.executable} -c "'
            "import json,sys; "
            "d=json.load(sys.stdin); "
            "cmd=d['tool_input'].get('command',''); "
            "json.dump({'hook_specific_output': "
            "{'tool_input': {**d['tool_input'], 'command': 'echo '+cmd}}}, sys.stdout)"
            '"'
        )
        second = (
            f'{sys.executable} -c "'
            "import json,sys; "
            "d=json.load(sys.stdin); "
            "cmd=d['tool_input'].get('command',''); "
            "json.dump({'hook_specific_output': "
            "{'tool_input': {**d['tool_input'], 'command': cmd.upper()}}}, sys.stdout)"
            '"'
        )
        handler = HooksManager([
            _make_tool_hook("first", first, type=HookType.PRE_TOOL),
            _make_tool_hook("second", second, type=HookType.PRE_TOOL),
        ])
        events = [
            ev
            async for ev in _run(
                handler,
                HookType.PRE_TOOL,
                ctx,
                tool_name="bash",
                tool_call_id="tc1",
                tool_input={"command": "hi"},
            )
        ]
        # The manager emits one HookToolInputRewrite per rewriting hook
        # (in chronological order), each carrying the cumulative
        # ``tool_input`` at that step. The agent loop validates each as
        # it arrives and aborts the chain on the first invalid one.
        rewrites = [e for e in events if isinstance(e, HookToolInputRewrite)]
        assert [r.hook_name for r in rewrites] == ["first", "second"]
        assert rewrites[0].tool_input == {"command": "echo hi"}
        assert rewrites[1].tool_input == {"command": "ECHO HI"}

    @pytest.mark.asyncio
    async def test_rewrite_chain_streams_per_hook(
        self, ctx: HookSessionContext
    ) -> None:
        # Three hooks each rewriting; expect exactly three
        # HookToolInputRewrite events in the stream, one per hook, each
        # attributed to its source.
        def script(out: str) -> str:
            return (
                f'{sys.executable} -c "'
                "import json,sys; "
                "json.dump({'hook_specific_output': "
                f"{{'tool_input': {{'command': {out!r}}}}}}}, sys.stdout)"
                '"'
            )

        handler = HooksManager([
            _make_tool_hook("a", script("a"), type=HookType.PRE_TOOL),
            _make_tool_hook("b", script("b"), type=HookType.PRE_TOOL),
            _make_tool_hook("c", script("c"), type=HookType.PRE_TOOL),
        ])
        events = [
            ev
            async for ev in _run(
                handler,
                HookType.PRE_TOOL,
                ctx,
                tool_name="bash",
                tool_call_id="tc1",
                tool_input={"command": "orig"},
            )
        ]
        rewrites = [e for e in events if isinstance(e, HookToolInputRewrite)]
        assert [r.hook_name for r in rewrites] == ["a", "b", "c"]
        assert [r.tool_input["command"] for r in rewrites] == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_structured_system_message_shown_on_passthrough(
        self, ctx: HookSessionContext
    ) -> None:
        script = (
            f'{sys.executable} -c "'
            "import json,sys; "
            "json.dump({'system_message': 'logged'}, sys.stdout)"
            '"'
        )
        handler = HooksManager([
            _make_tool_hook("audit", script, type=HookType.PRE_TOOL)
        ])
        events = [
            ev
            async for ev in _run(
                handler,
                HookType.PRE_TOOL,
                ctx,
                tool_name="bash",
                tool_call_id="tc1",
                tool_input={"command": "ls"},
            )
        ]
        ends = [e for e in events if isinstance(e, HookEndEvent)]
        assert len(ends) == 1
        assert ends[0].content == "logged"
        # No rewrite, no denial
        assert not any(isinstance(e, HookToolInputRewrite) for e in events)
        assert not any(isinstance(e, HookToolDenial) for e in events)

    @pytest.mark.asyncio
    async def test_non_json_stdout_is_a_warning(self, ctx: HookSessionContext) -> None:
        # The contract is strict: stdout is for the JSON response, full
        # stop. A hook that prints free-form text on stdout (e.g. debug
        # logs that should have gone to stderr) is treated as a failure
        # — surfaced as a UI warning, no denial, no rewrite.
        handler = HooksManager([
            _make_tool_hook(
                "chatty", "echo 'just some debug output'", type=HookType.PRE_TOOL
            )
        ])
        events = [
            ev
            async for ev in _run(
                handler,
                HookType.PRE_TOOL,
                ctx,
                tool_name="bash",
                tool_call_id="tc1",
                tool_input={"command": "ls"},
            )
        ]
        assert not any(isinstance(e, HookToolDenial) for e in events)
        assert not any(isinstance(e, HookToolInputRewrite) for e in events)
        warnings = [
            e
            for e in events
            if isinstance(e, HookEndEvent) and e.status == HookMessageSeverity.WARNING
        ]
        assert len(warnings) == 1
        assert warnings[0].content and "not valid JSON" in warnings[0].content

    @pytest.mark.asyncio
    async def test_strict_mode_escalates_invalid_stdout_to_denial(
        self, ctx: HookSessionContext
    ) -> None:
        # With strict=true the "bad stdout" path is escalated through
        # HookHandler.on_strict_failure — exactly like a non-zero exit
        # would be. For pre_tool that means denying the call with the
        # parse error as the reason.
        handler = HooksManager([
            _make_tool_hook(
                "guard", "echo 'not actually json'", type=HookType.PRE_TOOL, strict=True
            )
        ])
        events = [
            ev
            async for ev in _run(
                handler,
                HookType.PRE_TOOL,
                ctx,
                tool_name="bash",
                tool_call_id="tc1",
                tool_input={"command": "ls"},
            )
        ]
        denials = [e for e in events if isinstance(e, HookToolDenial)]
        assert len(denials) == 1
        assert "invalid response" in denials[0].content


class TestAgentLoopIntegration:
    @pytest.mark.asyncio
    async def test_post_agent_hook_runs_after_turn(self) -> None:
        backend = FakeBackend(mock_llm_chunk(content="Hello!"))
        hooks = [_make_hook(name="post-lint", command="echo ok")]
        agent_loop = build_test_agent_loop(
            backend=backend, hook_config_result=HookConfigResult(hooks=hooks, issues=[])
        )

        events = [ev async for ev in agent_loop.act("hi")]
        event_types = [type(e).__name__ for e in events]
        assert "HookStartEvent" in event_types
        assert "HookEndEvent" in event_types

    @pytest.mark.asyncio
    async def test_post_agent_hook_retry_reinjects_message(self) -> None:
        backend = FakeBackend([
            [mock_llm_chunk(content="first response")],
            [mock_llm_chunk(content="second response")],
        ])

        counter_file = Path.cwd() / ".hook_counter"
        # On the first call, emit a JSON deny so the manager treats it as a
        # retry-with-reason; on the second call, emit nothing so the agent
        # loop terminates normally.
        script = (
            f'{sys.executable} -c "'
            f"from pathlib import Path; "
            f"import sys, json; "
            f"p = Path({str(counter_file)!r}); "
            f"c = int(p.read_text()) if p.exists() else 0; "
            f"p.write_text(str(c + 1)); "
            f"sys.stdout.write(json.dumps({{'decision':'deny','reason':'fix this'}}) if c == 0 else '')"
            f'"'
        )
        hooks = [_make_hook(name="retry-hook", command=script)]
        agent_loop = build_test_agent_loop(
            backend=backend, hook_config_result=HookConfigResult(hooks=hooks, issues=[])
        )

        events = [ev async for ev in agent_loop.act("hi")]
        assistant_events = [e for e in events if isinstance(e, AssistantEvent)]
        assert len(assistant_events) == 2

        user_messages = [
            m for m in agent_loop.messages if m.role.value == "user" and m.injected
        ]
        assert any("fix this" in (m.content or "") for m in user_messages)

    @pytest.mark.asyncio
    async def test_no_hooks_no_events(self) -> None:
        backend = FakeBackend(mock_llm_chunk(content="Hello!"))
        agent_loop = build_test_agent_loop(backend=backend)

        events = [ev async for ev in agent_loop.act("hi")]
        hook_events = [
            e for e in events if isinstance(e, (HookStartEvent, HookEndEvent))
        ]
        assert hook_events == []

    @pytest.mark.asyncio
    async def test_pre_tool_deny_prevents_invocation(self) -> None:
        tool_call = _stub_tool_call("call_block")
        config = build_test_vibe_config(enabled_tools=["stub_tool"])
        hooks = [
            _make_tool_hook(
                "deny-stub",
                _deny_cmd("denied by policy"),
                type=HookType.PRE_TOOL,
                match="stub_tool",
            )
        ]
        backend = FakeBackend([
            [mock_llm_chunk(content="Calling stub.", tool_calls=[tool_call])],
            [mock_llm_chunk(content="ok then")],
        ])
        agent_loop = build_test_agent_loop(
            config=config,
            agent_name=BuiltinAgentName.AUTO_APPROVE,
            backend=backend,
            hook_config_result=HookConfigResult(hooks=hooks, issues=[]),
        )
        agent_loop.tool_manager._all_tools["stub_tool"] = FakeTool

        events = [ev async for ev in agent_loop.act("run it")]
        tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
        assert len(tool_results) == 1
        assert tool_results[0].skipped is True
        assert tool_results[0].skip_reason is not None
        assert "denied by policy" in tool_results[0].skip_reason
        assert agent_loop.stats.tool_calls_hook_denied == 1
        assert agent_loop.stats.tool_calls_rejected == 0

    @pytest.mark.asyncio
    async def test_pre_tool_deny_payload_appears_in_messages(self) -> None:
        tool_call = _stub_tool_call("call_msg")
        config = build_test_vibe_config(enabled_tools=["stub_tool"])
        hooks = [
            _make_tool_hook(
                "deny", _deny_cmd("forbidden"), type=HookType.PRE_TOOL, match="*"
            )
        ]
        backend = FakeBackend([
            [mock_llm_chunk(content="Try.", tool_calls=[tool_call])],
            [mock_llm_chunk(content="acknowledged")],
        ])
        agent_loop = build_test_agent_loop(
            config=config,
            agent_name=BuiltinAgentName.AUTO_APPROVE,
            backend=backend,
            hook_config_result=HookConfigResult(hooks=hooks, issues=[]),
        )
        agent_loop.tool_manager._all_tools["stub_tool"] = FakeTool

        async for _ev in agent_loop.act("go"):
            pass

        tool_msgs = [m for m in agent_loop.messages if m.role.value == "tool"]
        assert any("forbidden" in (m.content or "") for m in tool_msgs)

    @pytest.mark.asyncio
    async def test_post_tool_replaces_llm_text_not_event(self) -> None:
        tool_call = _stub_tool_call("call_after")
        config = build_test_vibe_config(enabled_tools=["stub_tool"])
        hooks = [
            _make_tool_hook(
                "rewrite",
                _deny_cmd("REWRITTEN"),
                type=HookType.POST_TOOL,
                match="stub_tool",
            )
        ]
        backend = FakeBackend([
            [mock_llm_chunk(content="Calling.", tool_calls=[tool_call])],
            [mock_llm_chunk(content="done")],
        ])
        agent_loop = build_test_agent_loop(
            config=config,
            agent_name=BuiltinAgentName.AUTO_APPROVE,
            backend=backend,
            hook_config_result=HookConfigResult(hooks=hooks, issues=[]),
        )
        agent_loop.tool_manager._all_tools["stub_tool"] = FakeTool

        events = [ev async for ev in agent_loop.act("go")]

        tool_results = [
            e for e in events if isinstance(e, ToolResultEvent) and not e.skipped
        ]
        assert len(tool_results) == 1
        # UI event preserves the original result_model
        assert tool_results[0].result is not None

        # But the LLM-bound message has been replaced.
        tool_msgs = [m for m in agent_loop.messages if m.role.value == "tool"]
        assert any((m.content or "").strip() == "REWRITTEN" for m in tool_msgs)

    @pytest.mark.asyncio
    async def test_post_tool_matcher_skips_non_matching(self) -> None:
        tool_call = _stub_tool_call("call_nope")
        config = build_test_vibe_config(enabled_tools=["stub_tool"])
        hooks = [
            _make_tool_hook(
                "wrong-match",
                _deny_cmd("should not run"),
                type=HookType.POST_TOOL,
                match="bash",
            )
        ]
        backend = FakeBackend([
            [mock_llm_chunk(content="Calling.", tool_calls=[tool_call])],
            [mock_llm_chunk(content="done")],
        ])
        agent_loop = build_test_agent_loop(
            config=config,
            agent_name=BuiltinAgentName.AUTO_APPROVE,
            backend=backend,
            hook_config_result=HookConfigResult(hooks=hooks, issues=[]),
        )
        agent_loop.tool_manager._all_tools["stub_tool"] = FakeTool

        async for _ev in agent_loop.act("go"):
            pass

        tool_msgs = [m for m in agent_loop.messages if m.role.value == "tool"]
        # The hook's stdout must not appear in the tool message — the matcher
        # skipped it.
        assert not any("should not run" in (m.content or "") for m in tool_msgs)

    @pytest.mark.asyncio
    async def test_pre_tool_rewrite_applies_to_tool_invocation(self) -> None:
        # The hook rewrites tool_input so the tool runs with text="rewritten".
        # The result message echoes that value (FakeTool returns it as
        # `message`), proving the rewrite reached the tool.
        tool_call = _stub_tool_call("call_rw", arguments='{"text": "original"}')
        config = build_test_vibe_config(enabled_tools=["stub_tool"])
        script = (
            f'{sys.executable} -c "'
            "import json,sys; "
            "json.dump({'hook_specific_output': "
            "{'tool_input': {'text': 'rewritten'}}}, sys.stdout)"
            '"'
        )
        hooks = [
            _make_tool_hook(
                "rewriter", script, type=HookType.PRE_TOOL, match="stub_tool"
            )
        ]
        backend = FakeBackend([
            [mock_llm_chunk(content="Calling.", tool_calls=[tool_call])],
            [mock_llm_chunk(content="done")],
        ])
        agent_loop = build_test_agent_loop(
            config=config,
            agent_name=BuiltinAgentName.AUTO_APPROVE,
            backend=backend,
            hook_config_result=HookConfigResult(hooks=hooks, issues=[]),
        )
        agent_loop.tool_manager._all_tools["stub_tool"] = FakeTool

        async for _ev in agent_loop.act("go"):
            pass

        tool_msgs = [m for m in agent_loop.messages if m.role.value == "tool"]
        assert any("message: rewritten" in (m.content or "") for m in tool_msgs)
        # And the assistant message's tool_call arguments were patched so
        # subsequent LLM turns see what actually ran.
        assistant_with_calls = [
            m
            for m in agent_loop.messages
            if m.role.value == "assistant" and m.tool_calls
        ]
        assert assistant_with_calls
        last_tool_calls = assistant_with_calls[-1].tool_calls
        assert last_tool_calls is not None
        tc_args = last_tool_calls[0].function.arguments
        assert tc_args is not None
        assert '"text": "rewritten"' in tc_args

    @pytest.mark.asyncio
    async def test_pre_tool_rewrite_is_persisted_to_messages_jsonl(self) -> None:
        # The in-memory patch (covered above) is necessary but not
        # sufficient: the on-disk ``messages.jsonl`` must also reflect the
        # rewritten args, otherwise a resumed session would replay the
        # model's original (never-actually-ran) intent to the LLM.
        tool_call = _stub_tool_call("call_persist", arguments='{"text": "original"}')
        config = build_test_vibe_config(
            enabled_tools=["stub_tool"],
            session_logging=SessionLoggingConfig(enabled=True),
        )
        script = (
            f'{sys.executable} -c "'
            "import json,sys; "
            "json.dump({'hook_specific_output': "
            "{'tool_input': {'text': 'rewritten'}}}, sys.stdout)"
            '"'
        )
        hooks = [
            _make_tool_hook(
                "rewriter", script, type=HookType.PRE_TOOL, match="stub_tool"
            )
        ]
        backend = FakeBackend([
            [mock_llm_chunk(content="Calling.", tool_calls=[tool_call])],
            [mock_llm_chunk(content="done")],
        ])
        agent_loop = build_test_agent_loop(
            config=config,
            agent_name=BuiltinAgentName.AUTO_APPROVE,
            backend=backend,
            hook_config_result=HookConfigResult(hooks=hooks, issues=[]),
        )
        agent_loop.tool_manager._all_tools["stub_tool"] = FakeTool

        async for _ev in agent_loop.act("go"):
            pass

        jsonl_path = agent_loop.session_logger.messages_filepath
        lines = [
            json.loads(line)
            for line in jsonl_path.read_text().splitlines()
            if line.strip()
        ]
        assistants_with_calls = [
            m for m in lines if m.get("role") == "assistant" and m.get("tool_calls")
        ]
        assert assistants_with_calls, f"no assistant tool call in {jsonl_path}"
        persisted_args = assistants_with_calls[-1]["tool_calls"][0]["function"][
            "arguments"
        ]
        assert '"text": "rewritten"' in persisted_args, (
            f"messages.jsonl still contains the original args: {persisted_args}"
        )

    @pytest.mark.asyncio
    async def test_pre_tool_rewrite_validation_failure_denies(self) -> None:
        # The hook returns a tool_input with a wrong type for `text` (int
        # instead of str). Re-validation should fail and the rewrite is
        # converted to a denial that the LLM sees as a tool error.
        tool_call = _stub_tool_call("call_bad")
        config = build_test_vibe_config(enabled_tools=["stub_tool"])
        # FakeToolArgs.text is `str`. A list forces a hard type mismatch
        # that pydantic cannot coerce, so we get a real ValidationError.
        script = (
            f'{sys.executable} -c "'
            "import json,sys; "
            "json.dump({'hook_specific_output': "
            "{'tool_input': {'text': [1,2,3]}}}, sys.stdout)"
            '"'
        )
        hooks = [
            _make_tool_hook(
                "bad-rewriter", script, type=HookType.PRE_TOOL, match="stub_tool"
            )
        ]
        backend = FakeBackend([
            [mock_llm_chunk(content="Calling.", tool_calls=[tool_call])],
            [mock_llm_chunk(content="acknowledged")],
        ])
        agent_loop = build_test_agent_loop(
            config=config,
            agent_name=BuiltinAgentName.AUTO_APPROVE,
            backend=backend,
            hook_config_result=HookConfigResult(hooks=hooks, issues=[]),
        )
        agent_loop.tool_manager._all_tools["stub_tool"] = FakeTool

        events = [ev async for ev in agent_loop.act("go")]
        tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
        assert len(tool_results) == 1
        assert tool_results[0].skipped is True
        assert tool_results[0].skip_reason is not None
        assert "failed validation" in tool_results[0].skip_reason
        assert "bad-rewriter" in tool_results[0].skip_reason
        assert agent_loop.stats.tool_calls_hook_denied == 1

    @pytest.mark.asyncio
    async def test_invalid_intermediate_rewrite_stops_subsequent_hooks(self) -> None:
        # Hook 1 produces an invalid tool_input (text=[1,2,3] but the
        # schema expects str). Hook 2 would have produced a valid rewrite
        # but must NEVER run: the agent loop validates after each hook
        # and aborts the chain at the first failure.
        tool_call = _stub_tool_call("call_abort")
        config = build_test_vibe_config(enabled_tools=["stub_tool"])
        bad_script = (
            f'{sys.executable} -c "'
            "import json,sys; "
            "json.dump({'hook_specific_output': "
            "{'tool_input': {'text': [1,2,3]}}}, sys.stdout)"
            '"'
        )
        sentinel = Path.cwd() / ".second_hook_ran"
        # If the second hook ever runs it would touch this file, which we
        # then assert was NOT created.
        good_script = (
            f'{sys.executable} -c "'
            "from pathlib import Path; "
            f"Path({str(sentinel)!r}).write_text('ran'); "
            "import sys,json; "
            "json.dump({'hook_specific_output': "
            "{'tool_input': {'text': 'salvaged'}}}, sys.stdout)"
            '"'
        )
        hooks = [
            _make_tool_hook(
                "broken", bad_script, type=HookType.PRE_TOOL, match="stub_tool"
            ),
            _make_tool_hook(
                "would-fix", good_script, type=HookType.PRE_TOOL, match="stub_tool"
            ),
        ]
        backend = FakeBackend([
            [mock_llm_chunk(content="Calling.", tool_calls=[tool_call])],
            [mock_llm_chunk(content="acknowledged")],
        ])
        agent_loop = build_test_agent_loop(
            config=config,
            agent_name=BuiltinAgentName.AUTO_APPROVE,
            backend=backend,
            hook_config_result=HookConfigResult(hooks=hooks, issues=[]),
        )
        agent_loop.tool_manager._all_tools["stub_tool"] = FakeTool

        events = [ev async for ev in agent_loop.act("go")]
        tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
        assert len(tool_results) == 1
        assert tool_results[0].skipped is True
        assert tool_results[0].skip_reason is not None
        assert "broken" in tool_results[0].skip_reason
        assert not sentinel.exists(), (
            "second hook should NOT have run after the first hook's invalid rewrite"
        )

    @pytest.mark.asyncio
    async def test_serialize_tool_input_failure_rejects_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tool_call = _stub_tool_call("call_set")
        config = build_test_vibe_config(enabled_tools=["stub_tool"])
        backend = FakeBackend([
            [mock_llm_chunk(content="Calling.", tool_calls=[tool_call])],
            [mock_llm_chunk(content="ok")],
        ])
        agent_loop = build_test_agent_loop(
            config=config, agent_name=BuiltinAgentName.AUTO_APPROVE, backend=backend
        )
        agent_loop.tool_manager._all_tools["stub_tool"] = FakeTool

        original = FakeToolArgs.model_dump

        def _blow_up(self: Any, **kwargs: Any) -> Any:
            if kwargs.get("mode") == "json":
                raise TypeError("cannot serialize")
            return original(self, **kwargs)

        monkeypatch.setattr(FakeToolArgs, "model_dump", _blow_up)

        events = [ev async for ev in agent_loop.act("go")]
        tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
        assert len(tool_results) == 1
        assert tool_results[0].error is not None
        assert "serialize" in tool_results[0].error.lower()


class TestHookOutputCap:
    @pytest.mark.asyncio
    async def test_stdout_capped_at_limit(
        self, sample_invocation: PostAgentInvocation
    ) -> None:
        from vibe.core.hooks.executor import _MAX_OUTPUT_BYTES

        overflow = _MAX_OUTPUT_BYTES + 4096
        script = (
            f'{sys.executable} -c "'
            f"import sys; sys.stdout.buffer.write(b'A' * {overflow})"
            '"'
        )
        hook = _make_hook(command=script)
        result = await HookExecutor().run(hook, sample_invocation)
        assert result.exit_code == 0
        assert len(result.stdout) <= _MAX_OUTPUT_BYTES

    @pytest.mark.asyncio
    async def test_stderr_capped_at_limit(
        self, sample_invocation: PostAgentInvocation
    ) -> None:
        from vibe.core.hooks.executor import _MAX_OUTPUT_BYTES

        overflow = _MAX_OUTPUT_BYTES + 4096
        script = (
            f'{sys.executable} -c "'
            f"import sys; sys.stderr.buffer.write(b'E' * {overflow})"
            '"'
        )
        hook = _make_hook(command=script)
        result = await HookExecutor().run(hook, sample_invocation)
        assert result.exit_code == 0
        assert len(result.stderr) <= _MAX_OUTPUT_BYTES


# ---------------------------------------------------------------------------
# Lifecycle hooks: user_prompt_submit / session_start / session_end /
# pre_compact. These were added in the Hydrate fork and re-implemented on
# the v2.16+ HookHandler architecture: they share the strict structured
# stdout contract (exit 0 + JSON) with the native tool / turn hooks.
# ---------------------------------------------------------------------------


def _user_prompt_invocation(prompt: str = "hi") -> UserPromptSubmitInvocation:
    return UserPromptSubmitInvocation(
        session_id="sess", transcript_path="", cwd=str(Path.cwd()), prompt=prompt
    )


class TestUserPromptSubmitHook:
    @pytest.mark.asyncio
    async def test_empty_stdout_is_passthrough(self) -> None:
        handler = HooksManager([
            _make_hook(command="true", type=HookType.USER_PROMPT_SUBMIT)
        ])
        events = [ev async for ev in handler.run(_user_prompt_invocation())]
        assert not any(isinstance(e, HookPromptDenial) for e in events)
        assert not any(isinstance(e, HookContextInjection) for e in events)
        ends = [e for e in events if isinstance(e, HookEndEvent)]
        assert ends and ends[0].status == HookMessageSeverity.OK

    @pytest.mark.asyncio
    async def test_additional_context_is_injected(self) -> None:
        handler = HooksManager([
            _make_hook(
                command=_context_cmd("remember X"), type=HookType.USER_PROMPT_SUBMIT
            )
        ])
        events = [ev async for ev in handler.run(_user_prompt_invocation())]
        injections = [e for e in events if isinstance(e, HookContextInjection)]
        assert len(injections) == 1
        assert injections[0].content == "remember X"

    @pytest.mark.asyncio
    async def test_decision_deny_yields_prompt_denial(self) -> None:
        handler = HooksManager([
            _make_hook(command=_deny_cmd("blocked"), type=HookType.USER_PROMPT_SUBMIT)
        ])
        events = [ev async for ev in handler.run(_user_prompt_invocation())]
        denials = [e for e in events if isinstance(e, HookPromptDenial)]
        assert len(denials) == 1
        assert denials[0].reason == "blocked"
        # The deny reason must not leak into the UI end-event content.
        assert not any(
            "blocked" in (e.content or "")
            for e in events
            if isinstance(e, HookEndEvent)
        )

    @pytest.mark.asyncio
    async def test_non_json_stdout_is_fail_open_warning(self) -> None:
        handler = HooksManager([
            _make_hook(command="echo chatty", type=HookType.USER_PROMPT_SUBMIT)
        ])
        events = [ev async for ev in handler.run(_user_prompt_invocation())]
        assert not any(isinstance(e, HookPromptDenial) for e in events)
        warnings = [
            e
            for e in events
            if isinstance(e, HookEndEvent) and e.status == HookMessageSeverity.WARNING
        ]
        assert len(warnings) == 1


class TestSessionStartHook:
    def _invocation(self, source: str = "new") -> SessionStartInvocation:
        return SessionStartInvocation(
            session_id="sess", transcript_path="", cwd=str(Path.cwd()), source=source
        )

    @pytest.mark.asyncio
    async def test_additional_context_is_injected(self) -> None:
        handler = HooksManager([
            _make_hook(command=_context_cmd("session note"), type=HookType.SESSION_START)
        ])
        events = [ev async for ev in handler.run(self._invocation())]
        injections = [e for e in events if isinstance(e, HookContextInjection)]
        assert len(injections) == 1
        assert injections[0].content == "session note"

    @pytest.mark.asyncio
    async def test_deny_is_ignored(self) -> None:
        handler = HooksManager([
            _make_hook(command=_deny_cmd("no"), type=HookType.SESSION_START)
        ])
        events = [ev async for ev in handler.run(self._invocation())]
        assert not any(isinstance(e, HookContextInjection) for e in events)
        warnings = [
            e
            for e in events
            if isinstance(e, HookEndEvent) and e.status == HookMessageSeverity.WARNING
        ]
        assert len(warnings) == 1


class TestPreCompactHook:
    def _invocation(self) -> PreCompactInvocation:
        return PreCompactInvocation(
            session_id="sess",
            transcript_path="",
            cwd=str(Path.cwd()),
            token_estimate_before=1234,
        )

    @pytest.mark.asyncio
    async def test_additional_context_is_injected(self) -> None:
        handler = HooksManager([
            _make_hook(command=_context_cmd("carry over"), type=HookType.PRE_COMPACT)
        ])
        events = [ev async for ev in handler.run(self._invocation())]
        injections = [e for e in events if isinstance(e, HookContextInjection)]
        assert len(injections) == 1
        assert injections[0].content == "carry over"

    @pytest.mark.asyncio
    async def test_deny_is_ignored(self) -> None:
        handler = HooksManager([
            _make_hook(command=_deny_cmd("no"), type=HookType.PRE_COMPACT)
        ])
        events = [ev async for ev in handler.run(self._invocation())]
        assert not any(isinstance(e, HookContextInjection) for e in events)


class TestSessionEndHook:
    def _invocation(self) -> SessionEndInvocation:
        return SessionEndInvocation(
            session_id="sess",
            transcript_path="",
            cwd=str(Path.cwd()),
            reason="exit",
            turn_count=3,
        )

    @pytest.mark.asyncio
    async def test_exit_0_emits_ok(self) -> None:
        handler = HooksManager([
            _make_hook(command="true", type=HookType.SESSION_END)
        ])
        events = [ev async for ev in handler.run(self._invocation())]
        ends = [e for e in events if isinstance(e, HookEndEvent)]
        assert ends and ends[0].status == HookMessageSeverity.OK

    @pytest.mark.asyncio
    async def test_deny_is_ignored(self) -> None:
        handler = HooksManager([
            _make_hook(command=_deny_cmd("stay"), type=HookType.SESSION_END)
        ])
        events = [ev async for ev in handler.run(self._invocation())]
        warnings = [
            e
            for e in events
            if isinstance(e, HookEndEvent) and e.status == HookMessageSeverity.WARNING
        ]
        assert len(warnings) == 1


class TestLifecycleHookConfig:
    def test_match_rejected_on_user_prompt_submit(self) -> None:
        with pytest.raises(ValueError, match="match is only valid for tool hooks"):
            HookConfig(
                name="x",
                type=HookType.USER_PROMPT_SUBMIT,
                command="echo ok",
                match="*",
            )

    def test_strict_rejected_on_session_start(self) -> None:
        with pytest.raises(ValueError, match="strict is only valid for tool hooks"):
            HookConfig(
                name="x",
                type=HookType.SESSION_START,
                command="echo ok",
                strict=True,
            )


class TestLifecycleAgentLoopIntegration:
    @pytest.mark.asyncio
    async def test_user_prompt_submit_deny_blocks_llm_turn(self) -> None:
        backend = FakeBackend(mock_llm_chunk(content="Hello!"))
        hooks = [
            _make_hook(
                name="gate",
                command=_deny_cmd("not allowed"),
                type=HookType.USER_PROMPT_SUBMIT,
            )
        ]
        agent_loop = build_test_agent_loop(
            backend=backend,
            hook_config_result=HookConfigResult(hooks=hooks, issues=[]),
        )
        events = [ev async for ev in agent_loop.act("hi")]
        assistant = [e for e in events if isinstance(e, AssistantEvent)]
        # The denial reason is surfaced and the model's response never runs.
        assert any(e.content == "not allowed" for e in assistant)
        assert not any(e.content == "Hello!" for e in assistant)

    @pytest.mark.asyncio
    async def test_user_prompt_submit_injects_context(self) -> None:
        backend = FakeBackend(mock_llm_chunk(content="ok"))
        hooks = [
            _make_hook(
                name="ctx",
                command=_context_cmd("INJECTED"),
                type=HookType.USER_PROMPT_SUBMIT,
            )
        ]
        agent_loop = build_test_agent_loop(
            backend=backend,
            hook_config_result=HookConfigResult(hooks=hooks, issues=[]),
        )
        _ = [ev async for ev in agent_loop.act("hi")]
        injected = [
            m
            for m in agent_loop.messages
            if getattr(m, "injected", False) and m.content == "INJECTED"
        ]
        assert len(injected) == 1

    @pytest.mark.asyncio
    async def test_session_start_fires_once(self) -> None:
        backend = FakeBackend([
            [mock_llm_chunk(content="one")],
            [mock_llm_chunk(content="two")],
        ])
        hooks = [
            _make_hook(
                name="start",
                command=_context_cmd("STARTED"),
                type=HookType.SESSION_START,
            )
        ]
        agent_loop = build_test_agent_loop(
            backend=backend,
            hook_config_result=HookConfigResult(hooks=hooks, issues=[]),
        )
        _ = [ev async for ev in agent_loop.act("first")]
        _ = [ev async for ev in agent_loop.act("second")]
        injected = [
            m
            for m in agent_loop.messages
            if getattr(m, "injected", False) and m.content == "STARTED"
        ]
        assert len(injected) == 1

    @pytest.mark.asyncio
    async def test_session_end_fires_once_with_turn_count(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "session_end.json"
        script = (
            f"{sys.executable} -c {shlex.quote('import sys; open(' + repr(str(out)) + chr(44) + repr('w') + ').write(sys.stdin.read())')}"
        )
        hooks = [_make_hook(name="end", command=script, type=HookType.SESSION_END)]
        backend = FakeBackend([
            [mock_llm_chunk(content="one")],
            [mock_llm_chunk(content="two")],
        ])
        agent_loop = build_test_agent_loop(
            backend=backend,
            hook_config_result=HookConfigResult(hooks=hooks, issues=[]),
        )
        _ = [ev async for ev in agent_loop.act("first")]
        _ = [ev async for ev in agent_loop.act("second")]
        await agent_loop.fire_session_end(reason="exit")
        # Second call is a no-op (single-fire guard).
        await agent_loop.fire_session_end(reason="exit")

        payload = json.loads(out.read_text())
        assert payload["hook_event_name"] == "session_end"
        assert payload["reason"] == "exit"
        assert payload["turn_count"] == 2

    @pytest.mark.asyncio
    async def test_pre_compact_context_survives_compaction(self) -> None:
        # auto_compact_threshold=1 forces compaction on the first turn; the
        # pre_compact hook's injected context must survive the message reset.
        backend = FakeBackend([
            [mock_llm_chunk(content="<summary>")],
            [mock_llm_chunk(content="<final>")],
        ])
        cfg = build_test_vibe_config(
            models=make_test_models(auto_compact_threshold=1)
        )
        hooks = [
            _make_hook(
                name="precompact",
                command=_context_cmd("SURVIVES"),
                type=HookType.PRE_COMPACT,
            )
        ]
        agent_loop = build_test_agent_loop(
            config=cfg,
            backend=backend,
            hook_config_result=HookConfigResult(hooks=hooks, issues=[]),
        )
        agent_loop.stats.context_tokens = 2

        _ = [ev async for ev in agent_loop.act("Hello")]

        survived = [
            m
            for m in agent_loop.messages
            if getattr(m, "injected", False) and m.content == "SURVIVES"
        ]
        assert len(survived) == 1


def _backend_error(status: int | None, body: str = "") -> BackendError:
    return BackendError(
        provider="test",
        endpoint="http://api.test/v1",
        status=status,
        reason=None,
        headers={},
        body_text=body,
        parsed_error=None,
        model="m",
        payload_summary=PayloadSummary(
            model="m",
            message_count=1,
            approx_chars=1,
            temperature=0.0,
            has_tools=False,
            tool_choice=None,
        ),
    )


class TestPostCompactHook:
    def _invocation(self) -> PostCompactInvocation:
        return PostCompactInvocation(
            session_id="sess",
            transcript_path="",
            cwd=str(Path.cwd()),
            summary_text="the summary",
            token_estimate_before=1234,
        )

    @pytest.mark.asyncio
    async def test_additional_context_is_injected(self) -> None:
        handler = HooksManager([
            _make_hook(command=_context_cmd("post note"), type=HookType.POST_COMPACT)
        ])
        events = [ev async for ev in handler.run(self._invocation())]
        injections = [e for e in events if isinstance(e, HookContextInjection)]
        assert len(injections) == 1
        assert injections[0].content == "post note"

    @pytest.mark.asyncio
    async def test_deny_is_ignored(self) -> None:
        handler = HooksManager([
            _make_hook(command=_deny_cmd("no"), type=HookType.POST_COMPACT)
        ])
        events = [ev async for ev in handler.run(self._invocation())]
        assert not any(isinstance(e, HookContextInjection) for e in events)
        warnings = [
            e
            for e in events
            if isinstance(e, HookEndEvent) and e.status == HookMessageSeverity.WARNING
        ]
        assert len(warnings) == 1


class TestStopFailureHook:
    def _invocation(self) -> StopFailureInvocation:
        return StopFailureInvocation(
            session_id="sess",
            transcript_path="",
            cwd=str(Path.cwd()),
            error_type="rate_limit",
            error_message="rate limited",
            turn_count=2,
        )

    @pytest.mark.asyncio
    async def test_exit_0_emits_ok(self) -> None:
        handler = HooksManager([
            _make_hook(command="true", type=HookType.STOP_FAILURE)
        ])
        events = [ev async for ev in handler.run(self._invocation())]
        ends = [e for e in events if isinstance(e, HookEndEvent)]
        assert ends and ends[0].status == HookMessageSeverity.OK

    @pytest.mark.asyncio
    async def test_deny_is_ignored(self) -> None:
        handler = HooksManager([
            _make_hook(command=_deny_cmd("retry"), type=HookType.STOP_FAILURE)
        ])
        events = [ev async for ev in handler.run(self._invocation())]
        assert not any(isinstance(e, HookContextInjection) for e in events)
        warnings = [
            e
            for e in events
            if isinstance(e, HookEndEvent) and e.status == HookMessageSeverity.WARNING
        ]
        assert len(warnings) == 1

    @pytest.mark.asyncio
    async def test_additional_context_is_ignored(self) -> None:
        handler = HooksManager([
            _make_hook(command=_context_cmd("inject me"), type=HookType.STOP_FAILURE)
        ])
        events = [ev async for ev in handler.run(self._invocation())]
        assert not any(isinstance(e, HookContextInjection) for e in events)


class TestNotificationHook:
    def _invocation(self) -> NotificationInvocation:
        return NotificationInvocation(
            session_id="sess",
            transcript_path="",
            cwd=str(Path.cwd()),
            notification_type="permission_prompt",
            message="Approval requested for tool 'bash'",
            tool_name="bash",
            tool_call_id="call_1",
        )

    @pytest.mark.asyncio
    async def test_exit_0_emits_ok(self) -> None:
        handler = HooksManager([
            _make_hook(command="true", type=HookType.NOTIFICATION)
        ])
        events = [ev async for ev in handler.run(self._invocation())]
        ends = [e for e in events if isinstance(e, HookEndEvent)]
        assert ends and ends[0].status == HookMessageSeverity.OK

    @pytest.mark.asyncio
    async def test_deny_is_ignored(self) -> None:
        handler = HooksManager([
            _make_hook(command=_deny_cmd("no prompt"), type=HookType.NOTIFICATION)
        ])
        events = [ev async for ev in handler.run(self._invocation())]
        assert not any(isinstance(e, HookContextInjection) for e in events)
        warnings = [
            e
            for e in events
            if isinstance(e, HookEndEvent) and e.status == HookMessageSeverity.WARNING
        ]
        assert len(warnings) == 1

    @pytest.mark.asyncio
    async def test_additional_context_is_ignored(self) -> None:
        handler = HooksManager([
            _make_hook(command=_context_cmd("inject me"), type=HookType.NOTIFICATION)
        ])
        events = [ev async for ev in handler.run(self._invocation())]
        assert not any(isinstance(e, HookContextInjection) for e in events)


class TestWaveALifecycleHookConfig:
    def test_new_lifecycle_types_load_from_toml(
        self, config_dir: Path
    ) -> None:
        _write_hooks_toml(
            config_dir / "hooks.toml",
            [
                {"name": "pc", "type": "post_compact", "command": "true"},
                {"name": "sf", "type": "stop_failure", "command": "true"},
                {"name": "nt", "type": "notification", "command": "true"},
            ],
        )
        result = load_hooks_from_fs()
        assert [h.type for h in result.hooks] == [
            HookType.POST_COMPACT,
            HookType.STOP_FAILURE,
            HookType.NOTIFICATION,
        ]
        assert result.issues == []

    def test_match_rejected_on_notification(self) -> None:
        with pytest.raises(ValueError, match="match is only valid for tool hooks"):
            HookConfig(
                name="x", type=HookType.NOTIFICATION, command="echo ok", match="*"
            )

    def test_strict_rejected_on_stop_failure(self) -> None:
        with pytest.raises(ValueError, match="strict is only valid for tool hooks"):
            HookConfig(
                name="x", type=HookType.STOP_FAILURE, command="echo ok", strict=True
            )


class TestTurnFailureClassification:
    def _classify(self, e: BaseException) -> str:
        from vibe.core.agent_loop._loop import _classify_turn_failure

        return _classify_turn_failure(e)

    @pytest.mark.parametrize(
        ("status", "body", "expected"),
        [
            (429, "", "rate_limit"),
            (401, "", "authentication_failed"),
            (403, "", "authentication_failed"),
            (402, "", "billing_error"),
            (404, "", "model_not_found"),
            (400, "", "invalid_request"),
            (422, "", "invalid_request"),
            (400, "maximum context length", "context_too_long"),
            (422, "max_tokens_exceeded", "max_output_tokens"),
            (500, "", "server_error"),
            (502, "", "server_error"),
            (503, "", "overloaded"),
            (529, "", "overloaded"),
            (None, "", "unknown"),
        ],
    )
    def test_backend_error_statuses(
        self, status: int | None, body: str, expected: str
    ) -> None:
        assert self._classify(_backend_error(status, body)) == expected

    def test_wrapped_backend_error_is_classified_via_cause(self) -> None:
        cause = _backend_error(429)
        wrapper = RuntimeError("API error from test (model: m): boom")
        wrapper.__cause__ = cause
        assert self._classify(wrapper) == "rate_limit"

    def test_loop_error_types(self) -> None:
        assert self._classify(RateLimitError("p", "m")) == "rate_limit"
        assert self._classify(ContextTooLongError("p", "m")) == "context_too_long"
        assert self._classify(ResponseTooLongError("p", "m")) == "max_output_tokens"

    def test_unrelated_exception_is_unknown(self) -> None:
        assert self._classify(ValueError("boom")) == "unknown"


class TestWaveALifecycleAgentLoopIntegration:
    def _compacting_loop(self, hooks: list[HookConfig]) -> Any:
        # auto_compact_threshold=1 + context_tokens=2 forces compaction on
        # the first turn; the backend's first stream answers the compaction
        # request (a well-formed summary), the second answers the user turn.
        backend = FakeBackend([
            [mock_llm_chunk(content="<summary>THE SUMMARY</summary>")],
            [mock_llm_chunk(content="<final>")],
        ])
        cfg = build_test_vibe_config(models=make_test_models(auto_compact_threshold=1))
        agent_loop = build_test_agent_loop(
            config=cfg,
            backend=backend,
            hook_config_result=HookConfigResult(hooks=hooks, issues=[]),
        )
        agent_loop.stats.context_tokens = 2
        return agent_loop

    @pytest.mark.asyncio
    async def test_post_compact_fires_after_compaction_with_summary(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "post_compact.json"
        hooks = [
            _make_hook(
                name="postcompact",
                command=_capture_cmd(out),
                type=HookType.POST_COMPACT,
            )
        ]
        agent_loop = self._compacting_loop(hooks)

        _ = [ev async for ev in agent_loop.act("Hello")]

        payload = json.loads(out.read_text())
        assert payload["hook_event_name"] == "post_compact"
        assert payload["reason"] == "auto_compact"
        assert payload["summary_text"] == "THE SUMMARY"
        assert payload["token_estimate_before"] == 2

    @pytest.mark.asyncio
    async def test_post_compact_injection_lands_after_reset(self) -> None:
        hooks = [
            _make_hook(
                name="postcompact",
                command=_context_cmd("POST NOTE"),
                type=HookType.POST_COMPACT,
            )
        ]
        agent_loop = self._compacting_loop(hooks)

        _ = [ev async for ev in agent_loop.act("Hello")]

        injected = [
            m
            for m in agent_loop.messages
            if getattr(m, "injected", False) and m.content == "POST NOTE"
        ]
        assert len(injected) == 1

    @pytest.mark.asyncio
    async def test_pre_compact_fires_before_post_compact(
        self, tmp_path: Path
    ) -> None:
        log = tmp_path / "order.log"
        hooks = [
            _make_hook(
                name="pre",
                command=_append_cmd(log, "pre"),
                type=HookType.PRE_COMPACT,
            ),
            _make_hook(
                name="post",
                command=_append_cmd(log, "post"),
                type=HookType.POST_COMPACT,
            ),
        ]
        agent_loop = self._compacting_loop(hooks)

        _ = [ev async for ev in agent_loop.act("Hello")]

        assert log.read_text().splitlines() == ["pre", "post"]

    @pytest.mark.asyncio
    async def test_stop_failure_fires_on_backend_error_and_propagates(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "stop_failure.json"
        backend = FakeBackend(exception_to_raise=_backend_error(429))
        hooks = [
            _make_hook(
                name="failwatch",
                command=_capture_cmd(out),
                type=HookType.STOP_FAILURE,
            )
        ]
        agent_loop = build_test_agent_loop(
            backend=backend,
            hook_config_result=HookConfigResult(hooks=hooks, issues=[]),
        )

        with pytest.raises(RateLimitError):
            _ = [ev async for ev in agent_loop.act("hi")]

        payload = json.loads(out.read_text())
        assert payload["hook_event_name"] == "stop_failure"
        assert payload["error_type"] == "rate_limit"
        assert payload["turn_count"] == 0
        assert "Rate limits exceeded" in payload["error_message"]

    @pytest.mark.asyncio
    async def test_stop_failure_error_message_truncated(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "stop_failure.json"
        backend = FakeBackend(exception_to_raise=ValueError("x" * 5000))
        hooks = [
            _make_hook(
                name="failwatch",
                command=_capture_cmd(out),
                type=HookType.STOP_FAILURE,
            )
        ]
        agent_loop = build_test_agent_loop(
            backend=backend,
            hook_config_result=HookConfigResult(hooks=hooks, issues=[]),
        )

        with pytest.raises(RuntimeError):
            _ = [ev async for ev in agent_loop.act("hi")]

        payload = json.loads(out.read_text())
        assert payload["error_type"] == "unknown"
        assert len(payload["error_message"]) == 2000

    @pytest.mark.asyncio
    async def test_stop_failure_not_fired_on_clean_turn(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "stop_failure.json"
        backend = FakeBackend(mock_llm_chunk(content="fine"))
        hooks = [
            _make_hook(
                name="failwatch",
                command=_capture_cmd(out),
                type=HookType.STOP_FAILURE,
            )
        ]
        agent_loop = build_test_agent_loop(
            backend=backend,
            hook_config_result=HookConfigResult(hooks=hooks, issues=[]),
        )

        _ = [ev async for ev in agent_loop.act("hi")]

        assert not out.exists()

    @pytest.mark.asyncio
    async def test_notification_fires_before_approval_and_result_unaffected(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "notification.json"
        tool_call = ToolCall(
            id="call_appr",
            index=0,
            function=FunctionCall(name="bash", arguments='{"command":"true"}'),
        )
        backend = FakeBackend([
            [mock_llm_chunk(content="Running.", tool_calls=[tool_call])],
            [mock_llm_chunk(content="Done.")],
        ])
        hooks = [
            _make_hook(
                name="notify", command=_capture_cmd(out), type=HookType.NOTIFICATION
            )
        ]
        agent_loop = build_test_agent_loop(
            config=build_test_vibe_config(enabled_tools=["bash"]),
            backend=backend,
            hook_config_result=HookConfigResult(hooks=hooks, issues=[]),
        )
        hook_fired_before_prompt: list[bool] = []

        events, _prompted = await _drive_with_approval(
            agent_loop,
            "run true",
            on_prompt=lambda _ev: hook_fired_before_prompt.append(out.exists()),
        )

        tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
        assert len(tool_results) == 1
        assert tool_results[0].skipped is False
        # The hook payload had landed before the approval callback ran.
        assert hook_fired_before_prompt == [True]
        payload = json.loads(out.read_text())
        assert payload["hook_event_name"] == "notification"
        assert payload["notification_type"] == "permission_prompt"
        assert payload["tool_name"] == "bash"
        assert payload["tool_call_id"] == "call_appr"
        assert "bash" in payload["message"]

    @pytest.mark.asyncio
    async def test_notification_not_fired_without_approval_prompt(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "notification.json"
        backend = FakeBackend(mock_llm_chunk(content="Hello!"))
        hooks = [
            _make_hook(
                name="notify", command=_capture_cmd(out), type=HookType.NOTIFICATION
            )
        ]
        agent_loop = build_test_agent_loop(
            backend=backend,
            hook_config_result=HookConfigResult(hooks=hooks, issues=[]),
        )

        _ = [ev async for ev in agent_loop.act("hi")]

        assert not out.exists()


class TestSessionStartResumeSource:
    def test_arm_session_start_sets_pending_source(self) -> None:
        agent_loop = build_test_agent_loop()
        agent_loop.arm_session_start("resume")
        assert agent_loop._pending_session_start_source == "resume"

    @pytest.mark.asyncio
    async def test_armed_resume_source_reaches_hook(self, tmp_path: Path) -> None:
        out = tmp_path / "session_start.json"
        backend = FakeBackend(mock_llm_chunk(content="hi"))
        hooks = [
            _make_hook(
                name="start", command=_capture_cmd(out), type=HookType.SESSION_START
            )
        ]
        agent_loop = build_test_agent_loop(
            backend=backend,
            hook_config_result=HookConfigResult(hooks=hooks, issues=[]),
        )
        agent_loop.arm_session_start("resume")

        _ = [ev async for ev in agent_loop.act("hello")]

        payload = json.loads(out.read_text())
        assert payload["hook_event_name"] == "session_start"
        assert payload["source"] == "resume"


# ---------------------------------------------------------------------------
# Wave B: subagent_start / subagent_stop (parent-side task-tool observation)
# ---------------------------------------------------------------------------


def _append_event_name_cmd(log: Path) -> str:
    """Hook command that appends the invocation's hook_event_name to *log*."""
    body = (
        "import sys, json; "
        f"open({str(log)!r}, 'a').write("
        "json.loads(sys.stdin.read())['hook_event_name'] + chr(10))"
    )
    return f"{sys.executable} -c {shlex.quote(body)}"


class TestSubagentHookConfig:
    def test_subagent_types_load_from_toml(
        self, config_dir: Path
    ) -> None:
        _write_hooks_toml(
            config_dir / "hooks.toml",
            [
                {"name": "sas", "type": "subagent_start", "command": "true"},
                {"name": "sap", "type": "subagent_stop", "command": "true"},
            ],
        )
        result = load_hooks_from_fs()
        assert [h.type for h in result.hooks] == [
            HookType.SUBAGENT_START,
            HookType.SUBAGENT_STOP,
        ]
        assert result.issues == []

    def test_match_rejected_on_subagent_start(self) -> None:
        with pytest.raises(ValueError, match="match is only valid for tool hooks"):
            HookConfig(
                name="x", type=HookType.SUBAGENT_START, command="echo ok", match="*"
            )

    def test_strict_rejected_on_subagent_stop(self) -> None:
        with pytest.raises(ValueError, match="strict is only valid for tool hooks"):
            HookConfig(
                name="x", type=HookType.SUBAGENT_STOP, command="echo ok", strict=True
            )


class TestSubagentStartHook:
    def _invocation(self) -> SubagentStartInvocation:
        return SubagentStartInvocation(
            session_id="parent-sess",
            transcript_path="",
            cwd=str(Path.cwd()),
            agent_id="child-sess",
            agent_type="explore",
            task_description="look around",
        )

    @pytest.mark.asyncio
    async def test_additional_context_is_injected(self) -> None:
        handler = HooksManager([
            _make_hook(
                command=_context_cmd("guardrails"), type=HookType.SUBAGENT_START
            )
        ])
        events = [ev async for ev in handler.run(self._invocation())]
        injections = [e for e in events if isinstance(e, HookContextInjection)]
        assert len(injections) == 1
        assert injections[0].content == "guardrails"

    @pytest.mark.asyncio
    async def test_deny_is_ignored(self) -> None:
        handler = HooksManager([
            _make_hook(command=_deny_cmd("no"), type=HookType.SUBAGENT_START)
        ])
        events = [ev async for ev in handler.run(self._invocation())]
        assert not any(isinstance(e, HookContextInjection) for e in events)
        warnings = [
            e
            for e in events
            if isinstance(e, HookEndEvent) and e.status == HookMessageSeverity.WARNING
        ]
        assert len(warnings) == 1


class TestSubagentStopHook:
    def _invocation(self) -> SubagentStopInvocation:
        return SubagentStopInvocation(
            session_id="parent-sess",
            transcript_path="/tmp/child/messages.jsonl",
            cwd=str(Path.cwd()),
            agent_id="child-sess",
            agent_type="explore",
            status="success",
            turn_count=2,
            parent_transcript_path="/tmp/parent/messages.jsonl",
        )

    @pytest.mark.asyncio
    async def test_exit_0_emits_ok(self) -> None:
        handler = HooksManager([
            _make_hook(command="true", type=HookType.SUBAGENT_STOP)
        ])
        events = [ev async for ev in handler.run(self._invocation())]
        ends = [e for e in events if isinstance(e, HookEndEvent)]
        assert ends and ends[0].status == HookMessageSeverity.OK

    @pytest.mark.asyncio
    async def test_deny_is_ignored(self) -> None:
        handler = HooksManager([
            _make_hook(command=_deny_cmd("keep going"), type=HookType.SUBAGENT_STOP)
        ])
        events = [ev async for ev in handler.run(self._invocation())]
        assert not any(isinstance(e, HookContextInjection) for e in events)
        warnings = [
            e
            for e in events
            if isinstance(e, HookEndEvent) and e.status == HookMessageSeverity.WARNING
        ]
        assert len(warnings) == 1


class TestSubagentHookRunners:
    def _loop_with(self, hooks: list[HookConfig]) -> Any:
        return build_test_agent_loop(
            backend=FakeBackend(),
            hook_config_result=HookConfigResult(hooks=hooks, issues=[]),
        )

    @pytest.mark.asyncio
    async def test_subagent_start_payload_carries_parent_session(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "start.json"
        agent_loop = self._loop_with([
            _make_hook(
                name="cap", command=_capture_cmd(out), type=HookType.SUBAGENT_START
            )
        ])
        _ = [
            ev
            async for ev in agent_loop._run_subagent_start_hooks(
                agent_id="child-1",
                agent_type="explore",
                task_description="x" * 600,
            )
        ]
        payload = json.loads(out.read_text())
        assert payload["hook_event_name"] == "subagent_start"
        assert payload["session_id"] == agent_loop.session_id
        assert payload["agent_id"] == "child-1"
        assert payload["agent_type"] == "explore"
        # The task prompt is truncated to 500 chars on the wire.
        assert payload["task_description"] == "x" * 500

    @pytest.mark.asyncio
    async def test_subagent_stop_payload_has_child_transcript_override(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "stop.json"
        agent_loop = self._loop_with([
            _make_hook(
                name="cap", command=_capture_cmd(out), type=HookType.SUBAGENT_STOP
            )
        ])
        _ = [
            ev
            async for ev in agent_loop._run_subagent_stop_hooks(
                agent_id="child-1",
                agent_type="explore",
                status="failure",
                turn_count=3,
                child_transcript_path="/tmp/child/messages.jsonl",
            )
        ]
        payload = json.loads(out.read_text())
        assert payload["hook_event_name"] == "subagent_stop"
        assert payload["session_id"] == agent_loop.session_id
        assert payload["agent_id"] == "child-1"
        assert payload["agent_type"] == "explore"
        assert payload["status"] == "failure"
        assert payload["turn_count"] == 3
        # transcript_path points at the CHILD transcript; the parent's own
        # transcript rides along in parent_transcript_path.
        assert payload["transcript_path"] == "/tmp/child/messages.jsonl"
        assert payload["parent_transcript_path"] == ""


class TestPermissionRequestHookConfig:
    def test_loads_from_toml_with_match_and_strict(
        self, config_dir: Path
    ) -> None:
        _write_hooks_toml(
            config_dir / "hooks.toml",
            [
                {
                    "name": "pr",
                    "type": "permission_request",
                    "command": "true",
                    "match": "bash",
                    "strict": True,
                }
            ],
        )
        result = load_hooks_from_fs()
        assert result.issues == []
        assert len(result.hooks) == 1
        hook = result.hooks[0]
        assert hook.type == HookType.PERMISSION_REQUEST
        assert hook.match == "bash"
        assert hook.strict is True

    def test_match_and_strict_accepted_as_tool_hook(self) -> None:
        hook = HookConfig(
            name="pr",
            type=HookType.PERMISSION_REQUEST,
            command="true",
            match="bash",
            strict=True,
        )
        assert hook.type == HookType.PERMISSION_REQUEST


class TestPermissionRequestResponseParsing:
    def test_base_schema_rejects_ask(self) -> None:
        with pytest.raises(HookOutputError):
            _parse_structured_response('{"decision": "ask"}')

    def test_permission_schema_accepts_ask(self) -> None:
        resp = _parse_structured_response(
            '{"decision": "ask"}', PermissionRequestHookResponse
        )
        assert resp is not None
        assert resp.decision == "ask"

    def test_permission_schema_defaults_to_ask(self) -> None:
        resp = _parse_structured_response("{}", PermissionRequestHookResponse)
        assert resp is not None
        assert resp.decision == "ask"


def _ask_cmd() -> str:
    return _emit_cmd({"decision": "ask"})


def _allow_cmd() -> str:
    return _emit_cmd({"decision": "allow"})


def _permission_invocation(
    tool_name: str = "bash",
    required: list[dict[str, Any]] | None = None,
) -> PermissionRequestInvocation:
    return PermissionRequestInvocation(
        session_id="sess",
        transcript_path="",
        cwd=str(Path.cwd()),
        tool_name=tool_name,
        tool_call_id="call_1",
        tool_input={"command": "true"},
        required_permissions=(
            required
            if required is not None
            else [
                {
                    "scope": "command_pattern",
                    "invocation_pattern": "true",
                    "session_pattern": "true *",
                    "label": "true",
                }
            ]
        ),
    )


async def _drive_with_approval(
    agent_loop: Any,
    prompt: str,
    *,
    response: ApprovalResponse = ApprovalResponse.YES,
    feedback: str | None = None,
    on_prompt: Any = None,
) -> tuple[list[Any], list[str]]:
    """Run ``act`` to completion, resolving any ApprovalRequestEvent with a
    canned response (the broker replaced the old set_approval_callback). Returns
    (events, tool names that reached the approval prompt).
    """
    events: list[Any] = []
    prompted: list[str] = []
    async for ev in agent_loop.act(prompt):
        events.append(ev)
        if isinstance(ev, ApprovalRequestEvent):
            prompted.append(ev.tool_name)
            if on_prompt is not None:
                on_prompt(ev)
            agent_loop.resolve_approval_request(ev.request_id, response, feedback)
    return events, prompted


class TestPermissionRequestHook:
    @pytest.mark.asyncio
    async def test_no_hooks_no_events(self) -> None:
        handler = HooksManager([])
        events = [ev async for ev in handler.run(_permission_invocation())]
        assert events == []

    @pytest.mark.asyncio
    async def test_matcher_filters_non_matching(self) -> None:
        handler = HooksManager([
            _make_tool_hook(
                "policy",
                _allow_cmd(),
                type=HookType.PERMISSION_REQUEST,
                match="bash",
            )
        ])
        events = [
            ev async for ev in handler.run(_permission_invocation("read_file"))
        ]
        assert events == []

    @pytest.mark.asyncio
    async def test_explicit_allow_emits_decision_and_stops_chain(self) -> None:
        handler = HooksManager([
            _make_tool_hook(
                "first", _allow_cmd(), type=HookType.PERMISSION_REQUEST
            ),
            _make_tool_hook(
                "second", _deny_cmd("never runs"), type=HookType.PERMISSION_REQUEST
            ),
        ])
        events = [ev async for ev in handler.run(_permission_invocation())]
        decisions = [e for e in events if isinstance(e, HookPermissionDecision)]
        assert len(decisions) == 1
        assert decisions[0].hook_name == "first"
        assert decisions[0].decision == "allow"
        starts = [e for e in events if isinstance(e, HookStartEvent)]
        assert [e.hook_name for e in starts] == ["first"]

    @pytest.mark.asyncio
    async def test_explicit_deny_with_reason(self) -> None:
        handler = HooksManager([
            _make_tool_hook(
                "guard", _deny_cmd("not on my watch"), type=HookType.PERMISSION_REQUEST
            )
        ])
        events = [ev async for ev in handler.run(_permission_invocation())]
        decisions = [e for e in events if isinstance(e, HookPermissionDecision)]
        assert len(decisions) == 1
        assert decisions[0].decision == "deny"
        assert decisions[0].reason == "not on my watch"

    @pytest.mark.asyncio
    async def test_first_decision_wins_deny_then_allow(self) -> None:
        handler = HooksManager([
            _make_tool_hook(
                "first", _deny_cmd("first says no"), type=HookType.PERMISSION_REQUEST
            ),
            _make_tool_hook(
                "second", _allow_cmd(), type=HookType.PERMISSION_REQUEST
            ),
        ])
        events = [ev async for ev in handler.run(_permission_invocation())]
        decisions = [e for e in events if isinstance(e, HookPermissionDecision)]
        assert len(decisions) == 1
        assert decisions[0].hook_name == "first"
        assert decisions[0].decision == "deny"
        starts = [e for e in events if isinstance(e, HookStartEvent)]
        assert [e.hook_name for e in starts] == ["first"]

    @pytest.mark.asyncio
    async def test_ask_is_passthrough(self) -> None:
        handler = HooksManager([
            _make_tool_hook("obs", _ask_cmd(), type=HookType.PERMISSION_REQUEST)
        ])
        events = [ev async for ev in handler.run(_permission_invocation())]
        assert not any(isinstance(e, HookPermissionDecision) for e in events)
        ends = [e for e in events if isinstance(e, HookEndEvent)]
        assert ends and ends[0].status == HookMessageSeverity.OK

    @pytest.mark.asyncio
    async def test_empty_stdout_is_passthrough(self) -> None:
        handler = HooksManager([
            _make_tool_hook("obs", "true", type=HookType.PERMISSION_REQUEST)
        ])
        events = [ev async for ev in handler.run(_permission_invocation())]
        assert not any(isinstance(e, HookPermissionDecision) for e in events)

    @pytest.mark.asyncio
    async def test_empty_json_object_is_passthrough(self) -> None:
        # Unlike other hook types, the decision default for
        # permission_request is "ask" — `{}` must NOT auto-allow.
        handler = HooksManager([
            _make_tool_hook("obs", _emit_cmd({}), type=HookType.PERMISSION_REQUEST)
        ])
        events = [ev async for ev in handler.run(_permission_invocation())]
        assert not any(isinstance(e, HookPermissionDecision) for e in events)
        ends = [e for e in events if isinstance(e, HookEndEvent)]
        assert ends and ends[0].status == HookMessageSeverity.OK

    @pytest.mark.asyncio
    async def test_plain_text_non_strict_passes_through(self) -> None:
        handler = HooksManager([
            _make_tool_hook(
                "broken", "echo not-json", type=HookType.PERMISSION_REQUEST
            )
        ])
        events = [ev async for ev in handler.run(_permission_invocation())]
        assert not any(isinstance(e, HookPermissionDecision) for e in events)
        warnings = [
            e
            for e in events
            if isinstance(e, HookEndEvent) and e.status == HookMessageSeverity.WARNING
        ]
        assert len(warnings) == 1

    @pytest.mark.asyncio
    async def test_plain_text_strict_denies(self) -> None:
        handler = HooksManager([
            _make_tool_hook(
                "broken",
                "echo not-json",
                type=HookType.PERMISSION_REQUEST,
                strict=True,
            )
        ])
        events = [ev async for ev in handler.run(_permission_invocation())]
        decisions = [e for e in events if isinstance(e, HookPermissionDecision)]
        assert len(decisions) == 1
        assert decisions[0].decision == "deny"
        errors = [
            e
            for e in events
            if isinstance(e, HookEndEvent) and e.status == HookMessageSeverity.ERROR
        ]
        assert any("strict" in (e.content or "") for e in errors)

    @pytest.mark.asyncio
    async def test_non_strict_timeout_passes_through(self) -> None:
        handler = HooksManager([
            _make_tool_hook(
                "slow", "sleep 10", type=HookType.PERMISSION_REQUEST, timeout=0.1
            )
        ])
        events = [ev async for ev in handler.run(_permission_invocation())]
        assert not any(isinstance(e, HookPermissionDecision) for e in events)
        warnings = [
            e
            for e in events
            if isinstance(e, HookEndEvent) and e.status == HookMessageSeverity.WARNING
        ]
        assert len(warnings) == 1

    @pytest.mark.asyncio
    async def test_strict_timeout_denies(self) -> None:
        handler = HooksManager([
            _make_tool_hook(
                "slow",
                "sleep 10",
                type=HookType.PERMISSION_REQUEST,
                timeout=0.1,
                strict=True,
            )
        ])
        events = [ev async for ev in handler.run(_permission_invocation())]
        decisions = [e for e in events if isinstance(e, HookPermissionDecision)]
        assert len(decisions) == 1
        assert decisions[0].decision == "deny"
        assert decisions[0].hook_name == "slow"

    @pytest.mark.asyncio
    async def test_payload_contains_tool_and_permissions(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "payload.json"
        handler = HooksManager([
            _make_tool_hook(
                "capture", _capture_cmd(out), type=HookType.PERMISSION_REQUEST
            )
        ])
        _ = [ev async for ev in handler.run(_permission_invocation())]
        payload = json.loads(out.read_text())
        assert payload["hook_event_name"] == "permission_request"
        assert payload["tool_name"] == "bash"
        assert payload["tool_call_id"] == "call_1"
        assert payload["tool_input"] == {"command": "true"}
        assert payload["required_permissions"] == [
            {
                "scope": "command_pattern",
                "invocation_pattern": "true",
                "session_pattern": "true *",
                "label": "true",
            }
        ]


class TestPermissionRequestAgentLoopIntegration:
    def _tool_call_backend(self) -> FakeBackend:
        tool_call = ToolCall(
            id="call_pr",
            index=0,
            function=FunctionCall(name="bash", arguments='{"command":"true"}'),
        )
        return FakeBackend([
            [mock_llm_chunk(content="Running.", tool_calls=[tool_call])],
            [mock_llm_chunk(content="Done.")],
        ])

    def _approval_loop(self, hooks: list[HookConfig]) -> Any:
        return build_test_agent_loop(
            config=build_test_vibe_config(enabled_tools=["bash"]),
            backend=self._tool_call_backend(),
            hook_config_result=HookConfigResult(hooks=hooks, issues=[]),
        )

    def _recording_callback(
        self, calls: list[str], response: ApprovalResponse = ApprovalResponse.YES
    ) -> Any:
        async def approval_callback(
            tool_name: str, args: Any, tool_call_id: str, required_permissions: Any
        ) -> tuple[ApprovalResponse, str | None]:
            calls.append(tool_name)
            return response, None

        return approval_callback

    @pytest.mark.asyncio
    async def test_hook_allow_executes_without_prompt_or_notification(
        self, tmp_path: Path
    ) -> None:
        notify_out = tmp_path / "notification.json"
        hooks = [
            _make_tool_hook(
                "policy", _allow_cmd(), type=HookType.PERMISSION_REQUEST
            ),
            _make_hook(
                name="notify",
                command=_capture_cmd(notify_out),
                type=HookType.NOTIFICATION,
            ),
        ]
        agent_loop = self._approval_loop(hooks)

        events, prompted = await _drive_with_approval(agent_loop, "run true")

        tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
        assert len(tool_results) == 1
        assert tool_results[0].skipped is False
        assert prompted == []  # prompt never shown
        assert not notify_out.exists()  # notification hook never fired

    @pytest.mark.asyncio
    async def test_hook_deny_skips_tool_with_model_visible_reason(
        self, tmp_path: Path
    ) -> None:
        notify_out = tmp_path / "notification.json"
        hooks = [
            _make_tool_hook(
                "policy",
                _deny_cmd("policy: no bash"),
                type=HookType.PERMISSION_REQUEST,
            ),
            _make_hook(
                name="notify",
                command=_capture_cmd(notify_out),
                type=HookType.NOTIFICATION,
            ),
        ]
        agent_loop = self._approval_loop(hooks)

        events, prompted = await _drive_with_approval(agent_loop, "run true")

        tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
        assert len(tool_results) == 1
        assert tool_results[0].skipped is True
        assert "policy: no bash" in (tool_results[0].skip_reason or "")
        assert prompted == []  # prompt never shown
        assert not notify_out.exists()
        # The reason reaches the model as the tool response.
        assert any(
            "policy: no bash" in (m.content or "") for m in agent_loop.messages
        )

    @pytest.mark.asyncio
    async def test_hook_ask_falls_through_to_prompt_and_notification(
        self, tmp_path: Path
    ) -> None:
        notify_out = tmp_path / "notification.json"
        hooks = [
            _make_tool_hook("obs", _ask_cmd(), type=HookType.PERMISSION_REQUEST),
            _make_hook(
                name="notify",
                command=_capture_cmd(notify_out),
                type=HookType.NOTIFICATION,
            ),
        ]
        agent_loop = self._approval_loop(hooks)

        events, prompted = await _drive_with_approval(agent_loop, "run true")

        tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
        assert len(tool_results) == 1
        assert tool_results[0].skipped is False
        assert prompted == ["bash"]  # normal prompt flow
        assert notify_out.exists()  # notification hook still fired

    @pytest.mark.asyncio
    async def test_headless_hook_allow_executes_tool(self) -> None:
        # Deliberate divergence from Claude Code: the hook fires even with
        # no approval_callback, so policy hooks can auto-allow headless.
        hooks = [
            _make_tool_hook("policy", _allow_cmd(), type=HookType.PERMISSION_REQUEST)
        ]
        agent_loop = self._approval_loop(hooks)

        events = [ev async for ev in agent_loop.act("run true")]

        tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
        assert len(tool_results) == 1
        assert tool_results[0].skipped is False
        assert agent_loop.stats.tool_calls_succeeded == 1

    @pytest.mark.asyncio
    async def test_headless_passthrough_keeps_skip(self, tmp_path: Path) -> None:
        # Passthrough (hook fires but declines to decide) preserves the
        # headless SKIP; the payload capture proves the hook did fire.
        out = tmp_path / "payload.json"
        hooks = [
            _make_tool_hook(
                "capture", _capture_cmd(out), type=HookType.PERMISSION_REQUEST
            )
        ]
        agent_loop = self._approval_loop(hooks)

        # No approver available: the passthrough hook fires, then the broker
        # request is declined (as a headless/non-interactive host would).
        events, prompted = await _drive_with_approval(
            agent_loop,
            "run true",
            response=ApprovalResponse.NO,
            feedback="Tool execution not permitted.",
        )

        tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
        assert len(tool_results) == 1
        assert tool_results[0].skipped is True
        assert "Tool execution not permitted." in (tool_results[0].skip_reason or "")
        assert prompted == ["bash"]  # the passthrough hook fell through to prompt
        payload = json.loads(out.read_text())
        assert payload["hook_event_name"] == "permission_request"
        assert payload["tool_name"] == "bash"
        # Serialized from the validated args model (like before_tool's
        # tool_input), so defaulted fields appear too.
        assert payload["tool_input"]["command"] == "true"
        assert isinstance(payload["required_permissions"], list)

    @pytest.mark.asyncio
    async def test_headless_without_hooks_keeps_skip(self) -> None:
        agent_loop = self._approval_loop([])

        # No hooks and no approver: the broker request is declined, as a
        # headless/non-interactive host would.
        events, _prompted = await _drive_with_approval(
            agent_loop,
            "run true",
            response=ApprovalResponse.NO,
            feedback="Tool execution not permitted.",
        )

        tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
        assert len(tool_results) == 1
        assert tool_results[0].skipped is True
        assert "Tool execution not permitted." in (tool_results[0].skip_reason or "")

    @pytest.mark.asyncio
    async def test_hook_deny_matches_only_named_tool(self, tmp_path: Path) -> None:
        # A deny hook matched to another tool must not affect this call.
        hooks = [
            _make_tool_hook(
                "other-guard",
                _deny_cmd("wrong tool"),
                type=HookType.PERMISSION_REQUEST,
                match="write_file",
            )
        ]
        agent_loop = self._approval_loop(hooks)

        events, prompted = await _drive_with_approval(agent_loop, "run true")

        tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
        assert len(tool_results) == 1
        assert tool_results[0].skipped is False
        assert prompted == ["bash"]


# ---------------------------------------------------------------------------
# worktree_create (Wave D)
# ---------------------------------------------------------------------------


def _make_prepared_worktree(tmp_path: Path, *, created: bool = True) -> Any:
    from vibe.core.worktree import PreparedWorktree

    return PreparedWorktree(
        name="feature-x",
        branch="feature-x",
        root=tmp_path / "worktrees" / "feature-x",
        path=tmp_path / "worktrees" / "feature-x",
        repo_root=tmp_path / "repo",
        base_commit="deadbeef",
        created=created,
        branch_created=created,
    )


@pytest.fixture
def worktree_module() -> Any:
    """The vibe.core.worktree module with the create announcement drained
    before and after the test (module-level state must not leak across
    tests).
    """
    from vibe.core import worktree

    worktree.consume_worktree_create_announcement()
    yield worktree
    worktree.consume_worktree_create_announcement()


class TestWorktreeCreateHookConfig:
    def test_worktree_create_loads_from_toml(
        self, config_dir: Path
    ) -> None:
        _write_hooks_toml(
            config_dir / "hooks.toml",
            [{"name": "wc", "type": "worktree_create", "command": "true"}],
        )
        result = load_hooks_from_fs()
        assert [h.type for h in result.hooks] == [HookType.WORKTREE_CREATE]
        assert result.issues == []

    def test_match_rejected_on_worktree_create(self) -> None:
        with pytest.raises(ValueError, match="match is only valid for tool hooks"):
            HookConfig(
                name="x", type=HookType.WORKTREE_CREATE, command="echo ok", match="*"
            )

    def test_strict_rejected_on_worktree_create(self) -> None:
        with pytest.raises(ValueError, match="strict is only valid for tool hooks"):
            HookConfig(
                name="x", type=HookType.WORKTREE_CREATE, command="echo ok", strict=True
            )


class TestWorktreeCreateHook:
    def _invocation(self) -> WorktreeCreateInvocation:
        return WorktreeCreateInvocation(
            session_id="sess",
            transcript_path="",
            cwd=str(Path.cwd()),
            branch_name="feature-x",
            worktree_path="/tmp/worktrees/feature-x",
            existing=False,
        )

    @pytest.mark.asyncio
    async def test_hook_receives_payload(self, tmp_path: Path) -> None:
        out = tmp_path / "worktree_create.json"
        handler = HooksManager([
            _make_hook(
                name="wc",
                command=_capture_cmd(out),
                type=HookType.WORKTREE_CREATE,
            )
        ])
        _ = [ev async for ev in handler.run(self._invocation())]
        payload = json.loads(out.read_text())
        assert payload["hook_event_name"] == "worktree_create"
        assert payload["branch_name"] == "feature-x"
        assert payload["worktree_path"] == "/tmp/worktrees/feature-x"
        assert payload["existing"] is False

    @pytest.mark.asyncio
    async def test_deny_is_observational(self) -> None:
        # The worktree already exists by the time the hook runs; deny is
        # logged and ignored.
        handler = HooksManager([
            _make_hook(
                name="wc", command=_deny_cmd("no"), type=HookType.WORKTREE_CREATE
            )
        ])
        events = [ev async for ev in handler.run(self._invocation())]
        warnings = [
            e
            for e in events
            if isinstance(e, HookEndEvent) and e.status == HookMessageSeverity.WARNING
        ]
        assert len(warnings) == 1
        assert warnings[0].content and "ignored" in warnings[0].content
        assert not any(isinstance(e, HookUserMessage) for e in events)


class TestWorktreeCreateAgentLoopIntegration:
    def _hooks(self, command: str) -> HookConfigResult:
        return HookConfigResult(
            hooks=[
                _make_hook(
                    name="wc", command=command, type=HookType.WORKTREE_CREATE
                )
            ],
            issues=[],
        )

    @pytest.mark.asyncio
    async def test_announced_worktree_fires_on_first_prompt(
        self, tmp_path: Path, worktree_module: Any
    ) -> None:
        worktree_module.announce_worktree_session(_make_prepared_worktree(tmp_path))
        out = tmp_path / "worktree_create.json"
        agent_loop = build_test_agent_loop(
            backend=FakeBackend(mock_llm_chunk(content="hi")),
            hook_config_result=self._hooks(_capture_cmd(out)),
        )
        _ = [ev async for ev in agent_loop.act("hello")]

        payload = json.loads(out.read_text())
        assert payload["hook_event_name"] == "worktree_create"
        assert payload["branch_name"] == "feature-x"
        assert payload["worktree_path"] == str(
            tmp_path / "worktrees" / "feature-x"
        )
        assert payload["existing"] is False

    @pytest.mark.asyncio
    async def test_reused_worktree_reports_existing_true(
        self, tmp_path: Path, worktree_module: Any
    ) -> None:
        worktree_module.announce_worktree_session(
            _make_prepared_worktree(tmp_path, created=False)
        )
        out = tmp_path / "worktree_create.json"
        agent_loop = build_test_agent_loop(
            backend=FakeBackend(mock_llm_chunk(content="hi")),
            hook_config_result=self._hooks(_capture_cmd(out)),
        )
        _ = [ev async for ev in agent_loop.act("hello")]

        assert json.loads(out.read_text())["existing"] is True

    @pytest.mark.asyncio
    async def test_fires_once_and_announcement_is_consumed(
        self, tmp_path: Path, worktree_module: Any
    ) -> None:
        worktree_module.announce_worktree_session(_make_prepared_worktree(tmp_path))
        log = tmp_path / "fires.log"
        backend = FakeBackend([
            [mock_llm_chunk(content="one")],
            [mock_llm_chunk(content="two")],
        ])
        agent_loop = build_test_agent_loop(
            backend=backend, hook_config_result=self._hooks(_append_cmd(log, "wc"))
        )
        _ = [ev async for ev in agent_loop.act("first")]
        _ = [ev async for ev in agent_loop.act("second")]

        # A second loop in the same process must not re-fire: the first
        # loop consumed the announcement at construction time.
        second_loop = build_test_agent_loop(
            backend=FakeBackend(mock_llm_chunk(content="three")),
            hook_config_result=self._hooks(_append_cmd(log, "wc")),
        )
        _ = [ev async for ev in second_loop.act("third")]

        assert log.read_text().splitlines() == ["wc"]

    @pytest.mark.asyncio
    async def test_fires_before_session_start(
        self, tmp_path: Path, worktree_module: Any
    ) -> None:
        worktree_module.announce_worktree_session(_make_prepared_worktree(tmp_path))
        log = tmp_path / "order.log"
        hooks = [
            _make_hook(
                name="wc",
                command=_append_event_name_cmd(log),
                type=HookType.WORKTREE_CREATE,
            ),
            _make_hook(
                name="ss",
                command=_append_event_name_cmd(log),
                type=HookType.SESSION_START,
            ),
        ]
        agent_loop = build_test_agent_loop(
            backend=FakeBackend(mock_llm_chunk(content="hi")),
            hook_config_result=HookConfigResult(hooks=hooks, issues=[]),
        )
        _ = [ev async for ev in agent_loop.act("hello")]

        assert log.read_text().splitlines() == ["worktree_create", "session_start"]

    @pytest.mark.asyncio
    async def test_subagent_loop_does_not_consume_announcement(
        self, tmp_path: Path, worktree_module: Any
    ) -> None:
        worktree_module.announce_worktree_session(_make_prepared_worktree(tmp_path))
        log = tmp_path / "fires.log"

        subagent_loop = build_test_agent_loop(
            backend=FakeBackend(mock_llm_chunk(content="child")),
            hook_config_result=self._hooks(_append_cmd(log, "sub")),
            is_subagent=True,
        )
        _ = [ev async for ev in subagent_loop.act("child work")]
        assert not log.exists()

        main_loop = build_test_agent_loop(
            backend=FakeBackend(mock_llm_chunk(content="main")),
            hook_config_result=self._hooks(_append_cmd(log, "main")),
        )
        _ = [ev async for ev in main_loop.act("main work")]
        assert log.read_text().splitlines() == ["main"]

    @pytest.mark.asyncio
    async def test_no_announcement_no_fire(
        self, tmp_path: Path, worktree_module: Any
    ) -> None:
        out = tmp_path / "worktree_create.json"
        agent_loop = build_test_agent_loop(
            backend=FakeBackend(mock_llm_chunk(content="hi")),
            hook_config_result=self._hooks(_capture_cmd(out)),
        )
        _ = [ev async for ev in agent_loop.act("hello")]
        assert not out.exists()


# ---------------------------------------------------------------------------
# Skill-invocation tool-hook matching (Wave D — documentation tests)
# ---------------------------------------------------------------------------


class TestSkillInvocationHookMatching:
    """Pin down the matcher pattern documented in the README hooks
    section: skills load through the built-in ``skill`` tool, so
    MODEL-invoked loads run the normal tool pipeline and tool hooks with
    ``match = "skill"`` fire (skill name in ``tool_input.name``). A
    USER-typed ``/skill`` command instead injects a synthetic ``skill``
    tool call straight into the transcript (v2.19.1) — the tool pipeline
    never runs, so tool hooks do NOT fire on that path.
    """

    def _install_skill(self, agent_loop: Any, name: str = "myskill") -> None:
        from types import MappingProxyType

        from vibe.core.skills.models import SkillInfo

        info = SkillInfo(name=name, description="test skill", prompt="Do the thing.")
        agent_loop.skill_manager.available_skills = MappingProxyType({name: info})

    @pytest.mark.asyncio
    async def test_tool_hooks_match_skill_on_model_invoked_load(
        self, tmp_path: Path
    ) -> None:
        before_out = tmp_path / "before_skill.json"
        after_out = tmp_path / "after_skill.json"
        tool_call = ToolCall(
            id="call_skill",
            index=0,
            function=FunctionCall(name="skill", arguments='{"name": "myskill"}'),
        )
        backend = FakeBackend([
            [mock_llm_chunk(content="Loading.", tool_calls=[tool_call])],
            [mock_llm_chunk(content="done")],
        ])
        hooks = [
            _make_tool_hook(
                "before-skill",
                _capture_cmd(before_out),
                type=HookType.PRE_TOOL,
                match="skill",
            ),
            _make_tool_hook(
                "after-skill",
                _capture_cmd(after_out),
                type=HookType.POST_TOOL,
                match="skill",
            ),
        ]
        agent_loop = build_test_agent_loop(
            config=build_test_vibe_config(enabled_tools=["skill"]),
            agent_name=BuiltinAgentName.AUTO_APPROVE,
            backend=backend,
            hook_config_result=HookConfigResult(hooks=hooks, issues=[]),
        )
        self._install_skill(agent_loop)

        events = [ev async for ev in agent_loop.act("load the skill")]

        before = json.loads(before_out.read_text())
        assert before["hook_event_name"] == "pre_tool"
        assert before["tool_name"] == "skill"
        assert before["tool_input"] == {"name": "myskill"}

        after = json.loads(after_out.read_text())
        assert after["tool_name"] == "skill"
        assert after["tool_status"] == "success"

        tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
        assert len(tool_results) == 1
        assert tool_results[0].skipped is False

    @pytest.mark.asyncio
    async def test_before_tool_deny_blocks_model_invoked_skill_load(
        self, tmp_path: Path
    ) -> None:
        tool_call = ToolCall(
            id="call_skill",
            index=0,
            function=FunctionCall(name="skill", arguments='{"name": "myskill"}'),
        )
        backend = FakeBackend([
            [mock_llm_chunk(content="Loading.", tool_calls=[tool_call])],
            [mock_llm_chunk(content="ok then")],
        ])
        hooks = [
            _make_tool_hook(
                "no-skills",
                _deny_cmd("skills are locked down"),
                type=HookType.PRE_TOOL,
                match="skill",
            )
        ]
        agent_loop = build_test_agent_loop(
            config=build_test_vibe_config(enabled_tools=["skill"]),
            agent_name=BuiltinAgentName.AUTO_APPROVE,
            backend=backend,
            hook_config_result=HookConfigResult(hooks=hooks, issues=[]),
        )
        self._install_skill(agent_loop)

        events = [ev async for ev in agent_loop.act("load the skill")]

        tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
        assert len(tool_results) == 1
        assert tool_results[0].skipped is True
        assert "skills are locked down" in (tool_results[0].skip_reason or "")

    @pytest.mark.asyncio
    async def test_user_invoked_skill_command_bypasses_tool_hooks(
        self, tmp_path: Path
    ) -> None:
        # /myskill is expanded by injecting a synthetic `skill` tool call
        # into the transcript; the tool pipeline never runs, so even a
        # match-everything before_tool hook must not fire.
        out = tmp_path / "before.json"
        backend = FakeBackend(mock_llm_chunk(content="ok"))
        hooks = [
            _make_tool_hook(
                "watch-all", _capture_cmd(out), type=HookType.PRE_TOOL, match="*"
            )
        ]
        agent_loop = build_test_agent_loop(
            backend=backend,
            hook_config_result=HookConfigResult(hooks=hooks, issues=[]),
        )
        self._install_skill(agent_loop)

        _ = [ev async for ev in agent_loop.act("/myskill")]

        # The synthetic skill tool call did land in the transcript ...
        synthetic_calls = [
            tc
            for m in agent_loop.messages
            if m.tool_calls
            for tc in m.tool_calls
            if tc.function.name == "skill"
        ]
        assert len(synthetic_calls) == 1
        # ... but no tool executed, so no tool hook fired.
        assert not out.exists()
