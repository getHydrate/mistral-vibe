from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
import tomli_w

from tests.conftest import build_test_agent_loop
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend
from vibe.core.config import VibeConfig
from vibe.core.hooks.config import (
    HookConfig,
    HookConfigResult,
    _load_hooks_file,
    load_hooks_from_fs,
)
from vibe.core.hooks.executor import HookExecutor
from vibe.core.hooks.manager import HooksManager, _parse_user_prompt_submit_result
from vibe.core.hooks.models import (
    HookDenied,
    HookEndEvent,
    HookInjectedContext,
    HookInvocation,
    HookMessageSeverity,
    HookStartEvent,
    HookType,
    HookUserMessage,
)
from vibe.core.types import BaseEvent


@pytest.fixture
def sample_invocation() -> HookInvocation:
    return HookInvocation(
        session_id="test-session",
        transcript_path="",
        cwd=str(Path.cwd()),
        hook_event_name="post_agent_turn",
    )


@pytest.fixture
def config_hooks_disabled() -> VibeConfig:
    return VibeConfig(enable_experimental_hooks=False)


@pytest.fixture
def config_hooks_enabled() -> VibeConfig:
    return VibeConfig(enable_experimental_hooks=True)


def _write_hooks_toml(path: Path, hooks: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        tomli_w.dump({"hooks": hooks}, f)


def _make_hook(
    name: str = "test-hook", command: str = "echo ok", timeout: float = 30.0
) -> HookConfig:
    return HookConfig(
        name=name, type=HookType.POST_AGENT_TURN, command=command, timeout=timeout
    )


class TestConfigLoading:
    def test_load_from_global_file(
        self, config_dir: Path, config_hooks_enabled: VibeConfig
    ) -> None:
        _write_hooks_toml(
            config_dir / "hooks.toml",
            [
                {
                    "name": "lint",
                    "type": HookType.POST_AGENT_TURN,
                    "command": "echo lint",
                }
            ],
        )
        result = load_hooks_from_fs(config_hooks_enabled)
        assert len(result.hooks) == 1
        assert result.hooks[0].name == "lint"
        assert result.issues == []

    def test_load_from_both_global_and_project(
        self,
        config_dir: Path,
        tmp_working_directory: Path,
        config_hooks_enabled: VibeConfig,
    ) -> None:
        _write_hooks_toml(
            config_dir / "hooks.toml",
            [
                {
                    "name": "global-hook",
                    "type": "post_agent_turn",
                    "command": "echo global",
                }
            ],
        )
        project_vibe = tmp_working_directory / ".vibe"
        _write_hooks_toml(
            project_vibe / "hooks.toml",
            [
                {
                    "name": "project-hook",
                    "type": "post_agent_turn",
                    "command": "echo project",
                }
            ],
        )
        from vibe.core.trusted_folders import trusted_folders_manager

        trusted_folders_manager.add_trusted(tmp_working_directory)

        result = load_hooks_from_fs(config_hooks_enabled)
        assert len(result.hooks) == 2
        names = {h.name for h in result.hooks}
        assert names == {"global-hook", "project-hook"}

    def test_project_file_skipped_when_untrusted(
        self, tmp_working_directory: Path, config_hooks_enabled: VibeConfig
    ) -> None:
        project_vibe = tmp_working_directory / ".vibe"
        _write_hooks_toml(
            project_vibe / "hooks.toml",
            [
                {
                    "name": "sneaky-hook",
                    "type": "post_agent_turn",
                    "command": "echo sneaky",
                }
            ],
        )
        result = load_hooks_from_fs(config_hooks_enabled)
        assert not any(h.name == "sneaky-hook" for h in result.hooks)

    def test_duplicate_hook_name_detection(
        self,
        config_dir: Path,
        tmp_working_directory: Path,
        config_hooks_enabled: VibeConfig,
    ) -> None:
        _write_hooks_toml(
            config_dir / "hooks.toml",
            [{"name": "dup-hook", "type": "post_agent_turn", "command": "echo global"}],
        )
        project_vibe = tmp_working_directory / ".vibe"
        _write_hooks_toml(
            project_vibe / "hooks.toml",
            [
                {
                    "name": "dup-hook",
                    "type": "post_agent_turn",
                    "command": "echo project",
                }
            ],
        )
        from vibe.core.trusted_folders import trusted_folders_manager

        trusted_folders_manager.add_trusted(tmp_working_directory)

        result = load_hooks_from_fs(config_hooks_enabled)
        assert len(result.hooks) == 1
        assert any("Duplicate" in i.message for i in result.issues)

    def test_toml_parse_error_reported(
        self, config_dir: Path, config_hooks_enabled: VibeConfig
    ) -> None:
        hooks_file = config_dir / "hooks.toml"
        hooks_file.write_text("this is not valid toml [[[", encoding="utf-8")
        result = load_hooks_from_fs(config_hooks_enabled)
        assert result.hooks == []
        assert len(result.issues) == 1
        assert (
            "parse" in result.issues[0].message.lower()
            or "Failed" in result.issues[0].message
        )

    def test_validation_error_reported(
        self, config_dir: Path, config_hooks_enabled: VibeConfig
    ) -> None:
        _write_hooks_toml(
            config_dir / "hooks.toml",
            [{"name": "bad", "type": "InvalidType", "command": "echo"}],
        )
        result = load_hooks_from_fs(config_hooks_enabled)
        assert result.hooks == []
        assert len(result.issues) == 1

    def test_missing_command_reported(
        self, config_dir: Path, config_hooks_enabled: VibeConfig
    ) -> None:
        _write_hooks_toml(
            config_dir / "hooks.toml", [{"name": "no-cmd", "type": "post_agent_turn"}]
        )
        result = load_hooks_from_fs(config_hooks_enabled)
        assert result.hooks == []
        assert len(result.issues) == 1

    def test_empty_command_reported(
        self, config_dir: Path, config_hooks_enabled: VibeConfig
    ) -> None:
        _write_hooks_toml(
            config_dir / "hooks.toml",
            [{"name": "empty-cmd", "type": "post_agent_turn", "command": "   "}],
        )
        result = load_hooks_from_fs(config_hooks_enabled)
        assert result.hooks == []
        assert len(result.issues) == 1

    def test_default_timeout(
        self, config_dir: Path, config_hooks_enabled: VibeConfig
    ) -> None:
        _write_hooks_toml(
            config_dir / "hooks.toml",
            [{"name": "h", "type": "post_agent_turn", "command": "echo ok"}],
        )
        result = load_hooks_from_fs(config_hooks_enabled)
        assert result.hooks[0].timeout == 30.0

    def test_nonexistent_file_returns_empty(self, tmp_path: Path) -> None:
        result = _load_hooks_file(tmp_path / "missing.toml")
        assert result.hooks == []
        assert result.issues == []

    def test_hooks_disabled_returns_empty(
        self, config_dir: Path, config_hooks_disabled: VibeConfig
    ) -> None:
        _write_hooks_toml(
            config_dir / "hooks.toml",
            [
                {
                    "name": "lint",
                    "type": HookType.POST_AGENT_TURN,
                    "command": "echo lint",
                }
            ],
        )
        result = load_hooks_from_fs(config_hooks_disabled)
        assert result.hooks == []
        assert result.issues == []


class TestHookExecutor:
    @pytest.mark.asyncio
    async def test_exit_0_success(self, sample_invocation: HookInvocation) -> None:
        hook = _make_hook(command="echo success")
        result = await HookExecutor().run(hook, sample_invocation)
        assert result.exit_code == 0
        assert result.stdout == "success"
        assert not result.timed_out

    @pytest.mark.asyncio
    async def test_exit_2_retry(self, sample_invocation: HookInvocation) -> None:
        hook = _make_hook(command="echo 'fix this'; exit 2")
        result = await HookExecutor().run(hook, sample_invocation)
        assert result.exit_code == 2
        assert "fix this" in result.stdout
        assert not result.timed_out

    @pytest.mark.asyncio
    async def test_other_exit_code(self, sample_invocation: HookInvocation) -> None:
        hook = _make_hook(command="echo 'oops'; exit 1")
        result = await HookExecutor().run(hook, sample_invocation)
        assert result.exit_code == 1
        assert "oops" in result.stdout

    @pytest.mark.asyncio
    async def test_timeout(self, sample_invocation: HookInvocation) -> None:
        hook = _make_hook(command="sleep 60", timeout=0.5)
        result = await HookExecutor().run(hook, sample_invocation)
        assert result.timed_out
        assert result.exit_code is None

    @pytest.mark.asyncio
    async def test_stderr_captured_separately(
        self, sample_invocation: HookInvocation
    ) -> None:
        hook = _make_hook(command="echo out; echo err >&2")
        result = await HookExecutor().run(hook, sample_invocation)
        assert result.exit_code == 0
        assert result.stdout == "out"
        assert result.stderr == "err"

    @pytest.mark.asyncio
    async def test_stdin_json_received(self, sample_invocation: HookInvocation) -> None:
        hook = _make_hook(
            command=f"{sys.executable} -c \"import sys,json; d=json.load(sys.stdin); print(d['session_id'])\""
        )
        result = await HookExecutor().run(hook, sample_invocation)
        assert result.exit_code == 0
        assert result.stdout == "test-session"


class TestHooksManager:
    @pytest.mark.asyncio
    async def test_exit_0_emits_start_and_end(self) -> None:
        handler = HooksManager([_make_hook(command="echo ok")])
        from vibe.core.config import SessionLoggingConfig
        from vibe.core.session.session_logger import SessionLogger

        logger = SessionLogger(SessionLoggingConfig(enabled=False), "test-id")
        events: list[BaseEvent | HookUserMessage] = []
        async for ev in handler.run(HookType.POST_AGENT_TURN, "sess", logger):
            events.append(ev)

        event_types = [type(e).__name__ for e in events]
        assert "HookStartEvent" in event_types
        assert "HookEndEvent" in event_types
        # Exit 0 with no retry = no HookUserMessage
        assert not any(isinstance(e, HookUserMessage) for e in events)

    @pytest.mark.asyncio
    async def test_exit_2_emits_retry_message(self) -> None:
        handler = HooksManager([_make_hook(command="echo 'fix it'; exit 2")])
        from vibe.core.config import SessionLoggingConfig
        from vibe.core.session.session_logger import SessionLogger

        logger = SessionLogger(SessionLoggingConfig(enabled=False), "test-id")
        events: list[BaseEvent | HookUserMessage] = []
        async for ev in handler.run(HookType.POST_AGENT_TURN, "sess", logger):
            events.append(ev)

        retry_msgs = [e for e in events if isinstance(e, HookUserMessage)]
        assert len(retry_msgs) == 1
        assert "fix it" in retry_msgs[0].content

        # Display message should be generic, not the stdout
        end_msgs = [
            e for e in events if isinstance(e, HookEndEvent) and e.content is not None
        ]
        assert any("retrying" in m.content.lower() for m in end_msgs if m.content)
        assert not any("fix it" in (m.content or "") for m in end_msgs)

    @pytest.mark.asyncio
    async def test_exit_2_without_output_emits_warning(self) -> None:
        handler = HooksManager([_make_hook(command="exit 2")])
        from vibe.core.config import SessionLoggingConfig
        from vibe.core.session.session_logger import SessionLogger

        logger = SessionLogger(SessionLoggingConfig(enabled=False), "test-id")
        events: list[BaseEvent | HookUserMessage] = []
        async for ev in handler.run(HookType.POST_AGENT_TURN, "sess", logger):
            events.append(ev)

        end_msgs = [e for e in events if isinstance(e, HookEndEvent)]
        assert len(end_msgs) == 1
        assert end_msgs[0].content == "Exited with code 2"

    @pytest.mark.asyncio
    async def test_max_retry_limit(self) -> None:
        handler = HooksManager([_make_hook(command="echo retry; exit 2")])
        from vibe.core.config import SessionLoggingConfig
        from vibe.core.session.session_logger import SessionLogger

        logger = SessionLogger(SessionLoggingConfig(enabled=False), "test-id")

        # Run 3 times (should get retry each time)
        for _ in range(3):
            events = [
                ev async for ev in handler.run(HookType.POST_AGENT_TURN, "sess", logger)
            ]
            assert any(isinstance(e, HookUserMessage) for e in events)

        # 4th time: max exceeded, no retry
        events = [
            ev async for ev in handler.run(HookType.POST_AGENT_TURN, "sess", logger)
        ]
        assert not any(isinstance(e, HookUserMessage) for e in events)
        # Should have error message about max retries
        error_events = [
            e
            for e in events
            if isinstance(e, HookEndEvent)
            and e.content
            and "exhausted" in e.content.lower()
        ]
        assert len(error_events) == 1

    @pytest.mark.asyncio
    async def test_warning_on_nonzero_exit(self) -> None:
        handler = HooksManager([_make_hook(command="echo warn; exit 1")])
        from vibe.core.config import SessionLoggingConfig
        from vibe.core.session.session_logger import SessionLogger

        logger = SessionLogger(SessionLoggingConfig(enabled=False), "test-id")
        events = [
            ev async for ev in handler.run(HookType.POST_AGENT_TURN, "sess", logger)
        ]

        warnings = [
            e
            for e in events
            if isinstance(e, HookEndEvent) and e.status == HookMessageSeverity.WARNING
        ]
        assert len(warnings) == 1
        assert warnings[0].content and "warn" in warnings[0].content

    @pytest.mark.asyncio
    async def test_warning_falls_back_to_stderr(self) -> None:
        hook = _make_hook(command="echo problem >&2; exit 1")
        handler = HooksManager([hook])
        from vibe.core.config import SessionLoggingConfig
        from vibe.core.session.session_logger import SessionLogger

        logger = SessionLogger(SessionLoggingConfig(enabled=False), "test-id")
        events = [
            ev async for ev in handler.run(HookType.POST_AGENT_TURN, "sess", logger)
        ]

        warnings = [
            e
            for e in events
            if isinstance(e, HookEndEvent) and e.status == HookMessageSeverity.WARNING
        ]
        assert len(warnings) == 1
        assert warnings[0].content and "problem" in warnings[0].content

    @pytest.mark.asyncio
    async def test_timeout_emits_warning(self) -> None:
        handler = HooksManager([_make_hook(command="sleep 60", timeout=0.5)])
        from vibe.core.config import SessionLoggingConfig
        from vibe.core.session.session_logger import SessionLogger

        logger = SessionLogger(SessionLoggingConfig(enabled=False), "test-id")
        events = [
            ev async for ev in handler.run(HookType.POST_AGENT_TURN, "sess", logger)
        ]

        warnings = [
            e
            for e in events
            if isinstance(e, HookEndEvent) and e.status == HookMessageSeverity.WARNING
        ]
        assert len(warnings) == 1
        assert warnings[0].content and "Timed out" in warnings[0].content


class TestAgentLoopIntegration:
    @pytest.mark.asyncio
    async def test_hooks_run_after_turn(self) -> None:
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
    async def test_hook_retry_reinjects_message(self) -> None:
        # First call: LLM responds. Hook requests retry with "fix this".
        # Second call (after retry injection): LLM responds again. Hook exits 0.
        backend = FakeBackend([
            [mock_llm_chunk(content="first response")],
            [mock_llm_chunk(content="second response")],
        ])

        # Create a script that exits 2 on first call, 0 on subsequent
        counter_file = Path.cwd() / ".hook_counter"
        script = (
            f'{sys.executable} -c "'
            f"from pathlib import Path; "
            f"p = Path({str(counter_file)!r}); "
            f"c = int(p.read_text()) if p.exists() else 0; "
            f"p.write_text(str(c + 1)); "
            f"import sys; "
            f"print('fix this'); "
            f"sys.exit(2 if c == 0 else 0)"
            f'"'
        )
        hooks = [_make_hook(name="retry-hook", command=script)]
        agent_loop = build_test_agent_loop(
            backend=backend, hook_config_result=HookConfigResult(hooks=hooks, issues=[])
        )

        events = [ev async for ev in agent_loop.act("hi")]

        # Should have two assistant events (two LLM turns)
        from vibe.core.types import AssistantEvent

        assistant_events = [e for e in events if isinstance(e, AssistantEvent)]
        assert len(assistant_events) == 2

        # Check that a retry user message was injected
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


class TestUserPromptSubmitResultParser:
    def test_empty_stdout_returns_none(self) -> None:
        assert _parse_user_prompt_submit_result("") is None

    def test_plain_text_returns_injected_context(self) -> None:
        result = _parse_user_prompt_submit_result("some context")
        assert isinstance(result, HookInjectedContext)
        assert result.content == "some context"

    def test_json_decision_inject(self) -> None:
        import json

        payload = json.dumps({"decision": "inject", "additional_context": "extra info"})
        result = _parse_user_prompt_submit_result(payload)
        assert isinstance(result, HookInjectedContext)
        assert result.content == "extra info"

    def test_json_decision_allow_no_context(self) -> None:
        import json

        payload = json.dumps({"decision": "allow"})
        result = _parse_user_prompt_submit_result(payload)
        assert result is None

    def test_json_decision_allow_with_context(self) -> None:
        import json

        payload = json.dumps({"decision": "allow", "additional_context": "ctx"})
        result = _parse_user_prompt_submit_result(payload)
        assert isinstance(result, HookInjectedContext)
        assert result.content == "ctx"

    def test_json_decision_deny(self) -> None:
        import json

        payload = json.dumps({"decision": "deny", "reason": "not allowed"})
        result = _parse_user_prompt_submit_result(payload)
        assert isinstance(result, HookDenied)
        assert result.reason == "not allowed"

    def test_json_decision_deny_no_reason(self) -> None:
        import json

        payload = json.dumps({"decision": "deny"})
        result = _parse_user_prompt_submit_result(payload)
        assert isinstance(result, HookDenied)
        assert result.reason is None

    def test_claude_style_additional_context(self) -> None:
        import json

        payload = json.dumps({
            "hookSpecificOutput": {
                "hookEventName": "UserPromptSubmit",
                "additionalContext": "injected context here",
            }
        })
        result = _parse_user_prompt_submit_result(payload)
        assert isinstance(result, HookInjectedContext)
        assert result.content == "injected context here"

    def test_claude_style_missing_additional_context(self) -> None:
        import json

        payload = json.dumps({
            "hookSpecificOutput": {"hookEventName": "UserPromptSubmit"}
        })
        result = _parse_user_prompt_submit_result(payload)
        assert result is None

    def test_invalid_json_treated_as_plain_text(self) -> None:
        result = _parse_user_prompt_submit_result("not json {")
        assert isinstance(result, HookInjectedContext)
        assert result.content == "not json {"


class TestHookTypeConfig:
    def test_user_prompt_submit_loads_from_toml(
        self, config_dir: Path, config_hooks_enabled: VibeConfig
    ) -> None:
        _write_hooks_toml(
            config_dir / "hooks.toml",
            [
                {
                    "name": "ctx-hook",
                    "type": "user_prompt_submit",
                    "command": "echo context",
                }
            ],
        )
        result = load_hooks_from_fs(config_hooks_enabled)
        assert len(result.hooks) == 1
        assert result.hooks[0].type == HookType.USER_PROMPT_SUBMIT
        assert result.issues == []

    def test_invalid_hook_type_reported(
        self, config_dir: Path, config_hooks_enabled: VibeConfig
    ) -> None:
        _write_hooks_toml(
            config_dir / "hooks.toml",
            [{"name": "bad", "type": "nonexistent_type", "command": "echo"}],
        )
        result = load_hooks_from_fs(config_hooks_enabled)
        assert result.hooks == []
        assert len(result.issues) == 1

    def test_both_hook_types_load(
        self, config_dir: Path, config_hooks_enabled: VibeConfig
    ) -> None:
        _write_hooks_toml(
            config_dir / "hooks.toml",
            [
                {"name": "h1", "type": "post_agent_turn", "command": "echo a"},
                {"name": "h2", "type": "user_prompt_submit", "command": "echo b"},
            ],
        )
        result = load_hooks_from_fs(config_hooks_enabled)
        assert len(result.hooks) == 2
        types = {h.type for h in result.hooks}
        assert HookType.POST_AGENT_TURN in types
        assert HookType.USER_PROMPT_SUBMIT in types


class TestHooksManagerUserPromptSubmit:
    def _make_logger(self):
        from vibe.core.config import SessionLoggingConfig
        from vibe.core.session.session_logger import SessionLogger

        return SessionLogger(SessionLoggingConfig(enabled=False), "test-id")

    def _make_ups_hook(
        self, name: str = "ctx-hook", command: str = "echo ok", timeout: float = 30.0
    ) -> HookConfig:
        return HookConfig(
            name=name, type=HookType.USER_PROMPT_SUBMIT, command=command, timeout=timeout
        )

    @pytest.mark.asyncio
    async def test_plain_stdout_yields_injected_context(self) -> None:
        handler = HooksManager([self._make_ups_hook(command="echo extra context")])
        logger = self._make_logger()
        events = [
            ev
            async for ev in handler.run_user_prompt_submit(
                "hello", "sess", logger
            )
        ]
        injected = [e for e in events if isinstance(e, HookInjectedContext)]
        assert len(injected) == 1
        assert injected[0].content == "extra context"

    @pytest.mark.asyncio
    async def test_json_inject_yields_injected_context(self) -> None:
        import json

        payload = json.dumps({"decision": "inject", "additional_context": "ctx-data"})
        handler = HooksManager([self._make_ups_hook(command=f"echo '{payload}'")])
        logger = self._make_logger()
        events = [
            ev
            async for ev in handler.run_user_prompt_submit(
                "hello", "sess", logger
            )
        ]
        injected = [e for e in events if isinstance(e, HookInjectedContext)]
        assert len(injected) == 1
        assert injected[0].content == "ctx-data"

    @pytest.mark.asyncio
    async def test_json_deny_yields_hook_denied(self) -> None:
        import json

        payload = json.dumps({"decision": "deny", "reason": "blocked"})
        handler = HooksManager([self._make_ups_hook(command=f"echo '{payload}'")])
        logger = self._make_logger()
        events = [
            ev
            async for ev in handler.run_user_prompt_submit(
                "hello", "sess", logger
            )
        ]
        denied = [e for e in events if isinstance(e, HookDenied)]
        assert len(denied) == 1
        assert denied[0].reason == "blocked"

    @pytest.mark.asyncio
    async def test_claude_style_output_yields_injected_context(self) -> None:
        import json

        payload = json.dumps({
            "hookSpecificOutput": {
                "hookEventName": "UserPromptSubmit",
                "additionalContext": "recall data",
            }
        })
        handler = HooksManager([self._make_ups_hook(command=f"echo '{payload}'")])
        logger = self._make_logger()
        events = [
            ev
            async for ev in handler.run_user_prompt_submit(
                "hello", "sess", logger
            )
        ]
        injected = [e for e in events if isinstance(e, HookInjectedContext)]
        assert len(injected) == 1
        assert injected[0].content == "recall data"

    @pytest.mark.asyncio
    async def test_timeout_fails_open(self) -> None:
        handler = HooksManager([
            self._make_ups_hook(command="sleep 60", timeout=0.5)
        ])
        logger = self._make_logger()
        events = [
            ev
            async for ev in handler.run_user_prompt_submit(
                "hello", "sess", logger
            )
        ]
        denied = [e for e in events if isinstance(e, HookDenied)]
        assert denied == []
        warnings = [
            e
            for e in events
            if isinstance(e, HookEndEvent) and e.status == HookMessageSeverity.WARNING
        ]
        assert len(warnings) == 1


class TestAgentLoopUserPromptSubmitIntegration:
    @pytest.mark.asyncio
    async def test_context_injected_before_llm_turn(self) -> None:
        backend = FakeBackend(mock_llm_chunk(content="Hi there!"))
        hooks = [
            HookConfig(
                name="ctx",
                type=HookType.USER_PROMPT_SUBMIT,
                command="echo test-context",
            )
        ]
        agent_loop = build_test_agent_loop(
            backend=backend,
            hook_config_result=HookConfigResult(hooks=hooks, issues=[]),
        )
        events = [ev async for ev in agent_loop.act("tell me something")]
        event_types = [type(e).__name__ for e in events]
        assert "HookStartEvent" in event_types
        assert "HookEndEvent" in event_types
        injected = [
            m
            for m in agent_loop.messages
            if m.role.value == "user" and m.injected
        ]
        assert any("test-context" in (m.content or "") for m in injected)

    @pytest.mark.asyncio
    async def test_deny_blocks_llm_turn(self) -> None:
        import json

        payload = json.dumps({"decision": "deny", "reason": "policy violation"})
        backend = FakeBackend(mock_llm_chunk(content="Should not be reached"))
        hooks = [
            HookConfig(
                name="guard",
                type=HookType.USER_PROMPT_SUBMIT,
                command=f"echo '{payload}'",
            )
        ]
        agent_loop = build_test_agent_loop(
            backend=backend,
            hook_config_result=HookConfigResult(hooks=hooks, issues=[]),
        )
        events = [ev async for ev in agent_loop.act("do something bad")]
        from vibe.core.types import AssistantEvent

        assistant_events = [e for e in events if isinstance(e, AssistantEvent)]
        assert len(assistant_events) == 1
        assert "policy violation" in (assistant_events[0].content or "")
        # LLM backend should not have been called
        assert len(backend.requests_messages) == 0

    @pytest.mark.asyncio
    async def test_post_agent_turn_retry_unchanged(self) -> None:
        backend = FakeBackend([
            [mock_llm_chunk(content="first")],
            [mock_llm_chunk(content="second")],
        ])
        counter_file = Path.cwd() / ".hook_counter2"
        script = (
            f'{sys.executable} -c "'
            f"from pathlib import Path; "
            f"p = Path({str(counter_file)!r}); "
            f"c = int(p.read_text()) if p.exists() else 0; "
            f"p.write_text(str(c + 1)); "
            f"import sys; "
            f"print('retry please'); "
            f"sys.exit(2 if c == 0 else 0)"
            f'"'
        )
        hooks = [
            HookConfig(
                name="post-retry",
                type=HookType.POST_AGENT_TURN,
                command=script,
            )
        ]
        agent_loop = build_test_agent_loop(
            backend=backend,
            hook_config_result=HookConfigResult(hooks=hooks, issues=[]),
        )
        events = [ev async for ev in agent_loop.act("hi")]
        from vibe.core.types import AssistantEvent

        assistant_events = [e for e in events if isinstance(e, AssistantEvent)]
        assert len(assistant_events) == 2
        injected = [
            m for m in agent_loop.messages if m.role.value == "user" and m.injected
        ]
        assert any("retry please" in (m.content or "") for m in injected)


class TestSessionStartAndPreCompactConfig:
    def test_session_start_loads_from_toml(
        self, config_dir: Path, config_hooks_enabled: VibeConfig
    ) -> None:
        _write_hooks_toml(
            config_dir / "hooks.toml",
            [{"name": "ss", "type": "session_start", "command": "echo ok"}],
        )
        result = load_hooks_from_fs(config_hooks_enabled)
        assert len(result.hooks) == 1
        assert result.hooks[0].type == HookType.SESSION_START

    def test_pre_compact_loads_from_toml(
        self, config_dir: Path, config_hooks_enabled: VibeConfig
    ) -> None:
        _write_hooks_toml(
            config_dir / "hooks.toml",
            [{"name": "pc", "type": "pre_compact", "command": "echo ok"}],
        )
        result = load_hooks_from_fs(config_hooks_enabled)
        assert len(result.hooks) == 1
        assert result.hooks[0].type == HookType.PRE_COMPACT

    def test_all_four_types_load(
        self, config_dir: Path, config_hooks_enabled: VibeConfig
    ) -> None:
        _write_hooks_toml(
            config_dir / "hooks.toml",
            [
                {"name": "a", "type": "post_agent_turn", "command": "echo a"},
                {"name": "b", "type": "user_prompt_submit", "command": "echo b"},
                {"name": "c", "type": "session_start", "command": "echo c"},
                {"name": "d", "type": "pre_compact", "command": "echo d"},
            ],
        )
        result = load_hooks_from_fs(config_hooks_enabled)
        assert len(result.hooks) == 4
        types = {h.type for h in result.hooks}
        assert types == {
            HookType.POST_AGENT_TURN,
            HookType.USER_PROMPT_SUBMIT,
            HookType.SESSION_START,
            HookType.PRE_COMPACT,
        }


class TestHooksManagerSessionStart:
    def _make_logger(self):
        from vibe.core.config import SessionLoggingConfig
        from vibe.core.session.session_logger import SessionLogger

        return SessionLogger(SessionLoggingConfig(enabled=False), "test-id")

    def _make_hook(self, command: str, name: str = "ss-hook") -> HookConfig:
        return HookConfig(
            name=name, type=HookType.SESSION_START, command=command, timeout=30.0
        )

    @pytest.mark.asyncio
    async def test_new_source_in_payload(self) -> None:
        hook = self._make_hook(
            f"{sys.executable} -c \"import sys,json; d=json.load(sys.stdin); print(d['source'])\""
        )
        handler = HooksManager([hook])
        logger = self._make_logger()
        events = [
            ev async for ev in handler.run_session_start("new", "sess", logger)
        ]
        end_events = [e for e in events if isinstance(e, HookEndEvent)]
        assert any(e.status == HookMessageSeverity.OK for e in end_events)

    @pytest.mark.asyncio
    async def test_resume_source_in_payload(self) -> None:
        hook = self._make_hook(
            f"{sys.executable} -c \"import sys,json; d=json.load(sys.stdin); assert d['source']=='resume'; print('ok')\""
        )
        handler = HooksManager([hook])
        logger = self._make_logger()
        events = [
            ev async for ev in handler.run_session_start("resume", "sess", logger)
        ]
        end_events = [e for e in events if isinstance(e, HookEndEvent)]
        assert any(e.status == HookMessageSeverity.OK for e in end_events)

    @pytest.mark.asyncio
    async def test_plain_stdout_yields_injected_context(self) -> None:
        hook = self._make_hook("echo session-context-data")
        handler = HooksManager([hook])
        logger = self._make_logger()
        events = [
            ev async for ev in handler.run_session_start("new", "sess", logger)
        ]
        injected = [e for e in events if isinstance(e, HookInjectedContext)]
        assert len(injected) == 1
        assert injected[0].content == "session-context-data"

    @pytest.mark.asyncio
    async def test_timeout_fails_open(self) -> None:
        hook = HookConfig(
            name="slow", type=HookType.SESSION_START, command="sleep 60", timeout=0.5
        )
        handler = HooksManager([hook])
        logger = self._make_logger()
        events = [
            ev async for ev in handler.run_session_start("new", "sess", logger)
        ]
        assert not any(isinstance(e, HookDenied) for e in events)
        warnings = [
            e
            for e in events
            if isinstance(e, HookEndEvent) and e.status == HookMessageSeverity.WARNING
        ]
        assert len(warnings) == 1


class TestHooksManagerPreCompact:
    def _make_logger(self):
        from vibe.core.config import SessionLoggingConfig
        from vibe.core.session.session_logger import SessionLogger

        return SessionLogger(SessionLoggingConfig(enabled=False), "test-id")

    def _make_hook(self, command: str, name: str = "pc-hook") -> HookConfig:
        return HookConfig(
            name=name, type=HookType.PRE_COMPACT, command=command, timeout=30.0
        )

    @pytest.mark.asyncio
    async def test_payload_contains_reason_and_tokens(self) -> None:
        hook = self._make_hook(
            f"{sys.executable} -c \""
            f"import sys,json; d=json.load(sys.stdin); "
            f"assert d['reason']=='auto_compact'; "
            f"assert d['token_estimate_before']==50000; "
            f"print('ok')\""
        )
        handler = HooksManager([hook])
        logger = self._make_logger()
        events = [
            ev
            async for ev in handler.run_pre_compact(
                "sess",
                logger,
                reason="auto_compact",
                token_estimate_before=50000,
                auto_compact_threshold=200000,
            )
        ]
        end_events = [e for e in events if isinstance(e, HookEndEvent)]
        assert any(e.status == HookMessageSeverity.OK for e in end_events)

    @pytest.mark.asyncio
    async def test_stdout_is_ignored_not_injected(self) -> None:
        hook = self._make_hook("echo some-output")
        handler = HooksManager([hook])
        logger = self._make_logger()
        events = [
            ev async for ev in handler.run_pre_compact("sess", logger)
        ]
        assert not any(isinstance(e, HookInjectedContext) for e in events)

    @pytest.mark.asyncio
    async def test_timeout_fails_open(self) -> None:
        hook = HookConfig(
            name="slow", type=HookType.PRE_COMPACT, command="sleep 60", timeout=0.5
        )
        handler = HooksManager([hook])
        logger = self._make_logger()
        events = [ev async for ev in handler.run_pre_compact("sess", logger)]
        warnings = [
            e
            for e in events
            if isinstance(e, HookEndEvent) and e.status == HookMessageSeverity.WARNING
        ]
        assert len(warnings) == 1

    @pytest.mark.asyncio
    async def test_json_inject_envelope_yields_context(self) -> None:
        payload = json.dumps(
            {"decision": "inject", "additional_context": "preserved-across-compact"}
        )
        hook = self._make_hook(f"echo '{payload}'")
        handler = HooksManager([hook])
        logger = self._make_logger()
        events = [ev async for ev in handler.run_pre_compact("sess", logger)]
        injected = [e for e in events if isinstance(e, HookInjectedContext)]
        assert len(injected) == 1
        assert injected[0].content == "preserved-across-compact"

    @pytest.mark.asyncio
    async def test_json_deny_is_logged_and_ignored(self) -> None:
        payload = json.dumps({"decision": "deny", "reason": "cannot block"})
        hook = self._make_hook(f"echo '{payload}'")
        handler = HooksManager([hook])
        logger = self._make_logger()
        events = [ev async for ev in handler.run_pre_compact("sess", logger)]
        assert not any(isinstance(e, HookInjectedContext) for e in events)
        end_events = [e for e in events if isinstance(e, HookEndEvent)]
        assert any(e.status == HookMessageSeverity.OK for e in end_events)


class TestPreCompactResultParser:
    def test_empty_stdout_returns_none(self) -> None:
        from vibe.core.hooks.manager import _parse_pre_compact_result

        assert _parse_pre_compact_result("") is None

    def test_plain_text_returns_none(self) -> None:
        from vibe.core.hooks.manager import _parse_pre_compact_result

        assert _parse_pre_compact_result("plain stdout") is None

    def test_inject_envelope_returns_injected_context(self) -> None:
        from vibe.core.hooks.manager import _parse_pre_compact_result

        payload = json.dumps(
            {"decision": "inject", "additional_context": "carry-over"}
        )
        result = _parse_pre_compact_result(payload)
        assert isinstance(result, HookInjectedContext)
        assert result.content == "carry-over"

    def test_deny_returns_none(self) -> None:
        from vibe.core.hooks.manager import _parse_pre_compact_result

        payload = json.dumps({"decision": "deny", "reason": "x"})
        assert _parse_pre_compact_result(payload) is None


class TestHooksManagerSessionEnd:
    def _make_logger(self):
        from vibe.core.config import SessionLoggingConfig
        from vibe.core.session.session_logger import SessionLogger

        return SessionLogger(SessionLoggingConfig(enabled=False), "test-id")

    def _make_hook(self, command: str, name: str = "se-hook") -> HookConfig:
        return HookConfig(
            name=name, type=HookType.SESSION_END, command=command, timeout=30.0
        )

    @pytest.mark.asyncio
    async def test_payload_contains_reason_turn_count_and_error(self) -> None:
        hook = self._make_hook(
            f"{sys.executable} -c \""
            f"import sys,json; d=json.load(sys.stdin); "
            f"assert d['hook_event_name']=='session_end'; "
            f"assert d['reason']=='error'; "
            f"assert d['turn_count']==7; "
            f"assert d['error']=='boom'; "
            f"print('ok')\""
        )
        handler = HooksManager([hook])
        logger = self._make_logger()
        events = [
            ev
            async for ev in handler.run_session_end(
                "sess", logger, reason="error", turn_count=7, error="boom"
            )
        ]
        end_events = [e for e in events if isinstance(e, HookEndEvent)]
        assert any(e.status == HookMessageSeverity.OK for e in end_events)

    @pytest.mark.asyncio
    async def test_timeout_fails_open(self) -> None:
        hook = HookConfig(
            name="slow", type=HookType.SESSION_END, command="sleep 60", timeout=0.5
        )
        handler = HooksManager([hook])
        logger = self._make_logger()
        events = [
            ev
            async for ev in handler.run_session_end(
                "sess", logger, reason="exit", turn_count=0
            )
        ]
        warnings = [
            e
            for e in events
            if isinstance(e, HookEndEvent) and e.status == HookMessageSeverity.WARNING
        ]
        assert len(warnings) == 1

    @pytest.mark.asyncio
    async def test_nonzero_exit_warns_does_not_block(self) -> None:
        hook = self._make_hook("sh -c 'echo whoops >&2; exit 1'")
        handler = HooksManager([hook])
        logger = self._make_logger()
        events = [
            ev
            async for ev in handler.run_session_end(
                "sess", logger, reason="exit", turn_count=1
            )
        ]
        warnings = [
            e
            for e in events
            if isinstance(e, HookEndEvent) and e.status == HookMessageSeverity.WARNING
        ]
        assert len(warnings) == 1

    @pytest.mark.asyncio
    async def test_deny_decision_is_ignored(self) -> None:
        payload = json.dumps({"decision": "deny", "reason": "cant block teardown"})
        hook = self._make_hook(f"echo '{payload}'")
        handler = HooksManager([hook])
        logger = self._make_logger()
        events = [
            ev
            async for ev in handler.run_session_end(
                "sess", logger, reason="exit", turn_count=2
            )
        ]
        end_events = [e for e in events if isinstance(e, HookEndEvent)]
        # Hook completed with OK status — deny was logged-and-ignored, not surfaced
        assert any(e.status == HookMessageSeverity.OK for e in end_events)

    @pytest.mark.asyncio
    async def test_missing_binary_fails_open(self) -> None:
        hook = self._make_hook("/nonexistent/path/to/hook-binary --flag")
        handler = HooksManager([hook])
        logger = self._make_logger()
        events = [
            ev
            async for ev in handler.run_session_end(
                "sess", logger, reason="exit", turn_count=0
            )
        ]
        # Some kind of HookEndEvent must be yielded; the generator must not raise
        assert any(isinstance(e, HookEndEvent) for e in events)


class TestAgentLoopSessionEndIntegration:
    @pytest.mark.asyncio
    async def test_fire_session_end_is_single_fire(self) -> None:
        backend = FakeBackend(mock_llm_chunk(content="hello"))
        hooks = [
            HookConfig(
                name="se",
                type=HookType.SESSION_END,
                command="echo end-data",
            )
        ]
        agent_loop = build_test_agent_loop(
            backend=backend,
            hook_config_result=HookConfigResult(hooks=hooks, issues=[]),
        )
        first = [ev async for ev in agent_loop.fire_session_end("exit")]
        second = [ev async for ev in agent_loop.fire_session_end("signal")]
        assert any(isinstance(e, HookEndEvent) for e in first)
        # Second call drains immediately — single-fire flag swallows it
        assert second == []

    @pytest.mark.asyncio
    async def test_turn_count_matches_post_agent_turns(self) -> None:
        backend = FakeBackend([
            [mock_llm_chunk(content="t1")],
            [mock_llm_chunk(content="t2")],
            [mock_llm_chunk(content="t3")],
        ])
        # session_end hook asserts turn_count==3 in its payload
        hooks = [
            HookConfig(
                name="se",
                type=HookType.SESSION_END,
                command=(
                    f"{sys.executable} -c \""
                    f"import sys,json; d=json.load(sys.stdin); "
                    f"assert d['turn_count']==3, f\\\"got {{d['turn_count']}}\\\"; "
                    f"print('ok')\""
                ),
            )
        ]
        agent_loop = build_test_agent_loop(
            backend=backend,
            hook_config_result=HookConfigResult(hooks=hooks, issues=[]),
        )
        [ev async for ev in agent_loop.act("p1")]
        [ev async for ev in agent_loop.act("p2")]
        [ev async for ev in agent_loop.act("p3")]
        events = [ev async for ev in agent_loop.fire_session_end("exit")]
        end_events = [e for e in events if isinstance(e, HookEndEvent)]
        assert end_events, "no HookEndEvent yielded"
        assert all(e.status == HookMessageSeverity.OK for e in end_events), (
            f"turn_count assertion failed in hook: {end_events}"
        )

    @pytest.mark.asyncio
    async def test_aclose_fires_session_end_if_not_already_fired(self) -> None:
        backend = FakeBackend(mock_llm_chunk(content="hi"))
        marker = Path("/tmp/vibe-session-end-aclose-marker.txt")
        marker.unlink(missing_ok=True)
        hooks = [
            HookConfig(
                name="se",
                type=HookType.SESSION_END,
                command=f"sh -c 'touch {marker}'",
            )
        ]
        agent_loop = build_test_agent_loop(
            backend=backend,
            hook_config_result=HookConfigResult(hooks=hooks, issues=[]),
        )
        await agent_loop.aclose()
        assert marker.exists()
        marker.unlink(missing_ok=True)


class TestAgentLoopSessionStartIntegration:
    @pytest.mark.asyncio
    async def test_session_start_fires_on_first_act(self) -> None:
        backend = FakeBackend(mock_llm_chunk(content="hello"))
        hooks = [
            HookConfig(
                name="ss",
                type=HookType.SESSION_START,
                command="echo session-init-data",
            )
        ]
        agent_loop = build_test_agent_loop(
            backend=backend,
            hook_config_result=HookConfigResult(hooks=hooks, issues=[]),
        )
        events = [ev async for ev in agent_loop.act("hi")]
        event_types = [type(e).__name__ for e in events]
        assert "HookStartEvent" in event_types
        injected = [
            m for m in agent_loop.messages if m.role.value == "user" and m.injected
        ]
        assert any("session-init-data" in (m.content or "") for m in injected)

    @pytest.mark.asyncio
    async def test_session_start_fires_only_once(self) -> None:
        backend = FakeBackend([
            [mock_llm_chunk(content="first")],
            [mock_llm_chunk(content="second")],
        ])
        hooks = [
            HookConfig(
                name="ss",
                type=HookType.SESSION_START,
                command="echo once",
            )
        ]
        agent_loop = build_test_agent_loop(
            backend=backend,
            hook_config_result=HookConfigResult(hooks=hooks, issues=[]),
        )
        await agent_loop.act("first prompt").__anext__()
        # drain first act
        [ev async for ev in agent_loop.act("first prompt")]
        injected_after_first = [
            m for m in agent_loop.messages if m.role.value == "user" and m.injected
        ]
        count_after_first = sum(
            1 for m in injected_after_first if "once" in (m.content or "")
        )

        # Second act — session_start should NOT fire again
        [ev async for ev in agent_loop.act("second prompt")]
        injected_after_second = [
            m for m in agent_loop.messages if m.role.value == "user" and m.injected
        ]
        count_after_second = sum(
            1 for m in injected_after_second if "once" in (m.content or "")
        )
        assert count_after_second == count_after_first

    @pytest.mark.asyncio
    async def test_pre_compact_hook_fires_before_compaction(self) -> None:
        from unittest.mock import AsyncMock, patch

        backend = FakeBackend([
            [mock_llm_chunk(content="summary")],
        ])
        fired: list[str] = []

        async def fake_run_pre_compact(*args, **kwargs):
            fired.append("pre_compact")
            return
            yield  # make it an async generator

        hooks = [
            HookConfig(
                name="pc",
                type=HookType.PRE_COMPACT,
                command="echo ok",
            )
        ]
        agent_loop = build_test_agent_loop(
            backend=backend,
            hook_config_result=HookConfigResult(hooks=hooks, issues=[]),
        )

        with patch.object(
            agent_loop._hooks_manager,
            "run_pre_compact",
            side_effect=fake_run_pre_compact,
        ):
            await agent_loop.compact()

        assert "pre_compact" in fired


class TestPostToolUseConfig:
    def test_post_tool_use_loads_from_toml(
        self, config_dir: Path, config_hooks_enabled: VibeConfig
    ) -> None:
        _write_hooks_toml(
            config_dir / "hooks.toml",
            [{"name": "obs", "type": "post_tool_use", "command": "echo ok"}],
        )
        result = load_hooks_from_fs(config_hooks_enabled)
        assert len(result.hooks) == 1
        assert result.hooks[0].type == HookType.POST_TOOL_USE


class TestPostToolUseResultParser:
    def test_empty_stdout_returns_none(self) -> None:
        from vibe.core.hooks.manager import _parse_post_tool_use_result

        assert _parse_post_tool_use_result("") is None

    def test_plain_text_not_injected(self) -> None:
        from vibe.core.hooks.manager import _parse_post_tool_use_result

        result = _parse_post_tool_use_result("some plain output")
        assert result is None

    def test_json_inject_returns_context(self) -> None:
        import json

        from vibe.core.hooks.manager import _parse_post_tool_use_result

        payload = json.dumps({"decision": "inject", "additional_context": "obs-data"})
        result = _parse_post_tool_use_result(payload)
        assert isinstance(result, HookInjectedContext)
        assert result.content == "obs-data"

    def test_json_allow_returns_none(self) -> None:
        import json

        from vibe.core.hooks.manager import _parse_post_tool_use_result

        payload = json.dumps({"decision": "allow"})
        assert _parse_post_tool_use_result(payload) is None

    def test_json_deny_returns_none_with_warning(self, caplog) -> None:
        import json
        import logging

        from vibe.core.hooks.manager import _parse_post_tool_use_result

        payload = json.dumps({"decision": "deny", "reason": "too late"})
        with caplog.at_level(logging.WARNING, logger="vibe.core.hooks.manager"):
            result = _parse_post_tool_use_result(payload)
        assert result is None
        assert any("deny" in r.message.lower() for r in caplog.records)

    def test_invalid_json_returns_none(self) -> None:
        from vibe.core.hooks.manager import _parse_post_tool_use_result

        assert _parse_post_tool_use_result("not json {") is None


class TestHooksManagerPostToolUse:
    def _make_logger(self):
        from vibe.core.config import SessionLoggingConfig
        from vibe.core.session.session_logger import SessionLogger

        return SessionLogger(SessionLoggingConfig(enabled=False), "test-id")

    def _make_hook(self, command: str, name: str = "obs-hook") -> HookConfig:
        return HookConfig(
            name=name, type=HookType.POST_TOOL_USE, command=command, timeout=30.0
        )

    @pytest.mark.asyncio
    async def test_payload_contains_tool_fields(self) -> None:
        hook = self._make_hook(
            f"{sys.executable} -c \""
            f"import sys,json; d=json.load(sys.stdin); "
            f"assert d['tool_name']=='bash'; "
            f"assert d['tool_call_id']=='call_1'; "
            f"assert d['duration_ms']==500; "
            f"print('ok')\""
        )
        handler = HooksManager([hook])
        logger = self._make_logger()
        events = [
            ev
            async for ev in handler.run_post_tool_use(
                "sess",
                logger,
                tool_name="bash",
                tool_call_id="call_1",
                tool_input={"command": "echo hi"},
                tool_result={"output": "hi"},
                duration_ms=500,
            )
        ]
        end_events = [e for e in events if isinstance(e, HookEndEvent)]
        assert any(e.status == HookMessageSeverity.OK for e in end_events)

    @pytest.mark.asyncio
    async def test_tool_error_absent_on_clean_result(self) -> None:
        hook = self._make_hook(
            f"{sys.executable} -c \""
            f"import sys,json; d=json.load(sys.stdin); "
            f"assert 'tool_error' not in d; "
            f"print('ok')\""
        )
        handler = HooksManager([hook])
        logger = self._make_logger()
        events = [
            ev
            async for ev in handler.run_post_tool_use(
                "sess",
                logger,
                tool_name="read",
                tool_call_id="call_2",
                tool_result={"content": "file content"},
                duration_ms=10,
            )
        ]
        end_events = [e for e in events if isinstance(e, HookEndEvent)]
        assert any(e.status == HookMessageSeverity.OK for e in end_events)

    @pytest.mark.asyncio
    async def test_tool_error_present_when_set(self) -> None:
        import json

        hook = self._make_hook(
            f"{sys.executable} -c \""
            f"import sys,json; d=json.load(sys.stdin); "
            f"assert d['tool_error']=='command failed'; "
            f"print('ok')\""
        )
        handler = HooksManager([hook])
        logger = self._make_logger()
        events = [
            ev
            async for ev in handler.run_post_tool_use(
                "sess",
                logger,
                tool_name="bash",
                tool_call_id="call_3",
                tool_error="command failed",
                duration_ms=5,
            )
        ]
        end_events = [e for e in events if isinstance(e, HookEndEvent)]
        assert any(e.status == HookMessageSeverity.OK for e in end_events)

    @pytest.mark.asyncio
    async def test_json_inject_yields_context(self) -> None:
        import json

        payload = json.dumps({"decision": "inject", "additional_context": "obs-note"})
        hook = self._make_hook(f"echo '{payload}'")
        handler = HooksManager([hook])
        logger = self._make_logger()
        events = [
            ev
            async for ev in handler.run_post_tool_use(
                "sess",
                logger,
                tool_name="bash",
                tool_call_id="call_4",
                duration_ms=1,
            )
        ]
        injected = [e for e in events if isinstance(e, HookInjectedContext)]
        assert len(injected) == 1
        assert injected[0].content == "obs-note"

    @pytest.mark.asyncio
    async def test_plain_stdout_not_injected(self) -> None:
        hook = self._make_hook("echo plain-observation")
        handler = HooksManager([hook])
        logger = self._make_logger()
        events = [
            ev
            async for ev in handler.run_post_tool_use(
                "sess",
                logger,
                tool_name="bash",
                tool_call_id="call_5",
                duration_ms=1,
            )
        ]
        assert not any(isinstance(e, HookInjectedContext) for e in events)

    @pytest.mark.asyncio
    async def test_timeout_fails_open(self) -> None:
        hook = HookConfig(
            name="slow", type=HookType.POST_TOOL_USE, command="sleep 60", timeout=0.5
        )
        handler = HooksManager([hook])
        logger = self._make_logger()
        events = [
            ev
            async for ev in handler.run_post_tool_use(
                "sess",
                logger,
                tool_name="bash",
                tool_call_id="call_6",
                duration_ms=1,
            )
        ]
        warnings = [
            e
            for e in events
            if isinstance(e, HookEndEvent) and e.status == HookMessageSeverity.WARNING
        ]
        assert len(warnings) == 1


class TestAgentLoopPostToolUseIntegration:
    @pytest.mark.asyncio
    async def test_post_tool_use_fires_after_tool_execution(self) -> None:
        from unittest.mock import patch

        from tests.conftest import build_test_vibe_config
        from vibe.core.agents.models import BuiltinAgentName
        from vibe.core.tools.base import ToolPermission
        from vibe.core.types import FunctionCall, ToolCall

        tool_call = ToolCall(
            id="call_1",
            index=0,
            function=FunctionCall(name="todo", arguments='{"action": "read"}'),
        )
        backend = FakeBackend([
            [mock_llm_chunk(content="checking todos", tool_calls=[tool_call])],
            [mock_llm_chunk(content="done")],
        ])
        config = build_test_vibe_config(
            enabled_tools=["todo"],
            tools={"todo": {"permission": ToolPermission.ALWAYS.value}},
        )
        hooks = [
            HookConfig(
                name="obs",
                type=HookType.POST_TOOL_USE,
                command="echo observed",
            )
        ]
        agent_loop = build_test_agent_loop(
            config=config,
            agent_name=BuiltinAgentName.AUTO_APPROVE,
            backend=backend,
            hook_config_result=HookConfigResult(hooks=hooks, issues=[]),
        )
        events = [ev async for ev in agent_loop.act("check todos")]
        event_types = [type(e).__name__ for e in events]
        assert "HookStartEvent" in event_types
        assert "HookEndEvent" in event_types


class TestPreToolUseConfig:
    def test_pre_tool_use_loads_from_toml(
        self, config_dir: Path, config_hooks_enabled: VibeConfig
    ) -> None:
        _write_hooks_toml(
            config_dir / "hooks.toml",
            [{"name": "guard", "type": "pre_tool_use", "command": "echo ok"}],
        )
        result = load_hooks_from_fs(config_hooks_enabled)
        assert len(result.hooks) == 1
        assert result.hooks[0].type == HookType.PRE_TOOL_USE


class TestPreToolUseResultParser:
    def _parse(self, stdout: str, stderr: str = "", exit_code: int = 0):
        from vibe.core.hooks.manager import _parse_pre_tool_use_result

        return _parse_pre_tool_use_result(stdout, stderr, exit_code)

    def test_empty_stdout_exit0_allows(self) -> None:
        assert self._parse("", exit_code=0) is None

    def test_json_deny_returns_denied(self) -> None:
        import json

        payload = json.dumps({"decision": "deny", "reason": "blocked by policy"})
        result = self._parse(payload, exit_code=0)
        assert isinstance(result, HookDenied)
        assert result.reason == "blocked by policy"

    def test_json_allow_returns_none(self) -> None:
        import json

        payload = json.dumps({"decision": "allow"})
        assert self._parse(payload, exit_code=0) is None

    def test_exit_code_2_denies(self) -> None:
        result = self._parse("", stderr="rm -rf denied", exit_code=2)
        assert isinstance(result, HookDenied)
        assert result.reason == "rm -rf denied"

    def test_exit_code_2_reason_from_stdout_if_no_stderr(self) -> None:
        result = self._parse("blocked stdout", stderr="", exit_code=2)
        assert isinstance(result, HookDenied)
        assert result.reason == "blocked stdout"

    def test_plain_text_allows(self) -> None:
        result = self._parse("some diagnostic text", exit_code=0)
        assert result is None

    def test_invalid_json_allows(self) -> None:
        result = self._parse("not json {", exit_code=0)
        assert result is None

    def test_nonzero_other_exit_allows(self) -> None:
        result = self._parse("", exit_code=1)
        assert result is None


class TestHooksManagerPreToolUse:
    def _make_logger(self):
        from vibe.core.config import SessionLoggingConfig
        from vibe.core.session.session_logger import SessionLogger

        return SessionLogger(SessionLoggingConfig(enabled=False), "test-id")

    def _make_hook(self, command: str, name: str = "guard-hook") -> HookConfig:
        return HookConfig(
            name=name, type=HookType.PRE_TOOL_USE, command=command, timeout=30.0
        )

    @pytest.mark.asyncio
    async def test_payload_contains_tool_fields(self) -> None:
        hook = self._make_hook(
            f"{sys.executable} -c \""
            f"import sys,json; d=json.load(sys.stdin); "
            f"assert d['tool_name']=='bash'; "
            f"assert d['tool_call_id']=='call_99'; "
            f"assert d['hook_event_name']=='pre_tool_use'; "
            f"print('ok')\""
        )
        handler = HooksManager([hook])
        logger = self._make_logger()
        events = [
            ev
            async for ev in handler.run_pre_tool_use(
                "sess",
                logger,
                tool_name="bash",
                tool_call_id="call_99",
                tool_input={"command": "echo hi"},
            )
        ]
        end_events = [e for e in events if isinstance(e, HookEndEvent)]
        assert any(e.status == HookMessageSeverity.OK for e in end_events)

    @pytest.mark.asyncio
    async def test_json_deny_yields_hook_denied(self) -> None:
        import json

        payload = json.dumps({"decision": "deny", "reason": "not allowed"})
        hook = self._make_hook(f"echo '{payload}'")
        handler = HooksManager([hook])
        logger = self._make_logger()
        events = [
            ev
            async for ev in handler.run_pre_tool_use(
                "sess",
                logger,
                tool_name="bash",
                tool_call_id="call_1",
            )
        ]
        denied = [e for e in events if isinstance(e, HookDenied)]
        assert len(denied) == 1
        assert denied[0].reason == "not allowed"

    @pytest.mark.asyncio
    async def test_json_allow_yields_no_denied(self) -> None:
        import json

        payload = json.dumps({"decision": "allow"})
        hook = self._make_hook(f"echo '{payload}'")
        handler = HooksManager([hook])
        logger = self._make_logger()
        events = [
            ev
            async for ev in handler.run_pre_tool_use(
                "sess",
                logger,
                tool_name="bash",
                tool_call_id="call_2",
            )
        ]
        assert not any(isinstance(e, HookDenied) for e in events)

    @pytest.mark.asyncio
    async def test_empty_stdout_allows(self) -> None:
        hook = self._make_hook("true")
        handler = HooksManager([hook])
        logger = self._make_logger()
        events = [
            ev
            async for ev in handler.run_pre_tool_use(
                "sess",
                logger,
                tool_name="bash",
                tool_call_id="call_3",
            )
        ]
        assert not any(isinstance(e, HookDenied) for e in events)

    @pytest.mark.asyncio
    async def test_exit_code_2_yields_denied(self) -> None:
        hook = self._make_hook(f"{sys.executable} -c \"import sys; sys.exit(2)\"")
        handler = HooksManager([hook])
        logger = self._make_logger()
        events = [
            ev
            async for ev in handler.run_pre_tool_use(
                "sess",
                logger,
                tool_name="bash",
                tool_call_id="call_4",
            )
        ]
        denied = [e for e in events if isinstance(e, HookDenied)]
        assert len(denied) == 1

    @pytest.mark.asyncio
    async def test_timeout_fails_open(self) -> None:
        hook = HookConfig(
            name="slow", type=HookType.PRE_TOOL_USE, command="sleep 60", timeout=0.5
        )
        handler = HooksManager([hook])
        logger = self._make_logger()
        events = [
            ev
            async for ev in handler.run_pre_tool_use(
                "sess",
                logger,
                tool_name="bash",
                tool_call_id="call_5",
            )
        ]
        assert not any(isinstance(e, HookDenied) for e in events)
        warnings = [
            e
            for e in events
            if isinstance(e, HookEndEvent) and e.status == HookMessageSeverity.WARNING
        ]
        assert len(warnings) == 1


class TestAgentLoopPreToolUseIntegration:
    @pytest.mark.asyncio
    async def test_deny_prevents_tool_execution(self) -> None:
        import json
        from unittest.mock import patch

        from tests.conftest import build_test_vibe_config
        from vibe.core.agents.models import BuiltinAgentName
        from vibe.core.tools.base import ToolPermission
        from vibe.core.types import FunctionCall, ToolCall

        tool_call = ToolCall(
            id="call_block",
            index=0,
            function=FunctionCall(name="todo", arguments='{"action": "read"}'),
        )
        backend = FakeBackend([
            [mock_llm_chunk(content="let me check todos", tool_calls=[tool_call])],
            [mock_llm_chunk(content="ok blocked")],
        ])
        config = build_test_vibe_config(
            enabled_tools=["todo"],
            tools={"todo": {"permission": ToolPermission.ALWAYS.value}},
        )
        deny_payload = json.dumps({"decision": "deny", "reason": "blocked by test"})
        hooks = [
            HookConfig(
                name="guard",
                type=HookType.PRE_TOOL_USE,
                command=f"echo '{deny_payload}'",
            )
        ]
        agent_loop = build_test_agent_loop(
            config=config,
            agent_name=BuiltinAgentName.AUTO_APPROVE,
            backend=backend,
            hook_config_result=HookConfigResult(hooks=hooks, issues=[]),
        )

        invoke_calls: list[str] = []

        from vibe.core.tools.builtins.todo import Todo
        original_invoke = Todo.invoke

        async def tracking_invoke(self_tool, *args, **kwargs):
            invoke_calls.append("invoked")
            async for item in original_invoke(self_tool, *args, **kwargs):
                yield item

        with patch.object(Todo, "invoke", tracking_invoke):
            events = [ev async for ev in agent_loop.act("check todos")]

        assert len(invoke_calls) == 0, "Tool should not have been invoked when pre_tool_use denies"
        error_events = [
            e for e in events
            if hasattr(e, "error") and e.error and "blocked by pre_tool_use hook" in e.error
        ]
        assert len(error_events) >= 1
