from __future__ import annotations

from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from enum import IntEnum
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vibe.core.hooks.config import HookConfig
from vibe.core.hooks.executor import HookExecutor
from vibe.core.hooks.models import (
    HookDecision,
    HookDenied,
    HookEndEvent,
    HookInjectedContext,
    HookInvocation,
    HookMessageSeverity,
    HookRunEndEvent,
    HookRunStartEvent,
    HookStartEvent,
    HookType,
    HookUserMessage,
)
from vibe.core.types import BaseEvent

if TYPE_CHECKING:
    from vibe.core.session.session_logger import SessionLogger

try:
    from vibe import __version__ as _VIBE_VERSION
except ImportError:
    _VIBE_VERSION = "unknown"

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3


class HookExitCode(IntEnum):
    SUCCESS = 0
    RETRY = 2


class HookRetryState:
    def __init__(self) -> None:
        self._counts: dict[str, int] = {}

    def reset(self) -> None:
        self._counts.clear()

    def remaining_retries(self, hook_name: str) -> int:
        return _MAX_RETRIES - self._counts.get(hook_name, 0)

    def track_retry(self, hook_name: str) -> None:
        self._counts[hook_name] = self._counts.get(hook_name, 0) + 1

    def track_success(self, hook_name: str) -> None:
        self._counts.pop(hook_name, None)

    def should_retry(self, hook_name: str) -> bool:
        return self._counts.get(hook_name, 0) < _MAX_RETRIES


class HooksManager:
    def __init__(self, hooks: list[HookConfig]) -> None:
        self._hooks_by_type: dict[HookType, list[HookConfig]] = {}
        for hook in hooks:
            self._hooks_by_type.setdefault(hook.type, []).append(hook)
        self._executor = HookExecutor()
        self._retry_state = HookRetryState()

    def has_hooks(self, hook_type: HookType) -> bool:
        return bool(self._hooks_by_type.get(hook_type))

    def reset_retry_count(self) -> None:
        self._retry_state.reset()

    async def run(
        self, hook_type: HookType, session_id: str, session_logger: SessionLogger
    ) -> AsyncGenerator[BaseEvent | HookUserMessage]:
        hooks = self._hooks_by_type.get(hook_type, [])
        if not hooks:
            return
        invocation = _build_invocation(hook_type, session_id, session_logger)

        yield HookRunStartEvent()
        for hook in hooks:
            yield HookStartEvent(hook_name=hook.name)
            result = await self._executor.run(hook, invocation)

            if result.timed_out or result.exit_code is None:
                yield HookEndEvent(
                    hook_name=hook.name,
                    status=HookMessageSeverity.WARNING,
                    content=f"Timed out after {hook.timeout}s",
                )
            elif result.exit_code == HookExitCode.SUCCESS:
                yield HookEndEvent(hook_name=hook.name, status=HookMessageSeverity.OK)
            elif result.exit_code == HookExitCode.RETRY and result.stdout:
                logger.debug("Hook %s retry output: %s", hook.name, result.stdout)

                if not self._retry_state.should_retry(hook.name):
                    yield HookEndEvent(
                        hook_name=hook.name,
                        status=HookMessageSeverity.ERROR,
                        content=f"Failed, retries exhausted ({_MAX_RETRIES}/{_MAX_RETRIES})",
                    )
                    continue

                remaining = self._retry_state.remaining_retries(hook.name)
                self._retry_state.track_retry(hook.name)
                yield HookEndEvent(
                    hook_name=hook.name,
                    status=HookMessageSeverity.ERROR,
                    content=f"Failed, retrying ({remaining} {'retry' if remaining == 1 else 'retries'} remaining)",
                )
                yield HookUserMessage(content=result.stdout)
                break
            else:
                yield HookEndEvent(
                    hook_name=hook.name,
                    status=HookMessageSeverity.WARNING,
                    content=(
                        result.stdout
                        or result.stderr
                        or f"Exited with code {result.exit_code}"
                    ),
                )

            if result.exit_code != HookExitCode.RETRY:
                self._retry_state.track_success(hook.name)

        yield HookRunEndEvent()

    async def run_user_prompt_submit(
        self,
        prompt: str,
        session_id: str,
        session_logger: SessionLogger,
        *,
        message_id: str | None = None,
        project: str | None = None,
    ) -> AsyncGenerator[BaseEvent | HookInjectedContext | HookDenied]:
        hooks = self._hooks_by_type.get(HookType.USER_PROMPT_SUBMIT, [])
        if not hooks:
            return
        invocation = _build_invocation(
            HookType.USER_PROMPT_SUBMIT,
            session_id,
            session_logger,
            prompt=prompt,
            message_id=message_id,
            project=project,
        )

        yield HookRunStartEvent()
        for hook in hooks:
            yield HookStartEvent(hook_name=hook.name)
            result = await self._executor.run(hook, invocation)

            if result.timed_out or result.exit_code is None:
                yield HookEndEvent(
                    hook_name=hook.name,
                    status=HookMessageSeverity.WARNING,
                    content=f"Timed out after {hook.timeout}s",
                )
                continue

            # Parse stdout before checking exit code — deny works on any exit code.
            decision_result = _parse_user_prompt_submit_result(result.stdout)

            if isinstance(decision_result, HookDenied):
                yield HookEndEvent(
                    hook_name=hook.name,
                    status=HookMessageSeverity.ERROR,
                    content=f"Prompt denied: {decision_result.reason or 'no reason given'}",
                )
                yield decision_result
                yield HookRunEndEvent()
                return

            if result.exit_code == HookExitCode.SUCCESS:
                if isinstance(decision_result, HookInjectedContext):
                    yield HookEndEvent(hook_name=hook.name, status=HookMessageSeverity.OK)
                    yield decision_result
                else:
                    yield HookEndEvent(hook_name=hook.name, status=HookMessageSeverity.OK)
            else:
                # Non-zero exit without a deny decision: warn and fail open.
                yield HookEndEvent(
                    hook_name=hook.name,
                    status=HookMessageSeverity.WARNING,
                    content=(
                        result.stdout
                        or result.stderr
                        or f"Exited with code {result.exit_code}"
                    ),
                )

        yield HookRunEndEvent()

    async def run_session_start(
        self,
        source: str,
        session_id: str,
        session_logger: SessionLogger,
        *,
        parent_session_id: str | None = None,
    ) -> AsyncGenerator[BaseEvent | HookInjectedContext]:
        hooks = self._hooks_by_type.get(HookType.SESSION_START, [])
        if not hooks:
            return
        invocation = _build_invocation(
            HookType.SESSION_START,
            session_id,
            session_logger,
            source=source,
            parent_session_id=parent_session_id,
        )

        yield HookRunStartEvent()
        for hook in hooks:
            yield HookStartEvent(hook_name=hook.name)
            result = await self._executor.run(hook, invocation)

            if result.timed_out or result.exit_code is None:
                yield HookEndEvent(
                    hook_name=hook.name,
                    status=HookMessageSeverity.WARNING,
                    content=f"Timed out after {hook.timeout}s",
                )
                continue

            if result.exit_code == HookExitCode.SUCCESS:
                if result.stdout:
                    # Plain text or JSON inject: reuse the shared parser.
                    decision_result = _parse_user_prompt_submit_result(result.stdout)
                    if isinstance(decision_result, HookInjectedContext):
                        yield HookEndEvent(
                            hook_name=hook.name, status=HookMessageSeverity.OK
                        )
                        yield decision_result
                        continue
                yield HookEndEvent(hook_name=hook.name, status=HookMessageSeverity.OK)
            else:
                yield HookEndEvent(
                    hook_name=hook.name,
                    status=HookMessageSeverity.WARNING,
                    content=(
                        result.stdout
                        or result.stderr
                        or f"Exited with code {result.exit_code}"
                    ),
                )

        yield HookRunEndEvent()

    async def run_pre_compact(
        self,
        session_id: str,
        session_logger: SessionLogger,
        *,
        reason: str = "auto_compact",
        token_estimate_before: int | None = None,
        auto_compact_threshold: int | None = None,
    ) -> AsyncGenerator[BaseEvent | HookInjectedContext]:
        hooks = self._hooks_by_type.get(HookType.PRE_COMPACT, [])
        if not hooks:
            return
        invocation = _build_invocation(
            HookType.PRE_COMPACT,
            session_id,
            session_logger,
            reason=reason,
            token_estimate_before=token_estimate_before,
            auto_compact_threshold=auto_compact_threshold,
        )

        yield HookRunStartEvent()
        for hook in hooks:
            yield HookStartEvent(hook_name=hook.name)
            result = await self._executor.run(hook, invocation)

            if result.timed_out or result.exit_code is None:
                yield HookEndEvent(
                    hook_name=hook.name,
                    status=HookMessageSeverity.WARNING,
                    content=f"Timed out after {hook.timeout}s",
                )
            elif result.exit_code == HookExitCode.SUCCESS:
                if result.stdout:
                    # JSON inject envelope (preferred) surfaces context that
                    # survives compaction; plain stdout is debug-logged only.
                    decision_result = _parse_pre_compact_result(result.stdout)
                    if isinstance(decision_result, HookInjectedContext):
                        yield HookEndEvent(
                            hook_name=hook.name, status=HookMessageSeverity.OK
                        )
                        yield decision_result
                        continue
                    logger.debug("pre_compact hook %s output: %s", hook.name, result.stdout)
                yield HookEndEvent(hook_name=hook.name, status=HookMessageSeverity.OK)
            else:
                yield HookEndEvent(
                    hook_name=hook.name,
                    status=HookMessageSeverity.WARNING,
                    content=(
                        result.stdout
                        or result.stderr
                        or f"Exited with code {result.exit_code}"
                    ),
                )

        yield HookRunEndEvent()

    async def run_session_end(
        self,
        session_id: str,
        session_logger: SessionLogger,
        *,
        reason: str,
        turn_count: int,
        error: str | None = None,
    ) -> AsyncGenerator[BaseEvent]:
        """Fire session_end hooks. Observational only: no decision semantics.

        deny is logged and ignored — the session cannot be kept alive.
        Plain stdout is debug-logged. Fail-open on every error path.
        """
        hooks = self._hooks_by_type.get(HookType.SESSION_END, [])
        if not hooks:
            return
        invocation = _build_invocation(
            HookType.SESSION_END,
            session_id,
            session_logger,
            reason=reason,
            turn_count=turn_count,
            error=error,
        )

        yield HookRunStartEvent()
        for hook in hooks:
            yield HookStartEvent(hook_name=hook.name)
            result = await self._executor.run(hook, invocation)

            if result.timed_out or result.exit_code is None:
                yield HookEndEvent(
                    hook_name=hook.name,
                    status=HookMessageSeverity.WARNING,
                    content=f"Timed out after {hook.timeout}s",
                )
            elif result.exit_code == HookExitCode.SUCCESS:
                if result.stdout:
                    try:
                        data = json.loads(result.stdout)
                        if (
                            isinstance(data, dict)
                            and data.get("decision") == "deny"
                        ):
                            logger.warning(
                                "session_end hook %s returned deny — session cannot be kept alive, ignoring",
                                hook.name,
                            )
                    except (json.JSONDecodeError, ValueError):
                        pass
                    logger.debug("session_end hook %s output: %s", hook.name, result.stdout)
                yield HookEndEvent(hook_name=hook.name, status=HookMessageSeverity.OK)
            else:
                yield HookEndEvent(
                    hook_name=hook.name,
                    status=HookMessageSeverity.WARNING,
                    content=(
                        result.stdout
                        or result.stderr
                        or f"Exited with code {result.exit_code}"
                    ),
                )

        yield HookRunEndEvent()

    async def run_pre_tool_use(
        self,
        session_id: str,
        session_logger: SessionLogger,
        *,
        tool_name: str,
        tool_call_id: str,
        tool_input: dict[str, Any] | None = None,
    ) -> AsyncGenerator[BaseEvent | HookDenied]:
        hooks = self._hooks_by_type.get(HookType.PRE_TOOL_USE, [])
        if not hooks:
            return
        invocation = _build_invocation(
            HookType.PRE_TOOL_USE,
            session_id,
            session_logger,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            tool_input=tool_input,
        )

        yield HookRunStartEvent()
        for hook in hooks:
            yield HookStartEvent(hook_name=hook.name)
            result = await self._executor.run(hook, invocation)

            if result.timed_out or result.exit_code is None:
                yield HookEndEvent(
                    hook_name=hook.name,
                    status=HookMessageSeverity.WARNING,
                    content=f"Timed out after {hook.timeout}s",
                )
                continue

            decision_result = _parse_pre_tool_use_result(result.stdout, result.stderr, result.exit_code)

            if isinstance(decision_result, HookDenied):
                yield HookEndEvent(
                    hook_name=hook.name,
                    status=HookMessageSeverity.ERROR,
                    content=f"Tool blocked: {decision_result.reason or 'no reason given'}",
                )
                yield decision_result
                yield HookRunEndEvent()
                return

            yield HookEndEvent(hook_name=hook.name, status=HookMessageSeverity.OK)

        yield HookRunEndEvent()

    async def run_post_tool_use(
        self,
        session_id: str,
        session_logger: SessionLogger,
        *,
        tool_name: str,
        tool_call_id: str,
        tool_input: dict[str, Any] | None = None,
        tool_result: Any | None = None,
        tool_error: str | None = None,
        exit_code: int | None = None,
        duration_ms: int | None = None,
    ) -> AsyncGenerator[BaseEvent | HookInjectedContext]:
        hooks = self._hooks_by_type.get(HookType.POST_TOOL_USE, [])
        if not hooks:
            return
        invocation = _build_invocation(
            HookType.POST_TOOL_USE,
            session_id,
            session_logger,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            tool_input=tool_input,
            tool_result=tool_result,
            tool_error=tool_error,
            exit_code=exit_code,
            duration_ms=duration_ms,
        )

        yield HookRunStartEvent()
        for hook in hooks:
            yield HookStartEvent(hook_name=hook.name)
            result = await self._executor.run(hook, invocation)

            if result.timed_out or result.exit_code is None:
                yield HookEndEvent(
                    hook_name=hook.name,
                    status=HookMessageSeverity.WARNING,
                    content=f"Timed out after {hook.timeout}s",
                )
                continue

            if result.exit_code == HookExitCode.SUCCESS:
                decision_result = _parse_post_tool_use_result(result.stdout)
                if isinstance(decision_result, HookInjectedContext):
                    yield HookEndEvent(hook_name=hook.name, status=HookMessageSeverity.OK)
                    yield decision_result
                else:
                    yield HookEndEvent(hook_name=hook.name, status=HookMessageSeverity.OK)
            else:
                yield HookEndEvent(
                    hook_name=hook.name,
                    status=HookMessageSeverity.WARNING,
                    content=(
                        result.stdout
                        or result.stderr
                        or f"Exited with code {result.exit_code}"
                    ),
                )

        yield HookRunEndEvent()


def _build_invocation(
    hook_type: HookType,
    session_id: str,
    session_logger: SessionLogger,
    *,
    prompt: str | None = None,
    message_id: str | None = None,
    project: str | None = None,
    source: str | None = None,
    parent_session_id: str | None = None,
    reason: str | None = None,
    token_estimate_before: int | None = None,
    auto_compact_threshold: int | None = None,
    tool_name: str | None = None,
    tool_call_id: str | None = None,
    tool_input: dict[str, Any] | None = None,
    tool_result: Any | None = None,
    tool_error: str | None = None,
    exit_code: int | None = None,
    duration_ms: int | None = None,
    turn_count: int | None = None,
    error: str | None = None,
) -> HookInvocation:
    transcript_path = ""
    if session_logger.enabled and session_logger.session_dir is not None:
        transcript_path = str(session_logger.messages_filepath.resolve())

    timestamp = datetime.now(timezone.utc).isoformat()

    return HookInvocation(
        session_id=session_id,
        transcript_path=transcript_path,
        cwd=str(Path.cwd().resolve()),
        hook_event_name=hook_type.value,
        timestamp=timestamp,
        vibe_version=_VIBE_VERSION,
        prompt=prompt,
        message_id=message_id,
        project=project,
        source=source,
        parent_session_id=parent_session_id,
        reason=reason,
        token_estimate_before=token_estimate_before,
        auto_compact_threshold=auto_compact_threshold,
        tool_name=tool_name,
        tool_call_id=tool_call_id,
        tool_input=tool_input,
        tool_result=tool_result,
        tool_error=tool_error,
        exit_code=exit_code,
        duration_ms=duration_ms,
        turn_count=turn_count,
        error=error,
    )


def _parse_user_prompt_submit_result(
    stdout: str,
) -> HookInjectedContext | HookDenied | None:
    """Parse hook stdout for user_prompt_submit into a structured decision."""
    if not stdout:
        return None

    try:
        data = json.loads(stdout)
        if isinstance(data, dict):
            # Claude-style: {hookSpecificOutput: {hookEventName, additionalContext}}
            if hook_specific := data.get("hookSpecificOutput"):
                if isinstance(hook_specific, dict):
                    ctx = hook_specific.get("additionalContext")
                    if ctx:
                        return HookInjectedContext(content=str(ctx))
                return None

            # Standard JSON decision schema
            if "decision" in data:
                decision = HookDecision.model_validate(data)
                if decision.decision == "deny":
                    return HookDenied(reason=decision.reason)
                if decision.additional_context:
                    return HookInjectedContext(content=decision.additional_context)
                return None
    except (json.JSONDecodeError, ValueError):
        pass

    # Plain text stdout: inject as additional context
    return HookInjectedContext(content=stdout)


def _parse_pre_compact_result(
    stdout: str,
) -> HookInjectedContext | None:
    """Parse hook stdout for pre_compact.

    Only an explicit ``{decision: "inject", additional_context: "..."}``
    (or ``{decision: "allow", additional_context: "..."}``) envelope triggers
    injection. ``deny`` has no semantics here (compaction proceeds) and is
    logged as a warning. Plain text is debug-logged only — pre_compact is
    fire-and-forget unless an envelope opts in.
    """
    if not stdout:
        return None

    try:
        data = json.loads(stdout)
        if isinstance(data, dict) and "decision" in data:
            decision = HookDecision.model_validate(data)
            if decision.decision == "deny":
                logger.warning(
                    "pre_compact hook returned deny — compaction cannot be blocked, ignoring"
                )
                return None
            if decision.additional_context:
                return HookInjectedContext(content=decision.additional_context)
            return None
    except (json.JSONDecodeError, ValueError):
        pass

    return None


def _parse_pre_tool_use_result(
    stdout: str,
    stderr: str,
    exit_code: int,
) -> HookDenied | None:
    """Parse hook stdout/exit_code for pre_tool_use.

    Exit code 2 means deny (reason from stderr). JSON {decision:"deny"} means deny.
    Everything else (empty, plain text, JSON allow, non-zero other) means allow (fail open).
    """
    # Exit code 2 is an explicit deny signal.
    if exit_code == HookExitCode.RETRY:
        reason = stderr.strip() or stdout.strip() or None
        return HookDenied(reason=reason)

    if stdout:
        try:
            data = json.loads(stdout)
            if isinstance(data, dict) and "decision" in data:
                decision = HookDecision.model_validate(data)
                if decision.decision == "deny":
                    return HookDenied(reason=decision.reason)
                # allow / inject / other: fall through
                return None
        except (json.JSONDecodeError, ValueError):
            pass

        # Plain text: debug-log only, allow through.
        logger.debug("pre_tool_use hook stdout (allow): %s", stdout)

    return None


def _parse_post_tool_use_result(
    stdout: str,
) -> HookInjectedContext | None:
    """Parse hook stdout for post_tool_use.

    Plain text is logged only — not injected. Only an explicit JSON
    ``{decision: "inject", additional_context: "..."}`` triggers injection.
    ``deny`` is not supported (tool already ran); it is logged as a warning.
    """
    if not stdout:
        return None

    try:
        data = json.loads(stdout)
        if isinstance(data, dict) and "decision" in data:
            decision = HookDecision.model_validate(data)
            if decision.decision == "inject" and decision.additional_context:
                return HookInjectedContext(content=decision.additional_context)
            if decision.decision == "deny":
                logger.warning(
                    "post_tool_use hook returned deny, but tool already ran — ignoring"
                )
    except (json.JSONDecodeError, ValueError):
        pass

    # Plain text or unhandled JSON: log at debug level, do not inject.
    logger.debug("post_tool_use hook stdout (not injected): %s", stdout)
    return None
