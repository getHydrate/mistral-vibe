"""Hook orchestration mixin for AgentLoop.

Provides before_tool, after_tool, and post_agent_turn hook lifecycle
methods. Extracted from the AgentLoop implementation module to keep it
focused on the core conversation loop and tool execution flow.

Implicit dependencies on the host class (AgentLoop):

Attributes:
    _hooks_manager   (HooksManager | None)
    session_id       (str)
    parent_session_id (str | None)
    session_logger   (SessionLogger)
    stats            (AgentStats)
    messages         (MessageList)

Methods:
    _handle_tool_response(tool_call, text, status, decision, result, span)
    _serialize_tool_input(tool_call) -> dict[str, Any]
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

from pydantic import ValidationError

from vibe.core.hooks.models import (
    AfterToolInvocation,
    BeforeToolInvocation,
    HookContextInjection,
    HookEvent,
    HookPromptDenial,
    HookSessionContext,
    HookTextReplacement,
    HookToolDenial,
    HookToolInputRewrite,
    HookUserMessage,
    NotificationInvocation,
    PostAgentTurnInvocation,
    PostCompactInvocation,
    PreCompactInvocation,
    SessionEndInvocation,
    SessionStartInvocation,
    StopFailureInvocation,
    ToolStatus,
    UserPromptSubmitInvocation,
)
from vibe.core.llm.format import ResolvedToolCall
from vibe.core.logger import logger
from vibe.core.types import ToolResultEvent
from vibe.core.utils import (
    CANCELLATION_TAG,
    TOOL_ERROR_TAG,
    CancellationReason,
    get_user_cancellation_message,
)

if TYPE_CHECKING:
    from opentelemetry import trace

    from vibe.core.agent_loop import ToolDecision
    from vibe.core.hooks.manager import HooksManager
    from vibe.core.session.session_logger import SessionLogger
    from vibe.core.types import AgentStats, BaseEvent, LLMMessage, MessageList


# stop_failure invocations carry the stringified error; cap it so hook
# subprocesses never receive an unbounded payload on stdin.
_STOP_FAILURE_MESSAGE_MAX_CHARS = 2000


class _BeforeToolResolution(NamedTuple):
    # ``denial_event`` is non-None when the pipeline ended in a denial
    # (explicit or synthesized from a failed rewrite re-validation);
    # callers yield it and stop.  Otherwise tool_call / tool_input hold
    # the (possibly rewritten) values to use for permission + execution.
    tool_call: ResolvedToolCall
    tool_input: dict[str, Any]
    denial_event: ToolResultEvent | None


class AgentLoopHooksMixin:
    """Mixin that adds hook orchestration to AgentLoop.

    See module docstring for the implicit contract with the host class.
    """

    # Declared for type-checking only; set by AgentLoop.__init__.
    _hooks_manager: HooksManager | None
    session_id: str
    parent_session_id: str | None
    session_logger: SessionLogger
    stats: AgentStats
    messages: MessageList
    _pending_session_start_source: str | None
    _prompt_blocked: bool

    def _handle_tool_response(
        self,
        tool_call: ResolvedToolCall,
        text: str,
        status: Literal["success", "failure", "skipped"],
        decision: ToolDecision | None = None,
        result: dict[str, Any] | None = None,
        span: trace.Span | None = None,
    ) -> None: ...

    def _serialize_tool_input(self, tool_call: ResolvedToolCall) -> dict[str, Any]:
        return tool_call.validated_args.model_dump(mode="json")

    # ------------------------------------------------------------------
    # Session context
    # ------------------------------------------------------------------

    def _hook_session_context(self) -> HookSessionContext:
        transcript = ""
        if self.session_logger.enabled and self.session_logger.session_dir is not None:
            transcript = str(self.session_logger.messages_filepath.resolve())
        return HookSessionContext(
            session_id=self.session_id,
            transcript_path=transcript,
            cwd=str(Path.cwd().resolve()),
            parent_session_id=self.parent_session_id,
        )

    # ------------------------------------------------------------------
    # Hook runners
    # ------------------------------------------------------------------

    async def _run_post_agent_turn_hooks(
        self,
    ) -> AsyncGenerator[HookEvent | HookUserMessage]:
        if not self._hooks_manager:
            return
        invocation = PostAgentTurnInvocation(
            **self._hook_session_context().model_dump()
        )
        async for ev in self._hooks_manager.run(invocation):
            if isinstance(ev, (HookEvent, HookUserMessage)):
                yield ev

    async def _run_before_tool_hooks(
        self, tool_call: ResolvedToolCall, tool_input: dict[str, Any]
    ) -> AsyncGenerator[HookEvent | HookToolDenial | HookToolInputRewrite]:
        if not self._hooks_manager:
            return
        invocation = BeforeToolInvocation(
            **self._hook_session_context().model_dump(),
            tool_name=tool_call.tool_name,
            tool_call_id=tool_call.call_id,
            tool_input=tool_input,
        )
        async for ev in self._hooks_manager.run(invocation):
            if isinstance(ev, (HookEvent, HookToolDenial, HookToolInputRewrite)):
                yield ev

    async def _run_after_tool_hooks(
        self,
        tool_call: ResolvedToolCall,
        *,
        tool_input: dict[str, Any],
        tool_status: ToolStatus,
        tool_output: dict[str, Any] | None = None,
        tool_error: str | None = None,
        duration_ms: float = 0.0,
        initial_text: str = "",
    ) -> AsyncGenerator[HookEvent | HookTextReplacement]:
        if not self._hooks_manager:
            return
        invocation = AfterToolInvocation(
            **self._hook_session_context().model_dump(),
            tool_name=tool_call.tool_name,
            tool_call_id=tool_call.call_id,
            tool_input=tool_input,
            tool_status=tool_status,
            tool_output=tool_output,
            tool_output_text=initial_text,
            tool_error=tool_error,
            duration_ms=duration_ms,
        )
        async for ev in self._hooks_manager.run(invocation):
            if isinstance(ev, (HookEvent, HookTextReplacement)):
                yield ev

    # ------------------------------------------------------------------
    # Lifecycle hook runners (user_prompt_submit / session_start /
    # session_end / pre_compact / post_compact / stop_failure /
    # notification)
    # ------------------------------------------------------------------

    async def _run_user_prompt_submit_hooks(
        self,
        prompt: str,
        *,
        message_id: str | None = None,
        project: str | None = None,
    ) -> AsyncGenerator[HookEvent | HookContextInjection | HookPromptDenial]:
        if not self._hooks_manager:
            return
        invocation = UserPromptSubmitInvocation(
            **self._hook_session_context().model_dump(),
            prompt=prompt,
            message_id=message_id,
            project=project,
        )
        async for ev in self._hooks_manager.run(invocation):
            if isinstance(ev, (HookEvent, HookContextInjection, HookPromptDenial)):
                yield ev

    async def _run_session_start_hooks(
        self, source: str
    ) -> AsyncGenerator[HookEvent | HookContextInjection]:
        if not self._hooks_manager:
            return
        invocation = SessionStartInvocation(
            **self._hook_session_context().model_dump(),
            source=source,
        )
        async for ev in self._hooks_manager.run(invocation):
            if isinstance(ev, (HookEvent, HookContextInjection)):
                yield ev

    async def _run_pre_compact_hooks(
        self,
        *,
        reason: str = "auto_compact",
        token_estimate_before: int | None = None,
        auto_compact_threshold: int | None = None,
    ) -> AsyncGenerator[HookEvent | HookContextInjection]:
        if not self._hooks_manager:
            return
        invocation = PreCompactInvocation(
            **self._hook_session_context().model_dump(),
            reason=reason,
            token_estimate_before=token_estimate_before,
            auto_compact_threshold=auto_compact_threshold,
        )
        async for ev in self._hooks_manager.run(invocation):
            if isinstance(ev, (HookEvent, HookContextInjection)):
                yield ev

    async def _run_post_compact_hooks(
        self,
        *,
        summary_text: str,
        reason: str = "auto_compact",
        token_estimate_before: int | None = None,
    ) -> AsyncGenerator[HookEvent | HookContextInjection]:
        if not self._hooks_manager:
            return
        invocation = PostCompactInvocation(
            **self._hook_session_context().model_dump(),
            reason=reason,
            summary_text=summary_text,
            token_estimate_before=token_estimate_before,
        )
        async for ev in self._hooks_manager.run(invocation):
            if isinstance(ev, (HookEvent, HookContextInjection)):
                yield ev

    async def _run_session_end_hooks(
        self,
        *,
        reason: str,
        turn_count: int,
        error: str | None = None,
    ) -> AsyncGenerator[HookEvent]:
        if not self._hooks_manager:
            return
        invocation = SessionEndInvocation(
            **self._hook_session_context().model_dump(),
            reason=reason,
            turn_count=turn_count,
            error=error,
        )
        async for ev in self._hooks_manager.run(invocation):
            if isinstance(ev, HookEvent):
                yield ev

    async def _run_stop_failure_hooks(
        self,
        *,
        error_type: str,
        error_message: str,
        turn_count: int,
    ) -> AsyncGenerator[HookEvent]:
        if not self._hooks_manager:
            return
        invocation = StopFailureInvocation(
            **self._hook_session_context().model_dump(),
            error_type=error_type,
            error_message=error_message[:_STOP_FAILURE_MESSAGE_MAX_CHARS],
            turn_count=turn_count,
        )
        async for ev in self._hooks_manager.run(invocation):
            if isinstance(ev, HookEvent):
                yield ev

    async def _run_notification_hooks(
        self,
        *,
        notification_type: str,
        message: str,
        tool_name: str | None = None,
        tool_call_id: str | None = None,
    ) -> AsyncGenerator[HookEvent]:
        if not self._hooks_manager:
            return
        invocation = NotificationInvocation(
            **self._hook_session_context().model_dump(),
            notification_type=notification_type,
            message=message,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
        )
        async for ev in self._hooks_manager.run(invocation):
            if isinstance(ev, HookEvent):
                yield ev

    async def _run_prompt_lifecycle_hooks(
        self, user_msg: str, message_id: str | None
    ) -> AsyncGenerator[BaseEvent]:
        """Fire session_start (once per session) then user_prompt_submit.

        Injected context is appended to the conversation as user messages.
        On a user_prompt_submit denial, the reason is surfaced as an
        ``AssistantEvent`` and ``_prompt_blocked`` is set so the caller can
        abort the turn before the model runs.
        """
        from vibe.core.types import AssistantEvent, LLMMessage, Role

        if self._pending_session_start_source is not None:
            source = self._pending_session_start_source
            self._pending_session_start_source = None
            async for ev in self._run_session_start_hooks(source):
                if isinstance(ev, HookContextInjection):
                    self.messages.append(
                        LLMMessage(role=Role.user, content=ev.content, injected=True)
                    )
                else:
                    yield ev

        async for ev in self._run_user_prompt_submit_hooks(
            user_msg, message_id=message_id, project=Path.cwd().name
        ):
            if isinstance(ev, HookPromptDenial):
                self._prompt_blocked = True
                yield AssistantEvent(content=ev.reason)
                return
            if isinstance(ev, HookContextInjection):
                self.messages.append(
                    LLMMessage(role=Role.user, content=ev.content, injected=True)
                )
            else:
                yield ev

    # ------------------------------------------------------------------
    # After-tool collection helpers
    # ------------------------------------------------------------------

    async def _collect_after_tool_events(
        self, tool_call: ResolvedToolCall, **kwargs: Any
    ) -> tuple[str, list[HookEvent]]:
        """List-returning variant for shielded paths (cancel / exception)
        where an async generator cannot be iterated inline.
        """
        final_text: str = kwargs.get("initial_text", "")
        events: list[HookEvent] = []
        async for ev in self._run_after_tool_hooks(tool_call, **kwargs):
            if isinstance(ev, HookTextReplacement):
                final_text = ev.text
            elif isinstance(ev, HookEvent):
                events.append(ev)
        return final_text, events

    async def _run_after_tool_and_finalize(
        self,
        tool_call: ResolvedToolCall,
        *,
        tool_input: dict[str, Any],
        tool_status: ToolStatus,
        response_status: Literal["success", "failure", "skipped"],
        decision: ToolDecision | None = None,
        span: trace.Span,
        tool_output: dict[str, Any] | None = None,
        tool_error: str | None = None,
        duration_ms: float = 0.0,
        initial_text: str = "",
    ) -> AsyncGenerator[HookEvent]:
        """Run after-tool hooks, apply text replacements, and record the response.

        Yields ``HookEvent`` instances for the caller to forward to the UI.
        The final text (after any ``HookTextReplacement``) is passed to
        ``_handle_tool_response`` together with the given *response_status*
        and *decision*.
        """
        final_text = initial_text
        async for ev in self._run_after_tool_hooks(
            tool_call,
            tool_input=tool_input,
            tool_status=tool_status,
            tool_output=tool_output,
            tool_error=tool_error,
            duration_ms=duration_ms,
            initial_text=initial_text,
        ):
            if isinstance(ev, HookTextReplacement):
                final_text = ev.text
            else:
                yield ev
        self._handle_tool_response(
            tool_call, final_text, response_status, decision, tool_output, span=span
        )

    # ------------------------------------------------------------------
    # Before-tool pipeline
    # ------------------------------------------------------------------

    async def _run_before_tool_pipeline(
        self,
        tool_call: ResolvedToolCall,
        tool_input: dict[str, Any],
        *,
        span: trace.Span,
    ) -> tuple[list[HookEvent], _BeforeToolResolution]:
        """Validate each rewrite as it arrives; first invalid one aborts the chain.

        Events are buffered (not streamed) because before_tool hooks are
        gating checks expected to complete quickly.
        """
        events: list[HookEvent] = []
        async for ev in self._run_before_tool_hooks(tool_call, tool_input):
            if isinstance(ev, HookToolDenial):
                return events, _BeforeToolResolution(
                    tool_call=tool_call,
                    tool_input=tool_input,
                    denial_event=self._handle_before_tool_denial(
                        tool_call, ev, span=span
                    ),
                )
            if isinstance(ev, HookToolInputRewrite):
                rewritten = self._apply_tool_input_rewrite(tool_call, ev)
                if isinstance(rewritten, HookToolDenial):
                    return events, _BeforeToolResolution(
                        tool_call=tool_call,
                        tool_input=tool_input,
                        denial_event=self._handle_before_tool_denial(
                            tool_call, rewritten, span=span
                        ),
                    )
                tool_call, tool_input = rewritten
                continue
            events.append(ev)

        return events, _BeforeToolResolution(
            tool_call=tool_call, tool_input=tool_input, denial_event=None
        )

    def _apply_tool_input_rewrite(
        self, tool_call: ResolvedToolCall, rewrite: HookToolInputRewrite
    ) -> tuple[ResolvedToolCall, dict[str, Any]] | HookToolDenial:
        """Re-validate a rewrite against the tool's args model.

        Rebuilds ``ResolvedToolCall``, patches the assistant message so the
        LLM sees the rewritten args next turn.  Returns a synthesized
        denial on validation failure.
        """
        tool_class = tool_call.tool_class
        args_model, _ = tool_class._get_tool_args_results()
        try:
            new_validated = args_model.model_validate(rewrite.tool_input)
        except ValidationError as e:
            logger.warning(
                "Hook %s produced invalid tool_input for '%s': %s",
                rewrite.hook_name,
                tool_call.tool_name,
                e,
            )
            return HookToolDenial(
                hook_name=rewrite.hook_name,
                content=(
                    f"Hook '{rewrite.hook_name}' rewrote tool_input but the"
                    f" result failed validation against"
                    f" {tool_call.tool_name}: {e}"
                ),
            )

        new_tool_call = tool_call.model_copy(update={"validated_args": new_validated})
        new_tool_input = self._serialize_tool_input(new_tool_call)
        self._patch_assistant_tool_call_args(tool_call.call_id, new_tool_input)
        return new_tool_call, new_tool_input

    def _patch_assistant_tool_call_args(
        self, call_id: str, new_args: dict[str, Any]
    ) -> None:
        """Mutate the assistant message's tool_calls so the transcript reflects
        what the tool actually ran with (not the model's original args).
        """
        if not call_id:
            return
        encoded = json.dumps(new_args)
        for message in reversed(self.messages):
            if not message.tool_calls:
                continue
            for tc in message.tool_calls:
                if tc.id == call_id:
                    tc.function.arguments = encoded
                    return

    def _handle_before_tool_denial(
        self, tool_call: ResolvedToolCall, denial: HookToolDenial, *, span: trace.Span
    ) -> ToolResultEvent:
        self.stats.tool_calls_hook_denied += 1
        denial_text = (
            f"<{TOOL_ERROR_TAG}>Tool '{tool_call.tool_name}' was denied by "
            f"hook '{denial.hook_name}': {denial.content}</{TOOL_ERROR_TAG}>"
        )
        self._handle_tool_response(tool_call, denial_text, "skipped", None, span=span)
        return ToolResultEvent(
            tool_name=tool_call.tool_name,
            tool_class=tool_call.tool_class,
            skipped=True,
            skip_reason=denial_text,
            cancelled=False,
            tool_call_id=tool_call.call_id,
        )

    # ------------------------------------------------------------------
    # Skip / cancel helpers
    # ------------------------------------------------------------------

    async def _handle_tool_skip(
        self, tool_call: ResolvedToolCall, decision: ToolDecision, *, span: trace.Span
    ) -> AsyncGenerator[ToolResultEvent | HookEvent]:
        self.stats.tool_calls_rejected += 1
        skip_reason = decision.feedback or str(
            get_user_cancellation_message(
                CancellationReason.TOOL_SKIPPED, tool_call.tool_name
            )
        )
        yield ToolResultEvent(
            tool_name=tool_call.tool_name,
            tool_class=tool_call.tool_class,
            skipped=True,
            skip_reason=skip_reason,
            cancelled=f"<{CANCELLATION_TAG}>" in skip_reason,
            tool_call_id=tool_call.call_id,
        )
        self._handle_tool_response(
            tool_call, skip_reason, "skipped", decision, span=span
        )

    async def _finalize_cancelled_tool(
        self,
        tool_call: ResolvedToolCall,
        tool_input: dict[str, Any],
        decision: ToolDecision | None,
        cancel_text: str,
        *,
        span: trace.Span,
        tool_started: bool,
    ) -> AsyncGenerator[HookEvent]:
        """Shield after-tool hooks from cancellation so audit/redaction hooks
        still observe the cancelled call.  Yields ``HookEvent`` instances.

        Skips after_tool entirely when ``tool_started`` is False (cancel
        landed before the tool body ran — e.g. during the approval prompt).
        That matches the before_tool denial path, which also doesn't fire
        after_tool: hooks never observe a phantom completion for a tool
        that never executed.
        """
        if not tool_started:
            self._handle_tool_response(
                tool_call, cancel_text, "failure", decision, span=span
            )
            return
        try:
            final_text, hook_events = await asyncio.shield(
                self._collect_after_tool_events(
                    tool_call,
                    tool_input=tool_input,
                    tool_status="cancelled",
                    tool_error=cancel_text,
                    initial_text=cancel_text,
                )
            )
            for ev in hook_events:
                yield ev
            self._handle_tool_response(
                tool_call, final_text, "failure", decision, span=span
            )
        except asyncio.CancelledError:
            self._handle_tool_response(
                tool_call, cancel_text, "failure", decision, span=span
            )

    # ------------------------------------------------------------------
    # Post-turn hook dispatch
    # ------------------------------------------------------------------

    async def _dispatch_post_turn_hooks(
        self,
    ) -> tuple[LLMMessage | None, list[HookEvent]]:
        """Run post-agent-turn hooks and separate retry injection from events.

        Returns a ``(retry_message, events)`` tuple.  ``retry_message`` is
        an injected ``LLMMessage`` when a hook requests a retry, else ``None``.
        """
        from vibe.core.types import LLMMessage, Role

        events: list[HookEvent] = []
        retry_msg: LLMMessage | None = None
        async for hook_event in self._run_post_agent_turn_hooks():
            if isinstance(hook_event, HookUserMessage):
                retry_msg = LLMMessage(
                    role=Role.user, content=hook_event.content, injected=True
                )
            else:
                events.append(hook_event)
        return retry_msg, events
