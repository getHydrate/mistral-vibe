from __future__ import annotations

from typing import ClassVar

from vibe.core.hooks._handler import (
    HookExternalAttrs,
    HookHandler,
    HookRetryState,
    _HookAction,
)
from vibe.core.hooks.config import HookConfig
from vibe.core.hooks.models import (
    HookEndEvent,
    HookInvocation,
    HookMessageSeverity,
    HookPermissionDecision,
    HookStructuredResponse,
    PermissionRequestHookResponse,
    PermissionRequestInvocation,
)
from vibe.core.utils.matching import name_matches


def _as_permission(invocation: HookInvocation) -> PermissionRequestInvocation:
    if not isinstance(invocation, PermissionRequestInvocation):
        raise TypeError(
            f"PermissionRequestHandler expected PermissionRequestInvocation,"
            f" got {type(invocation).__name__}"
        )
    return invocation


class PermissionRequestHandler(HookHandler):
    """Fires when a tool call needs user approval (permission verdict
    ASK), before the ``notification`` hook and before the approval
    prompt.

    Decision semantics (see :class:`PermissionRequestHookResponse` — the
    ``decision`` default for this hook type is ``"ask"``):

    - explicit ``allow`` → ``HookPermissionDecision`` approving on the
      user's behalf; the prompt is skipped entirely.
    - explicit ``deny`` → ``HookPermissionDecision`` declining on the
      user's behalf; ``reason`` becomes the model-visible skip feedback.
    - ``ask`` / omitted / empty stdout → passthrough to the next hook
      and ultimately the normal approval flow.

    The first hook to return an explicit decision wins — either an allow
    OR a deny stops the chain (extending before_tool's first-deny-wins
    convention to both decisive outcomes).
    """

    response_model: ClassVar[type[HookStructuredResponse]] = (
        PermissionRequestHookResponse
    )

    def matches(self, hook: HookConfig, invocation: HookInvocation) -> bool:
        return name_matches(_as_permission(invocation).tool_name, [hook.match or "*"])

    def external_attributes(self, invocation: HookInvocation) -> HookExternalAttrs:
        inv = _as_permission(invocation)
        return {"tool_name": inv.tool_name, "tool_call_id": inv.tool_call_id}

    def on_structured(
        self,
        hook: HookConfig,
        invocation: HookInvocation,
        response: HookStructuredResponse,
        retry_state: HookRetryState,
    ) -> _HookAction:
        if response.decision == "ask":
            # Explicit (or defaulted) "ask": the hook declines to decide.
            return _HookAction(
                events=[
                    HookEndEvent(
                        hook_name=hook.name,
                        status=HookMessageSeverity.OK,
                        content=response.system_message,
                    )
                ],
                next_invocation=None,
                should_break=False,
            )
        return super().on_structured(hook, invocation, response, retry_state)

    def _on_deny(
        self,
        hook: HookConfig,
        invocation: HookInvocation,
        response: HookStructuredResponse,
        retry_state: HookRetryState,
    ) -> _HookAction:
        inv = _as_permission(invocation)
        return _HookAction(
            events=[
                HookEndEvent(
                    hook_name=hook.name,
                    status=HookMessageSeverity.ERROR,
                    content=f"Denied permission for tool '{inv.tool_name}'",
                ),
                HookPermissionDecision(
                    hook_name=hook.name,
                    decision="deny",
                    reason=response.reason or "",
                ),
            ],
            next_invocation=None,
            should_break=True,
        )

    def _on_allow(
        self,
        hook: HookConfig,
        invocation: HookInvocation,
        response: HookStructuredResponse,
        retry_state: HookRetryState,
    ) -> _HookAction:
        inv = _as_permission(invocation)
        return _HookAction(
            events=[
                HookEndEvent(
                    hook_name=hook.name,
                    status=HookMessageSeverity.OK,
                    content=response.system_message
                    or f"Approved tool '{inv.tool_name}' on the user's behalf",
                ),
                HookPermissionDecision(hook_name=hook.name, decision="allow"),
            ],
            next_invocation=None,
            should_break=True,
        )

    def on_passthrough(self, hook: HookConfig, retry_state: HookRetryState) -> None:
        return

    def on_strict_failure(
        self, hook: HookConfig, invocation: HookInvocation, reason: str
    ) -> _HookAction | None:
        inv = _as_permission(invocation)
        return _HookAction(
            events=[
                HookEndEvent(
                    hook_name=hook.name,
                    status=HookMessageSeverity.ERROR,
                    content=(
                        f"Denied permission for tool '{inv.tool_name}' (strict)"
                    ),
                ),
                HookPermissionDecision(
                    hook_name=hook.name, decision="deny", reason=reason
                ),
            ],
            next_invocation=None,
            should_break=True,
        )
