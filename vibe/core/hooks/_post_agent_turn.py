from __future__ import annotations

import logging

from vibe.core.hooks._handler import (
    _MAX_RETRIES,
    HookHandler,
    HookRetryState,
    _HookAction,
)
from vibe.core.hooks.config import HookConfig
from vibe.core.hooks.models import (
    HookEndEvent,
    HookInvocation,
    HookMessageSeverity,
    HookStructuredResponse,
    HookUserMessage,
    PostAgentTurnInvocation,
)

logger = logging.getLogger(__name__)


def _as_post_agent_turn(invocation: HookInvocation) -> PostAgentTurnInvocation:
    if not isinstance(invocation, PostAgentTurnInvocation):
        raise TypeError(
            f"PostAgentTurnHandler expected PostAgentTurnInvocation, got"
            f" {type(invocation).__name__}"
        )
    return invocation


class PostAgentTurnHandler(HookHandler):
    """Deny → inject ``reason`` as a retry user message, capped at
    :data:`_MAX_RETRIES` per hook per user turn.
    """

    def matches(self, hook: HookConfig, invocation: HookInvocation) -> bool:
        return True

    def prepare_invocation(
        self,
        hook: HookConfig,
        invocation: HookInvocation,
        retry_state: HookRetryState,
    ) -> HookInvocation:
        # stop_hook_active mirrors Claude Code's Stop contract: True when
        # this run is a retry caused by this hook's own previous deny, so
        # loop-guard scripts can break deny loops.
        return _as_post_agent_turn(invocation).model_copy(
            update={"stop_hook_active": retry_state.retry_count(hook.name) > 0}
        )

    def _on_deny(
        self,
        hook: HookConfig,
        invocation: HookInvocation,
        response: HookStructuredResponse,
        retry_state: HookRetryState,
    ) -> _HookAction:
        reason = response.reason or ""
        logger.debug("Hook %s retry reason: %s", hook.name, reason)
        if not retry_state.should_retry(hook.name):
            return _HookAction(
                events=[
                    HookEndEvent(
                        hook_name=hook.name,
                        status=HookMessageSeverity.ERROR,
                        content=f"Failed, retries exhausted ({_MAX_RETRIES}/{_MAX_RETRIES})",
                    )
                ],
                next_invocation=None,
                should_break=False,
            )
        remaining = retry_state.remaining_retries(hook.name)
        retry_state.track_retry(hook.name)
        return _HookAction(
            events=[
                HookEndEvent(
                    hook_name=hook.name,
                    status=HookMessageSeverity.ERROR,
                    content=f"Failed, retrying ({remaining} {'retry' if remaining == 1 else 'retries'} remaining)",
                ),
                HookUserMessage(content=reason),
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
        retry_state.track_no_retry(hook.name)
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

    def on_passthrough(self, hook: HookConfig, retry_state: HookRetryState) -> None:
        retry_state.track_no_retry(hook.name)
