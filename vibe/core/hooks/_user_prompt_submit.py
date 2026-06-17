from __future__ import annotations

import logging

from vibe.core.hooks._handler import HookHandler, HookRetryState, _HookAction
from vibe.core.hooks.config import HookConfig
from vibe.core.hooks.models import (
    HookContextInjection,
    HookEndEvent,
    HookInvocation,
    HookMessageSeverity,
    HookPromptDenial,
    HookStructuredResponse,
)

logger = logging.getLogger(__name__)


class UserPromptSubmitHandler(HookHandler):
    """Runs before the user prompt reaches the model.

    deny → ``HookPromptDenial`` (the agent loop blocks the prompt and
    surfaces ``reason`` to the user). allow → inject
    ``hook_specific_output.additional_context`` as a user message, if any.
    """

    def matches(self, hook: HookConfig, invocation: HookInvocation) -> bool:
        return True

    def _on_deny(
        self,
        hook: HookConfig,
        invocation: HookInvocation,
        response: HookStructuredResponse,
        retry_state: HookRetryState,
    ) -> _HookAction:
        reason = response.reason or ""
        return _HookAction(
            events=[
                HookEndEvent(
                    hook_name=hook.name,
                    status=HookMessageSeverity.ERROR,
                    content="Blocked user prompt",
                ),
                HookPromptDenial(hook_name=hook.name, reason=reason),
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
        events: list = [
            HookEndEvent(
                hook_name=hook.name,
                status=HookMessageSeverity.OK,
                content=response.system_message,
            )
        ]
        context = response.hook_specific_output.additional_context
        if context is not None:
            events.append(HookContextInjection(content=context))
        return _HookAction(events=events, next_invocation=None, should_break=False)

    def on_passthrough(self, hook: HookConfig, retry_state: HookRetryState) -> None:
        return
