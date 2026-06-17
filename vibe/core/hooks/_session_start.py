from __future__ import annotations

import logging

from vibe.core.hooks._handler import HookHandler, HookRetryState, _HookAction
from vibe.core.hooks.config import HookConfig
from vibe.core.hooks.models import (
    HookContextInjection,
    HookEndEvent,
    HookInvocation,
    HookMessageSeverity,
    HookStructuredResponse,
)

logger = logging.getLogger(__name__)


class SessionStartHandler(HookHandler):
    """Runs once when a session begins (new / continue / clear / fork).

    Observational: a session start cannot be blocked. allow → inject
    ``hook_specific_output.additional_context`` as a user message, if any.
    deny is logged and ignored.
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
        logger.warning(
            "Hook %s: session_start cannot be denied; ignoring deny", hook.name
        )
        return _HookAction(
            events=[
                HookEndEvent(
                    hook_name=hook.name,
                    status=HookMessageSeverity.WARNING,
                    content="deny ignored (session_start cannot be blocked)",
                )
            ],
            next_invocation=None,
            should_break=False,
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
