from __future__ import annotations

import logging

from vibe.core.hooks._handler import HookHandler, HookRetryState, _HookAction
from vibe.core.hooks.config import HookConfig
from vibe.core.hooks.models import (
    HookEndEvent,
    HookInvocation,
    HookMessageSeverity,
    HookStructuredResponse,
)

logger = logging.getLogger(__name__)


class StopFailureHandler(HookHandler):
    """Runs when an agent turn aborts with an error. Purely observational:
    it cannot retry the turn or inject context (the original error is
    re-raised unchanged after the hooks ran). deny and any
    ``additional_context`` are logged and ignored.
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
            "Hook %s: stop_failure cannot be denied; ignoring deny", hook.name
        )
        return _HookAction(
            events=[
                HookEndEvent(
                    hook_name=hook.name,
                    status=HookMessageSeverity.WARNING,
                    content="deny ignored (stop_failure is observational)",
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
        return
