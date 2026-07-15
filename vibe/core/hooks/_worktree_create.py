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


class WorktreeCreateHandler(HookHandler):
    """Runs once per session inside a ``--worktree`` checkout. Purely
    observational: the worktree was prepared by the CLI entrypoint before
    the agent loop existed, so by the time the hook fires the checkout is
    a fait accompli. deny and any ``additional_context`` are logged and
    ignored.
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
            "Hook %s: worktree_create cannot be denied; ignoring deny", hook.name
        )
        return _HookAction(
            events=[
                HookEndEvent(
                    hook_name=hook.name,
                    status=HookMessageSeverity.WARNING,
                    content="deny ignored (the worktree already exists)",
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
