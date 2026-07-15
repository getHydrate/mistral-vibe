from __future__ import annotations

from enum import auto
from pathlib import Path
from typing import Any, Literal, Self, assert_never

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from vibe.core.types import BaseEvent, StrEnum

# --- Types & enums ---


class HookMessageSeverity(StrEnum):
    OK = auto()
    WARNING = auto()
    ERROR = auto()


class HookType(StrEnum):
    POST_AGENT_TURN = auto()
    BEFORE_TOOL = auto()
    AFTER_TOOL = auto()
    USER_PROMPT_SUBMIT = auto()
    SESSION_START = auto()
    SESSION_END = auto()
    PRE_COMPACT = auto()
    POST_COMPACT = auto()
    STOP_FAILURE = auto()
    NOTIFICATION = auto()
    PERMISSION_REQUEST = auto()


# Tool hooks accept ``match`` / ``strict``; the lifecycle hooks below do not.
_TOOL_HOOK_TYPES = frozenset(
    {HookType.BEFORE_TOOL, HookType.AFTER_TOOL, HookType.PERMISSION_REQUEST}
)


ToolStatus = Literal["success", "failure", "cancelled"]


_DEFAULT_HOOK_TIMEOUT = 60.0


# --- Declarative hook config (TOML on disk) ---


class HookConfig(BaseModel):
    name: str
    type: HookType
    command: str
    match: str | None = None
    timeout: float | None = None
    strict: bool = False
    description: str | None = None

    @field_validator("command")
    @classmethod
    def command_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("command must not be empty")
        return v

    @field_validator("match")
    @classmethod
    def match_not_blank(cls, v: str | None) -> str | None:
        if v is not None and not v.strip():
            raise ValueError("match must not be empty")
        return v

    @model_validator(mode="after")
    def _apply_defaults_and_constraints(self) -> Self:
        if self.match is not None and self.type not in _TOOL_HOOK_TYPES:
            raise ValueError(
                "match is only valid for tool hooks"
                " (before_tool / after_tool / permission_request)"
            )
        if self.strict and self.type not in _TOOL_HOOK_TYPES:
            raise ValueError(
                "strict is only valid for tool hooks"
                " (before_tool / after_tool / permission_request)"
            )
        if self.timeout is None:
            self.timeout = _DEFAULT_HOOK_TIMEOUT
        return self


class HookConfigIssue(BaseModel):
    file: Path
    message: str


class HookConfigResult(BaseModel):
    hooks: list[HookConfig]
    issues: list[HookConfigIssue]


# --- Subprocess execution ---


class HookSessionContext(BaseModel):
    """Shared session fields passed to every hook invocation."""

    session_id: str
    transcript_path: str
    cwd: str
    parent_session_id: str | None = None


class PostAgentTurnInvocation(HookSessionContext):
    hook_event_name: Literal[HookType.POST_AGENT_TURN] = HookType.POST_AGENT_TURN


class BeforeToolInvocation(HookSessionContext):
    hook_event_name: Literal[HookType.BEFORE_TOOL] = HookType.BEFORE_TOOL
    tool_name: str
    tool_call_id: str
    tool_input: dict[str, Any]


class AfterToolInvocation(HookSessionContext):
    hook_event_name: Literal[HookType.AFTER_TOOL] = HookType.AFTER_TOOL
    tool_name: str
    tool_call_id: str
    tool_input: dict[str, Any]
    tool_status: ToolStatus
    tool_output: dict[str, Any] | None
    tool_output_text: str
    tool_error: str | None
    duration_ms: float


class UserPromptSubmitInvocation(HookSessionContext):
    hook_event_name: Literal[HookType.USER_PROMPT_SUBMIT] = HookType.USER_PROMPT_SUBMIT
    prompt: str
    message_id: str | None = None
    project: str | None = None


class SessionStartInvocation(HookSessionContext):
    hook_event_name: Literal[HookType.SESSION_START] = HookType.SESSION_START
    # One of: "new", "continue", "clear", "fork", "resume".
    source: str


class SessionEndInvocation(HookSessionContext):
    hook_event_name: Literal[HookType.SESSION_END] = HookType.SESSION_END
    # One of: "exit", "signal", "parent_close", "error", "clear".
    reason: str
    turn_count: int
    error: str | None = None


class PreCompactInvocation(HookSessionContext):
    hook_event_name: Literal[HookType.PRE_COMPACT] = HookType.PRE_COMPACT
    reason: str = "auto_compact"
    token_estimate_before: int | None = None
    auto_compact_threshold: int | None = None


class PostCompactInvocation(HookSessionContext):
    hook_event_name: Literal[HookType.POST_COMPACT] = HookType.POST_COMPACT
    # Mirrors the reason of the paired pre_compact invocation.
    reason: str = "auto_compact"
    summary_text: str
    token_estimate_before: int | None = None


class StopFailureInvocation(HookSessionContext):
    hook_event_name: Literal[HookType.STOP_FAILURE] = HookType.STOP_FAILURE
    # One of: "rate_limit", "overloaded", "authentication_failed",
    # "billing_error", "invalid_request", "model_not_found", "server_error",
    # "max_output_tokens", "context_too_long", "unknown".
    error_type: str
    # Truncated to 2000 characters.
    error_message: str
    turn_count: int


class NotificationInvocation(HookSessionContext):
    hook_event_name: Literal[HookType.NOTIFICATION] = HookType.NOTIFICATION
    # Only "permission_prompt" today; free-form so future types slot in.
    notification_type: str
    message: str
    tool_name: str | None = None
    tool_call_id: str | None = None


class PermissionRequestInvocation(HookSessionContext):
    hook_event_name: Literal[HookType.PERMISSION_REQUEST] = (
        HookType.PERMISSION_REQUEST
    )
    tool_name: str
    tool_call_id: str
    tool_input: dict[str, Any]
    # One dict per uncovered RequiredPermission (scope / invocation_pattern
    # / session_pattern / label) — see vibe/core/tools/permissions.py.
    required_permissions: list[dict[str, Any]]


HookInvocation = (
    PostAgentTurnInvocation
    | BeforeToolInvocation
    | AfterToolInvocation
    | UserPromptSubmitInvocation
    | SessionStartInvocation
    | SessionEndInvocation
    | PreCompactInvocation
    | PostCompactInvocation
    | StopFailureInvocation
    | NotificationInvocation
    | PermissionRequestInvocation
)


def build_invocation(
    hook_type: HookType,
    ctx: HookSessionContext,
    *,
    tool_name: str | None = None,
    tool_call_id: str | None = None,
    tool_input: dict[str, Any] | None = None,
    tool_status: ToolStatus | None = None,
    tool_output: dict[str, Any] | None = None,
    tool_output_text: str = "",
    tool_error: str | None = None,
    duration_ms: float = 0.0,
) -> HookInvocation:
    """Build the right HookInvocation subclass for *hook_type*."""
    base = ctx.model_dump()
    match hook_type:
        case HookType.POST_AGENT_TURN:
            return PostAgentTurnInvocation(**base)
        case HookType.BEFORE_TOOL:
            if tool_name is None or tool_call_id is None:
                raise ValueError(
                    "tool_name and tool_call_id are required for before_tool hooks"
                )
            return BeforeToolInvocation(
                **base,
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                tool_input=tool_input or {},
            )
        case HookType.AFTER_TOOL:
            if tool_name is None or tool_call_id is None or tool_status is None:
                raise ValueError(
                    "tool_name, tool_call_id, and tool_status are required"
                    " for after_tool hooks"
                )
            return AfterToolInvocation(
                **base,
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                tool_input=tool_input or {},
                tool_status=tool_status,
                tool_output=tool_output,
                tool_output_text=tool_output_text,
                tool_error=tool_error,
                duration_ms=duration_ms,
            )
        case (
            HookType.USER_PROMPT_SUBMIT
            | HookType.SESSION_START
            | HookType.SESSION_END
            | HookType.PRE_COMPACT
            | HookType.POST_COMPACT
            | HookType.STOP_FAILURE
            | HookType.NOTIFICATION
            | HookType.PERMISSION_REQUEST
        ):
            # Lifecycle invocations (and permission_request, which carries
            # required_permissions) have event-specific required fields
            # (prompt / source / reason / turn_count / …) and are constructed
            # directly by the agent loop rather than through this factory.
            raise ValueError(
                f"{hook_type.value} invocations are constructed directly,"
                " not via build_invocation()"
            )
        case _:
            assert_never(hook_type)


class HookExecutionResult(BaseModel):
    hook_name: str
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool


# --- Structured stdout response (exit 0 + JSON) ---


class HookSpecificOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # before_tool only.
    tool_input: dict[str, Any] | None = None
    # after_tool only.
    additional_context: str | None = None


class HookStructuredResponse(BaseModel):
    """The hook spec is "exit 0 + JSON object on stdout". ``decision:
    "deny"`` has per-type effect (denial / text replacement / retry
    injection). Unknown fields at any level are tolerated.
    """

    model_config = ConfigDict(extra="ignore")

    decision: Literal["allow", "deny"] = "allow"
    reason: str | None = None
    system_message: str | None = None
    hook_specific_output: HookSpecificOutput = Field(default_factory=HookSpecificOutput)


class PermissionRequestHookResponse(HookStructuredResponse):
    """permission_request stdout schema. Unlike other hook types the
    default ``decision`` is ``"ask"``: only an explicit ``"allow"`` /
    ``"deny"`` is decisive. ``"ask"`` — or omitting the field — passes
    through to the next hook and ultimately the normal approval prompt,
    so an observational hook echoing ``{}`` can never auto-approve.
    """

    decision: Literal["allow", "deny", "ask"] = "ask"  # pyright: ignore[reportIncompatibleVariableOverride]


# --- Decision values (consumed by the agent loop) ---


class HookUserMessage(BaseModel):
    """post_agent_turn deny: ``content`` is injected as a retry user
    message.
    """

    content: str


class HookToolDenial(BaseModel):
    """before_tool deny: ``content`` becomes the tool error returned to
    the LLM.
    """

    hook_name: str
    content: str


class HookToolInputRewrite(BaseModel):
    """before_tool: one per rewriting hook in the chain. The agent loop
    validates each as it arrives — the first invalid rewrite aborts the
    chain and synthesizes a denial.
    """

    hook_name: str
    tool_input: dict[str, Any]


class HookTextReplacement(BaseModel):
    """after_tool: ``text`` is the cumulative LLM-bound output after the
    handler applied its replacement or append.
    """

    text: str


class HookContextInjection(BaseModel):
    """user_prompt_submit / session_start / pre_compact / post_compact
    allow: ``content`` (the ``hook_specific_output.additional_context`` of
    an allowing hook) is injected into the conversation as a user message.
    """

    content: str


class HookPromptDenial(BaseModel):
    """user_prompt_submit deny: the user prompt is blocked before it
    reaches the model and ``reason`` is surfaced to the user.
    """

    hook_name: str
    reason: str


class HookPermissionDecision(BaseModel):
    """permission_request: an explicit hook decision replaces the approval
    prompt. ``allow`` mirrors a user approval; ``deny`` mirrors a user
    decline (``reason`` becomes the model-visible skip feedback). The
    first decisive hook wins — the handler stops the chain (same
    first-answer-wins convention as before_tool's first deny; here either
    an allow or a deny ends the run).
    """

    hook_name: str
    decision: Literal["allow", "deny"]
    reason: str | None = None


# --- Transcript / UI events (BaseEvent) ---


class HookEvent(BaseEvent):
    pass


class HookRunStartEvent(HookEvent):
    scope: HookType = HookType.POST_AGENT_TURN
    tool_name: str | None = None
    tool_call_id: str | None = None


class HookRunEndEvent(HookEvent):
    scope: HookType = HookType.POST_AGENT_TURN
    tool_call_id: str | None = None


# scope / tool_call_id let consumers route events when concurrent tool-call
# chains interleave on the wire.
class HookStartEvent(HookEvent):
    hook_name: str
    scope: HookType = HookType.POST_AGENT_TURN
    tool_call_id: str | None = None


class HookEndEvent(HookEvent):
    hook_name: str
    status: HookMessageSeverity
    content: str | None = None
    scope: HookType = HookType.POST_AGENT_TURN
    tool_call_id: str | None = None
