from __future__ import annotations

from enum import auto
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from vibe.core.types import BaseEvent, StrEnum

# --- Types & enums ---


class HookMessageSeverity(StrEnum):
    OK = auto()
    WARNING = auto()
    ERROR = auto()


class HookType(StrEnum):
    POST_AGENT_TURN = auto()
    USER_PROMPT_SUBMIT = auto()
    PRE_COMPACT = auto()
    SESSION_START = auto()
    SESSION_END = auto()
    POST_TOOL_USE = auto()
    PRE_TOOL_USE = auto()


# --- Declarative hook config (TOML on disk) ---


class HookConfig(BaseModel):
    name: str
    type: HookType
    command: str
    timeout: float = 30.0
    description: str | None = None

    @field_validator("command")
    @classmethod
    def command_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("command must not be empty")
        return v


class HookConfigIssue(BaseModel):
    file: Path
    message: str


class HookConfigResult(BaseModel):
    hooks: list[HookConfig]
    issues: list[HookConfigIssue]


# --- Subprocess execution ---


class HookInvocation(BaseModel):
    session_id: str
    transcript_path: str
    cwd: str
    hook_event_name: str
    timestamp: str | None = None
    vibe_version: str | None = None
    # user_prompt_submit fields
    prompt: str | None = None
    message_id: str | None = None
    project: str | None = None
    # session_start fields
    source: str | None = None
    parent_session_id: str | None = None
    # pre_compact fields
    reason: str | None = None
    token_estimate_before: int | None = None
    auto_compact_threshold: int | None = None
    # post_tool_use fields
    tool_name: str | None = None
    tool_call_id: str | None = None
    tool_input: dict[str, Any] | None = None
    tool_result: Any | None = None
    tool_error: str | None = None
    exit_code: int | None = None
    duration_ms: int | None = None
    # session_end fields
    turn_count: int | None = None
    error: str | None = None


# --- Hook result parsing ---


class HookDecision(BaseModel):
    decision: Literal["allow", "deny", "inject", "rewrite", "retry"] = "allow"
    reason: str | None = None
    additional_context: str | None = None
    updated_input: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class HookInjectedContext(BaseModel):
    content: str


class HookDenied(BaseModel):
    reason: str | None = None


class HookExecutionResult(BaseModel):
    hook_name: str
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool


# --- Injected user message (retry / hook stdout) ---


class HookUserMessage(BaseModel):
    content: str


# --- Transcript / UI events (BaseEvent) ---


class HookEvent(BaseEvent):
    pass


class HookRunStartEvent(HookEvent):
    pass


class HookRunEndEvent(HookEvent):
    pass


class HookStartEvent(HookEvent):
    hook_name: str


class HookEndEvent(HookEvent):
    hook_name: str
    status: HookMessageSeverity
    content: str | None = None
