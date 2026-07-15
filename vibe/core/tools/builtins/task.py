from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import aclosing, suppress
import fnmatch

from pydantic import BaseModel, Field

from vibe.core.agent_loop import AgentLoop
from vibe.core.agents.models import AgentType, BuiltinAgentName
from vibe.core.config import SessionLoggingConfig, VibeConfig
from vibe.core.hooks.models import HookContextInjection, HookEvent
from vibe.core.tools.base import (
    BaseTool,
    BaseToolConfig,
    BaseToolState,
    InvokeContext,
    ToolError,
    ToolPermission,
)
from vibe.core.tools.permissions import PermissionContext
from vibe.core.tools.ui import (
    ToolCallDisplay,
    ToolResultDisplay,
    ToolUIData,
    ToolUIDataAdapter,
)
from vibe.core.types import (
    AssistantEvent,
    LLMMessage,
    Role,
    ToolCallEvent,
    ToolResultEvent,
    ToolStreamEvent,
)


async def _drain_hook_events(events: AsyncGenerator[HookEvent]) -> None:
    """Consume a hook-runner generator for its side effects. The task tool
    has no way to surface hook UI events mid-tool-call, so they are
    drained and discarded.
    """
    async for _ in events:
        pass


class TaskArgs(BaseModel):
    task: str = Field(description="The task for the agent to perform")
    agent: str = Field(
        default="explore",
        description="The type of specialized subagent to use for this task",
    )


class TaskResult(BaseModel):
    response: str = Field(description="The accumulated response from the subagent")
    turns_used: int = Field(description="Number of turns the subagent used")
    completed: bool = Field(description="Whether the task completed normally")


class TaskToolConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK
    allowlist: list[str] = Field(default=[BuiltinAgentName.EXPLORE])


class Task(
    BaseTool[TaskArgs, TaskResult, TaskToolConfig, BaseToolState],
    ToolUIData[TaskArgs, TaskResult],
):
    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        args = event.args
        if isinstance(args, TaskArgs):
            return ToolCallDisplay(summary=f"Running {args.agent} agent: {args.task}")
        return ToolCallDisplay(summary="Running subagent")

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        result = event.result
        if isinstance(result, TaskResult):
            turn_word = "turn" if result.turns_used == 1 else "turns"
            if not result.completed:
                return ToolResultDisplay(
                    success=False,
                    message=f"Agent interrupted after {result.turns_used} {turn_word}",
                )
            return ToolResultDisplay(
                success=True,
                message=f"Agent completed in {result.turns_used} {turn_word}",
            )
        return ToolResultDisplay(success=True, message="Agent completed")

    @classmethod
    def get_status_text(cls) -> str:
        return "Running subagent"

    def resolve_permission(self, args: TaskArgs) -> PermissionContext | None:
        agent_name = args.agent

        for pattern in self.config.denylist:
            if fnmatch.fnmatch(agent_name, pattern):
                return PermissionContext(permission=ToolPermission.NEVER)

        for pattern in self.config.allowlist:
            if fnmatch.fnmatch(agent_name, pattern):
                return PermissionContext(permission=ToolPermission.ALWAYS)

        return None

    async def run(
        self, args: TaskArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[ToolStreamEvent | TaskResult, None]:
        if not ctx or not ctx.agent_manager:
            raise ToolError("Task tool requires agent_manager in context")

        agent_manager = ctx.agent_manager

        try:
            agent_profile = agent_manager.get_agent(args.agent)
        except ValueError as e:
            raise ToolError(f"Unknown agent: {args.agent}") from e

        if agent_profile.agent_type != AgentType.SUBAGENT:
            raise ToolError(
                f"Agent '{args.agent}' is a {agent_profile.agent_type.value} agent. "
                f"Only subagents can be used with the task tool. "
                f"This is a security constraint to prevent recursive spawning."
            )

        subagent_loop = self._build_subagent_loop(args, ctx)

        await self._fire_subagent_start_hooks(ctx, subagent_loop, args)

        task_text = args.task
        if ctx.scratchpad_dir:
            task_text = (
                f"Scratchpad directory: {ctx.scratchpad_dir}\n"
                "You can read and write files here without permission prompts.\n\n"
                f"{args.task}"
            )

        accumulated_response: list[str] = []
        completed = True
        # Reported to the parent-side subagent_stop hooks from the finally
        # below: "success", "failure", or "cancelled".
        status = "success"
        try:
            async with aclosing(subagent_loop.act(task_text)) as events:
                async for event in events:
                    if isinstance(event, AssistantEvent) and event.content:
                        accumulated_response.append(event.content)
                        if event.stopped_by_middleware:
                            completed = False
                    elif isinstance(event, ToolResultEvent):
                        if event.skipped:
                            completed = False
                        elif event.result and event.tool_class:
                            adapter = ToolUIDataAdapter(event.tool_class)
                            display = adapter.get_result_display(event)
                            message = f"{event.tool_name}: {display.message}"
                            yield ToolStreamEvent(
                                tool_name=self.get_name(),
                                message=message,
                                tool_call_id=ctx.tool_call_id,
                            )

            turns_used = sum(
                msg.role == Role.assistant for msg in subagent_loop.messages
            )

        except (asyncio.CancelledError, GeneratorExit):
            status = "cancelled"
            raise
        except Exception as e:
            status = "failure"
            completed = False
            accumulated_response.append(f"\n[Subagent error: {e}]")
            turns_used = sum(
                msg.role == Role.assistant for msg in subagent_loop.messages
            )
        finally:
            with suppress(Exception):
                await subagent_loop.aclose()
            await self._fire_subagent_stop_hooks(
                ctx, subagent_loop, agent_type=args.agent, status=status
            )

        yield TaskResult(
            response="".join(accumulated_response),
            turns_used=turns_used,
            completed=completed,
        )

    def _build_subagent_loop(self, args: TaskArgs, ctx: InvokeContext) -> AgentLoop:
        session_logging = SessionLoggingConfig(
            save_dir=str(ctx.session_dir / "agents") if ctx.session_dir else "",
            session_prefix=args.agent,
            enabled=ctx.session_dir is not None,
        )
        base_config = VibeConfig.load(session_logging=session_logging)
        subagent_loop = AgentLoop(
            config=base_config,
            agent_name=args.agent,
            launch_context=ctx.launch_context,
            is_subagent=True,
            defer_heavy_init=True,
            permission_store=ctx.permission_store,
            hook_config_result=ctx.hook_config_result,
        )
        if ctx.session_id:
            subagent_loop.parent_session_id = ctx.session_id

        if ctx.approval_callback:
            subagent_loop.set_approval_callback(ctx.approval_callback)
        return subagent_loop

    async def _fire_subagent_start_hooks(
        self, ctx: InvokeContext, subagent_loop: AgentLoop, args: TaskArgs
    ) -> None:
        """Parent-side subagent_start observation, fired from the parent
        loop's hooks manager (the child fires its own session events). An
        allowing hook's additional_context is injected into the child
        conversation before it runs.
        """
        if ctx.run_subagent_start_hooks is None:
            return
        async for hook_ev in ctx.run_subagent_start_hooks(
            agent_id=subagent_loop.session_id,
            agent_type=args.agent,
            task_description=args.task,
        ):
            if isinstance(hook_ev, HookContextInjection):
                subagent_loop.messages.append(
                    LLMMessage(role=Role.user, content=hook_ev.content, injected=True)
                )

    async def _fire_subagent_stop_hooks(
        self,
        ctx: InvokeContext,
        subagent_loop: AgentLoop,
        *,
        agent_type: str,
        status: str,
    ) -> None:
        """Parent-side subagent_stop observation — fires on success,
        failure, and cancellation alike. Shielded so a cancellation
        arriving during teardown cannot kill the audit; hook failures
        never mask the original outcome.
        """
        if ctx.run_subagent_stop_hooks is None:
            return
        turn_count = sum(
            msg.role == Role.assistant for msg in subagent_loop.messages
        )
        child_transcript_path = ""
        session_logger = subagent_loop.session_logger
        if session_logger.enabled and session_logger.session_dir is not None:
            child_transcript_path = str(session_logger.messages_filepath.resolve())
        try:
            await asyncio.shield(
                _drain_hook_events(
                    ctx.run_subagent_stop_hooks(
                        agent_id=subagent_loop.session_id,
                        agent_type=agent_type,
                        status=status,
                        turn_count=turn_count,
                        child_transcript_path=child_transcript_path,
                    )
                )
            )
        except (Exception, asyncio.CancelledError):
            pass
