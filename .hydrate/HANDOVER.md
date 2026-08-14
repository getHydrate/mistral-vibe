# Handover — mistral-vibe session 2026-07-15

**Project:** mistral-vibe
**Last touched:** 2026-07-15
**Author:** Claude Fable 5 (claude-fable-5)

## Now
The entire hook-parity programme shipped today in one session. The fork (`upstream-src/`, nested git repo) went from v2.16.1 to upstream v2.19.1, then gained all four proposal waves plus the status line. Final state: 13 hook events (upstream's post_agent_turn/before_tool/after_tool + fork's user_prompt_submit, session_start with resume source, session_end, pre_compact, post_compact, stop_failure, notification, subagent_start, subagent_stop, permission_request, worktree_create), after_tool match_status filtering, stop_hook_active + retry cap 8 (Claude Code Stop parity), and an external status line (status_line_command config) whose stdin JSON is byte-compatible with Claude Code's statusLine contract. Final suite: 5,243 passed; the only failure is tests/tools/test_grep.py latin-1 accent test, which fails identically on pristine upstream (environmental, macOS).

GitHub layout changed today: `hydrate-main` is the new default branch on getHydrate/mistral-vibe (created at user request after confusion that the repo "looked official"); `main` stays a pristine upstream mirror; `reconcile-v2.19.1` is the working branch for this upstream cycle. Both are pushed and level at `4095022`. README now leads with the fork identity (title "Mistral Vibe — Hydrate fork", IMPORTANT callout, fork install one-liner) and every fork feature section carries an "Added by the Hydrate fork" attribution. Hydrate links point at github.com/getHydrate/hydrate-public, not the website.

The May tmux fleet no longer exists; all four waves were executed by worktree-isolated general-purpose agents driven from this session, with HANDOFF briefs + hydrate orchestrator dispatch records + worker-written canon preserving the case-study trail.

## Next
1. Dispatch the Go shims in /Users/seamus_waldron/Documents/Dev/go/hydrate: vibe-postcompact, vibe-stopfailure, vibe-notification, vibe-subagent-stop, vibe-permission. Wire facts are already written (wire-format-<event> keys in orchestration vibe-hooks-fork-2026-05-14) — write NO shim before reading its fact.
2. Two one-line follow-ups in the fork: TUI /resume picker (vibe/cli/textual_ui/app.py ~2849) and ACP resume (vibe/acp/acp_agent_loop.py ~1276) don't arm session_start source=resume yet; CLI resume does.
3. Optional: hydrate install-hooks --runtime=vibe to write status_line_command into ~/.vibe/config.toml the way it writes statusLine into Claude Code settings.json.
4. Seamus drafted a Mistral Discord post (final text in this session, written via humaniser-seamus skill) — if it lands interest, first upstream PR should be user_prompt_submit + session_start/session_end.

## In-flight
- Nothing dispatched and unlanded. All four waves (A: post_compact/stop_failure/notification/resume-source; B: subagent events; C: permission_request; D: parity niceties) are merged, tested, pushed. Worktrees removed. No background tasks running.

## Decisions made this session
- Branch strategy (user chose via AskUserQuestion): hydrate-main = long-lived GitHub default, fast-forwarded from each reconcile-vX.Y.Z after integration; main = upstream mirror for diffing; never re-point the default again. Recorded in canon key branch-strategy.
- Waves B and C ran in parallel despite touching the same hook files; I resolved the 5 both-added merge conflicts myself. test_hooks.py could NOT be union-merged (git aligned conflicts mid-class because both branches' test classes share method signatures) — rebuilt the tail as base + B's tests + C's tests instead. 182 tests = exact B+C sum proved nothing was lost.
- permission_request semantics: allow/deny produce ToolDecision shapes byte-identical to user YES/decline; per-event response schema defaults decision to "ask" so {} can never auto-approve; deliberate divergence from Claude Code: fires headless too so hooks can auto-allow in -p runs.
- worktree_remove deliberately NOT implemented (documented-out): removal runs after loop teardown behind an interactive prompt — no live hook surface. worktree_create fires via a pending-registry pattern (entrypoint registers pre-loop, first main loop consumes before session_start).
- Status line stdin JSON deliberately copies Claude Code's statusLine payload keys exactly so `hydrate statusline` works on both runtimes with zero Go changes; shape asserted literally in tests as a wire contract.
- Wave A pre_compact wiring wraps the new v2.19.1 CompactionManager: hooks fire after the pre-compact save, injections re-appended after the manager's reset + saved again, session_start re-armed with source="continue".
- Process rule (canonised): write the wire-format-<event> canon fact BEFORE dispatching the matching Go shim — the never-written wire-format-pre-user-prompt fact from PR1 dangled for two months until I backfilled it today.

## References
- Fork repo: `https://github.com/getHydrate/mistral-vibe` (origin), upstream `https://github.com/mistralai/mistral-vibe`
- Hydrate public repo (README links target): `https://github.com/getHydrate/hydrate-public`
- Fork source: `/Users/seamus_waldron/Documents/Dev/python/mistral-vibe/upstream-src` (nested git repo; outer repo has no commits)
- Go shims: `/Users/seamus_waldron/Documents/Dev/go/hydrate/cmd/` (claude-*, vibe-*, codex-*); statusline: `cmd/hydrate/statusline.go`, render: `internal/statusline/render.go`
- Plans + briefs + worker reports: `docs/plans/2026-07-15-hook-parity/` (PROPOSAL.md, STATUSLINE.md, claude-code-hooks-reference.md, orchestrator/HANDOFF-*.md, reports/report-wave-*.md)
- Orchestration: `hydrate orchestrator list --in vibe-hooks-fork-2026-05-14` (ALWAYS pass --in; never join)
- Skill installed today: `~/.claude/skills/humaniser-seamus/`

## Recent commits
- 4095022 — docs: attribute the status line section to the Hydrate fork (HEAD of hydrate-main + reconcile-v2.19.1, pushed)
- 4bbb0db — docs: link Hydrate to its GitHub repo instead of the website
- 862635e — docs: lead README with the Hydrate fork identity
- a08801a / c8446ff / afe0244 / 11dc415 — Wave D (skill docs, worktree_create, stop_hook_active+cap8, match_status)
- e6c08eb — merge waves B+C; f2ffd7e — subagent events; 4eaf7e3 — permission_request
- 40493d3 — merge Wave A + status line; aed6baa — Wave A events; 71167ee — status line widget
- 348527b — merge upstream v2.19.1 into hooks fork (the morning's reconcile)

## Findings worth keeping
- Vibe AgentStats listener registry holds ONE listener per attribute (dict keyed by attr name) — a second add_listener("context_tokens", ...) silently clobbers the first. Status line piggybacks inside the existing ContextProgress closure at app.py ~830.
- Skill invocation reality (Wave D corrected my proposal): model-invoked `skill` tool calls run the tool-hook pipeline (match="skill", tool_input.name carries the skill name, deny blocks); user-typed /skill commands BYPASS tool hooks entirely — gate those via user_prompt_submit.
- tests/tools/test_grep.py::test_preserves_accents_when_matching_latin1_encoded_file fails on macOS on pristine upstream v2.19.1 — never chase it as a regression.
- Upstream squashes each release into one commit; v2.19.1 refactored agent_loop.py into agent_loop/_loop.py (git rename detection follows it) and extracted CompactionManager — the validated merge procedure is in the project CLAUDE.md "Updating from upstream".
- The claude-code-hooks reference doc (30 events, July 2026) came from a claude-code-guide subagent reading official docs; spot-check per-event output schemas before implementing anything beyond what shipped.

## Open questions
- Does the Vibe team want any of this upstream? Discord post drafted; first PR candidate is user_prompt_submit + session_start/session_end.
- Should hydrate-main get CI? The upstream ci.yml workflows exist in the fork but the badge still points at mistralai's repo.

## Pre-emptive gotchas
- Outer repo /Users/seamus_waldron/Documents/Dev/python/mistral-vibe is a git repo with NO commits; the real repo is upstream-src/. Don't git-init or commit the outer dir without deciding that deliberately.
- Project CLAUDE.md was updated today (fork state v2.19.1, one-fact-per-event wire convention, "Updating from upstream" procedure) but still describes the tmux pane fleet that no longer exists — worktree agents replaced it this session.
- hydrate distill draft's rails_draft references were garbage (truncated URLs, a sed pattern) — trimmed; don't trust its reference extraction in this repo.
- The canon fact hook-parity-complete is the single best summary key: read it first on resume.
