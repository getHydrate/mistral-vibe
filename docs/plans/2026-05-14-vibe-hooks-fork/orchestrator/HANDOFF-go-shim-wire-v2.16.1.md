# HANDOFF — Go shim lockstep: v2.16.1 wire-format reconciliation

**Repo to change:** `/Users/seamus_waldron/Documents/Dev/go/hydrate`
(NOT mistral-vibe). Do this on a fresh branch off a clean tree — the
working tree was dirty on `feat/peernet-v2-step3-identity` at dispatch
time; stash/branch appropriately so this change lands isolated.

**Trigger:** mistral-vibe fork reconciled v2.12.1 → upstream **v2.16.1**
(commit `6770f4a` on branch `reconcile-v2.16.1`). v2.16.1 rewrote the
hooks subsystem and enforces a **strict JSON stdout contract**. The
current shims speak the old fork wire format and will now be treated as
`HookOutputError` (fail-open warning, no injection). Canon:
`wire-format-user-prompt-submit` (superseded) and
`wire-format-lifecycle-v2.16.1` (new) under orchestration
`vibe-hooks-fork-2026-05-14`.

## The wire-format delta

### 1. Injection output shape (vibe-context, vibe-session-start, vibe-precompact)

OLD (no longer accepted):
```json
{"decision":"inject","additional_context":"..."}
```

NEW (v2.16.1):
```json
{"decision":"allow","hook_specific_output":{"additional_context":"..."}}
```

Two changes: `decision` value `"inject"` → `"allow"` (only `allow`/`deny`
are valid now), and `additional_context` moves from top-level into a
nested `hook_specific_output` object. Concretely, in each shim replace:

```go
type vibeOutput struct {
    Decision          string `json:"decision"`
    AdditionalContext string `json:"additional_context"`
}
// ...
out := vibeOutput{Decision: "inject", AdditionalContext: rendered}
```

with:

```go
type hookSpecificOutput struct {
    AdditionalContext string `json:"additional_context,omitempty"`
}
type vibeOutput struct {
    Decision           string             `json:"decision"`
    HookSpecificOutput hookSpecificOutput `json:"hook_specific_output"`
}
// ...
out := vibeOutput{
    Decision:           "allow",
    HookSpecificOutput: hookSpecificOutput{AdditionalContext: rendered},
}
```

Update each shim's `main_test.go` golden JSON assertions to the nested
shape. Files: `cmd/vibe-context/main.go`, `cmd/vibe-session-start/main.go`,
`cmd/vibe-precompact/main.go` (+ their `_test.go`).

### 2. Deny output (vibe-context, vibe-tool-pre)

`{"decision":"deny","reason":"..."}` is **unchanged** — `reason` stays
top-level in v2.16.1. before_tool deny may also still use exit code 2.
No change needed beyond the event-name rename below.

### 3. Tool hooks renamed + repayloaded (vibe-tool-pre, vibe-tool-post)

The fork **dropped** `pre_tool_use` / `post_tool_use`; v2.16.1 ships
these natively as `before_tool` / `after_tool`. Update:

- **hooks.toml registration**: `type = "pre_tool_use"` → `"before_tool"`,
  `type = "post_tool_use"` → `"after_tool"`. (Wherever Hydrate writes the
  Vibe `~/.vibe/hooks.toml` / project hooks.toml — grep for the old type
  strings.)
- **vibe-tool-pre** stdin `vibeInput`: `before_tool` payload =
  `{tool_name, tool_call_id, tool_input}` (+ session ctx). Largely
  compatible; drop any reliance on fields no longer sent.
- **vibe-tool-post** stdin `vibeInput`: `after_tool` payload changed from
  `{tool_result, tool_error, exit_code, duration_ms}` to
  `{tool_name, tool_call_id, tool_input, tool_status (success|failure|cancelled),
  tool_output (dict|null), tool_output_text (string), tool_error, duration_ms}`.
  Update the input struct + any logic that read `tool_result`/`exit_code`.
  after_tool injection (if any) also uses the nested `hook_specific_output.additional_context`
  shape from §1.

### 4. Removed input fields (all shims)

v2.16.1 invocations no longer carry `timestamp` or `vibe_version`. If any
shim reads them, make them optional/ignored. Session context now includes
`parent_session_id`. Unknown fields are tolerated, so additive-only.

## Acceptance

- `go test ./cmd/vibe-context/... ./cmd/vibe-session-start/... ./cmd/vibe-precompact/... ./cmd/vibe-tool-pre/... ./cmd/vibe-tool-post/...` green.
- A manual round-trip: pipe a sample v2.16.1 invocation JSON into each
  built shim; confirm stdout is either empty or matches the new strict
  schema (validate against `vibe/core/hooks/_handler.py::_parse_structured_response`
  in the fork — non-conforming stdout is a warning).
- `hooks.toml` type strings updated; no remaining `pre_tool_use` /
  `post_tool_use` / `"decision":"inject"` literals in `cmd/vibe-*`.

## Reference

- Fork contract: mistral-vibe `reconcile-v2.16.1` commit `6770f4a`,
  `vibe/core/hooks/models.py` (HookStructuredResponse, HookSpecificOutput),
  `README.md` → "Hooks (Experimental)" + "Lifecycle hooks (Hydrate fork)".
- Canon: `hydrate orchestrator get --in vibe-hooks-fork-2026-05-14 wire-format-lifecycle-v2.16.1`.
