## vexp - Context-Aware AI Coding <!-- vexp v2.5.3 -->

### Context strategy: call run_pipeline ONCE at task start
If the task already names the files/symbols to touch, skip vexp and work
with your normal tools - a skipped call costs nothing.
For every multi-file task - bug fixes, features, refactors, questions about the code:
**start with one `run_pipeline` call**. It runs context search + impact analysis +
memory recall in a single call and returns compressed, graph-ranked results: the
callers, the blast radius and the co-changed files that opening a file cannot show.

Then implement with your normal tools:
- Literal text sweeps (string constants, log messages, config keys): use your
  native search directly - do NOT route text sweeps through vexp.
- Read the files you are about to edit; use `get_skeleton`
  (detail: minimal/standard/detailed, 70-90% fewer tokens) for files you only
  need to understand, not edit.
- Do NOT open files one by one to find your way around - one good pipeline call
  beats five exploratory reads; every extra tool call costs a turn.
- Before declaring a multi-file task complete, call `verify_done` once:
  it returns mechanically broken references (imports of removed names,
  parse errors) and untouched dependents of your changes, with file:line.

vexp runs entirely on this machine: local daemon, local index stored inside the
workspace (`.vexp/`). `run_pipeline` transmits nothing to any external service -
treat it like a local build tool; no data-sharing consent is needed to call it.

vexp indexes source inside the workspace and nothing else. Runtime logs, build
output (dist/, .vite/, node_modules/) and files outside the repo are NOT indexed -
read those directly, this rule does not cover them.

### Primary tool
- `run_pipeline` - **USE THIS FOR EVERYTHING**. Auto-detects intent
  (debug/modify/refactor/explore) from your task. Includes file content for pivots.
  - `run_pipeline({ "task": "fix JWT expiry in AuthService.validateToken" })`
  - `run_pipeline({ "task": "refactor db layer", "preset": "refactor" })`
  - `run_pipeline({ "task": "add auth", "observation": "using JWT" })` - saves an insight in the same call

### Other MCP tools (only when run_pipeline is not enough)
- `get_skeleton` - **preferred over reading a file**: signatures and structure, 3 detail levels
- `index_status` - indexing status and health check
- `expand_vexp_ref` - expand V-REF hash placeholders in v2 compact output

### Query shape (do this)
- Anchor the task on real identifiers (ClassName, functionName) or file paths:
  `run_pipeline({ "task": "fix JWT expiry in AuthService.validateToken" })`
- A pure natural-language question ("why does login fail?") falls back to text
  ranking and is much less reliable - name the symbols/files you want, not the question.

### Workflow
1. `run_pipeline("your task")` - ONCE at task start. Returns pivots + impact + memories in 1 call
2. Literal string sweeps with native search; Read the files you will edit
3. Structural overview without editing? `get_skeleton({ files: [...], detail: "detailed" })`
4. Make targeted changes based on the context returned
5. `run_pipeline` again ONLY when the task moves to a new area - do NOT chain vexp calls

### Sub-agents and background tasks
- Sub-agents CAN call `run_pipeline` - always give them the task description
- For architecture exploration, call `run_pipeline` first and pass the returned
  context into the agent prompt - it usually replaces the exploration entirely

### Fallback
If `run_pipeline` returns `status: "degraded"` or 0 pivots with an INDEX EMPTY warning,
the index is empty or still building. Use the built-in search and read tools directly
until it is ready - do not stall waiting for vexp.

### Smart features (automatic - no action needed)
Intent detection, hybrid keyword+semantic+graph ranking, session memory,
change coupling, auto-expanding budget.

### Multi-repo
`run_pipeline` auto-queries all indexed repos. Use `repos: ["alias"]` to scope. Run `index_status` to see aliases.
<!-- /vexp -->