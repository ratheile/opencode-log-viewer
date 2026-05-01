# Agent Guide

## Project Overview

OpenCode Viewer is a small Python/Panel dashboard for browsing OpenCode SQLite
session data. The app is intentionally read-only: it summarizes sessions,
messages, parts, tool calls, child sessions, workflow state, sidecar logs, stored
diffs, and raw DB tables.

Core files:

- `src/opencode_viewer/app.py` builds the Panel dashboard, CLI entry point, tab
  layout, display columns, and markdown renderers.
- `src/opencode_viewer/db.py` contains read-only SQLite access, DataFrame
  shaping, JSON parsing, sidecar file readers, and redaction logic.
- `tests/test_app.py` covers dashboard helper behavior.
- `tests/test_db.py` builds synthetic SQLite fixtures and covers store behavior.

The `opencode/` directory and `opencode.zip` are local sample data and are
ignored by git.

## Development Commands

Use `uv` for the project environment.

```bash
uv sync
uv run pytest
uv run ruff check .
```

Run the dashboard against the local sample DB:

```bash
uv run opencode-viewer --db ./opencode --show
```

Preferred local run command:

```bash
BOKEH_ALLOW_WS_ORIGIN=127.0.0.1:5006,localhost:5006 uv run opencode-viewer --db /Users/rtheiler/Workspaces/work/agentic-picid-results/opencode
```

`BOKEH_ALLOW_WS_ORIGIN` is required when Panel's WebSocket origin check would
otherwise reject the browser connection (common when running behind a proxy or
in certain terminal environments).

**`--db` path resolution** — the argument is interpreted in this order:

1. If the path is a **file**, it is used as the database directly.
2. If the path is a **directory**, it is scanned in priority order:
   - `<dir>/opencode.db` — the directory itself is a run folder
   - `<dir>/opencode/opencode.db` — one level of nesting
   - `<dir>/*/opencode/opencode.db` — each child is a run folder (multi-run root)

`OPENCODE_DB_PATH` is also supported as an environment variable fallback.

## Code Conventions

- Target Python 3.11+.
- Keep Ruff clean with the configured rules in `pyproject.toml`: `E`, `F`, `I`,
  `UP`, and `B`; line length is 100.
- Prefer small helper functions over broad refactors. This codebase is compact,
  and most behavior is easiest to verify near `app.py` or `db.py`.
- Preserve stable DataFrame column names. The Panel tables and tests depend on
  specific columns being present.
- When returning empty DataFrames from public store methods, include expected
  columns when the UI or tests rely on them.
- OpenCode JSON payloads can vary across versions. Parse defensively and keep
  fallback behavior for missing, malformed, or nested fields.
- Redaction in `raw_table` covers any column in `SENSITIVE_COLUMNS`
  (`access_token`, `refresh_token`, `secret`) **or** whose name contains the
  substring `"token"` (case-insensitive). Both conditions are checked; do not
  narrow the guard.
- Known nested JSON shapes that must remain handled:
  - Model ID: try `data["modelID"]`, fall back to `data["model"]["modelID"]`,
    then `data["model"]["id"]`, then `data["model"]["name"]`.
  - Provider ID: try `data["providerID"]`, fall back to
    `data["model"]["providerID"]`, then `data["model"]["provider"]`.
  - Cache tokens: `data["tokens"]["cache"]["read"]` / `["write"]` — `cache` is
    a nested dict, not a flat key.

## Data Safety

- Agents should actively inspect available OpenCode databases in read-only mode
  when that helps discover schema details, payload shapes, or real-world edge
  cases. The ignored local `opencode/` data is valid reference material for
  exploration.
- Treat SQLite databases and sidecar data as read-only unless the user
  explicitly asks for a data mutation.
- `OpenCodeStore.connect()` opens SQLite using `mode=ro` and enables
  `PRAGMA query_only = ON`; preserve that behavior.
- Prefer read-only database queries for discovery, for example:

```bash
sqlite3 -readonly ./opencode/opencode.db ".tables"
sqlite3 -readonly ./opencode/opencode.db "SELECT id, title FROM session LIMIT 5;"
```

- Do not commit `.venv/`, caches, `opencode/`, `opencode.zip`, or generated
  local artifacts.
- Raw table display must continue to redact token and secret-like columns.

### Sidecar File Layout

Sidecar data lives in directories adjacent to the database file. All paths are
relative to `db_path.parent`:

| Directory | Content |
| --- | --- |
| `storage/session_diff/<session_id>.json` | Per-session diff JSON arrays |
| `tool-output/<filename>` | Raw tool output files |
| `log/*.log` | Application log files |

`OpenCodeStore` exposes `storage_dir`, `log_dir`, and `tool_output_dir` properties
for these paths. All three are read-only; treat them the same as the database.

## Key Patterns

### Connection and Retry Logic

`OpenCodeStore.connect()` opens the database as:

```
file:{path}?mode=ro&cache=shared
```

followed immediately by `PRAGMA query_only = ON`. Both are required; preserve
them when modifying the connection path.

`rows()` retries up to `self.retries` (default 3) on `sqlite3.OperationalError`
containing `"locked"`, with 200 ms × attempt back-off. Do not collapse this into
a single call.

### DataFrame Column Contracts

**Sessions** (`store.sessions()`): includes aggregated columns `message_count`,
`tool_count`, `task_count`, `child_session_count`, `error_count`, `total_cost`,
and `token_total`. All are filled with `0` when missing — never `NaN`.

**Diffs** (`store.session_diff()`): always returns
`["file", "status", "additions", "deletions", "patch"]`, even when the sidecar
JSON is absent.

When adding public store methods, return an empty DataFrame with the expected
columns rather than a bare `pd.DataFrame()`.

### Task-to-Child-Session Linking

`TASK_ID_RE = re.compile(r"task_id:\s*(ses_[A-Za-z0-9]+)")` is applied to text
output of every `tool == "task"` part. The extracted session ID becomes the
`task_id` field used by `subagents()` to join the tool call to its child session
record. This is the only linking mechanism; preserve the regex and join logic
when modifying `subagents()` or `_task_id()`.

## Testing Notes

- Add focused pytest coverage for changes to DB parsing, DataFrame schemas,
  dashboard helper output, or CLI path resolution.
- Tests should create temporary SQLite databases with `tmp_path`; do not depend
  on the ignored local `opencode/` sample data.
- If changing readable markdown previews, update tests around exact strings with
  care because the UI uses those helpers directly.

## UI Notes

- The app uses Panel with Tabulator enabled through `pn.extension("tabulator")`.
- The dashboard is organized around a sidebar for database/session controls and
  tabs for summaries, conversation, tools, workflows, sidecar outputs, diffs,
  logs, and raw tables.
- Keep the UI read-only and provenance-oriented. Artifact paths found inside
  tool payloads are displayed as references; external artifact files are not
  opened.
