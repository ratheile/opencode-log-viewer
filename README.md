# OpenCode Viewer

A small Python/Panel dashboard for browsing an opencode SQLite database.

## Setup

```bash
uv sync
```

## Run

The repository includes an example database at `opencode/opencode.db`, so the default command works from the project root:

```bash
uv run opencode-viewer --show
```

To point at another database:

```bash
uv run opencode-viewer --db /path/to/opencode.db --show
```

You can also set `OPENCODE_DB_PATH`:

```bash
OPENCODE_DB_PATH=/path/to/opencode.db uv run opencode-viewer --show
```

The app reads the SQLite database in read-only mode and summarizes sessions, messages, parts, token usage, tool calls, and stored session diffs from `storage/session_diff`.

## Development

```bash
uv run pytest
uv run ruff check .
```

