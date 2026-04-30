# OpenCode Viewer

A small Python/Panel dashboard for browsing an opencode SQLite database.

## Setup

```bash
uv sync
```

## Run

The repository can be run against the local `opencode/` runs folder from the project root:

```bash
uv run opencode-viewer --db ./opencode --show
```

When `--db` points at a folder, the viewer scans for `*/opencode/opencode.db`, selects the first run, and lets you switch runs from the dashboard. You can also point at a single database file:

```bash
uv run opencode-viewer --db /path/to/opencode.db --show
```

Or a folder outside the repo with the same run layout:

```bash
uv run opencode-viewer --db ../opencode --show
```

Runs currently present under `./opencode`:

- `battery-soh-pipeline-enforced-bs-0-1`
- `battnn-pinn-eod-enforced-bs-0-1`
- `battnn-pinn-eod-safety-relaxed-0-1`
- `bayesian-gated-transformer-rul`
- `cart-net-aeroengine-rul`
- `cata-tcn-engine-rul-30-04`
- `ica-transformer-soh-safety-relaxed-0-3`
- `lstm-ae-health-rul-30-04-0-1`
- `vmd-ssa-patchtst-battery-rul-light-test`
- `vmd-ssa-patchtst-battery-rul-safety-relaxed`

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
