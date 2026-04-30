from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd

DEFAULT_DB_PATH = Path(os.environ.get("OPENCODE_DB_PATH", "opencode/opencode.db"))
SENSITIVE_COLUMNS = {"access_token", "refresh_token", "secret"}
TASK_ID_RE = re.compile(r"task_id:\s*(ses_[A-Za-z0-9]+)")
PATH_RE = re.compile(r"(?P<path>/(?:[^\s`<>\")]+))")


def resolve_db_path(path: str | Path | None = None) -> Path:
    """Resolve a database path without requiring it to exist yet."""
    return Path(path or DEFAULT_DB_PATH).expanduser()


def _sqlite_uri(path: Path) -> str:
    quoted_path = quote(str(path.resolve()), safe="/:")
    return f"file:{quoted_path}?mode=ro&cache=shared"


def _load_json(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError:
        return {"_raw": raw}
    return loaded if isinstance(loaded, dict) else {"value": loaded}


def _dump_json(value: Any) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _format_time_ms(value: Any) -> str:
    if value in (None, ""):
        return ""
    try:
        timestamp = int(value) / 1000
    except (TypeError, ValueError):
        return ""
    return datetime.fromtimestamp(timestamp).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def _tokens(data: dict[str, Any], key: str) -> int:
    tokens = data.get("tokens") or {}
    if key == "cache_read":
        return int((tokens.get("cache") or {}).get("read") or 0)
    if key == "cache_write":
        return int((tokens.get("cache") or {}).get("write") or 0)
    return int(tokens.get(key) or 0)


def _preview(value: Any, limit: int = 280) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, default=str)
    value = " ".join(value.split())
    if len(value) <= limit:
        return value
    return f"{value[: limit - 1]}..."


def _read_text_preview(path: Path, limit: int = 120_000) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            return handle.read(limit)
    except OSError as exc:
        return f"Unable to read {path}: {exc}"


def _model_id(data: dict[str, Any]) -> str:
    if data.get("modelID"):
        return str(data["modelID"])
    model = data.get("model") or {}
    if isinstance(model, dict):
        return str(model.get("modelID") or model.get("id") or model.get("name") or "")
    return ""


def _provider_id(data: dict[str, Any]) -> str:
    if data.get("providerID"):
        return str(data["providerID"])
    model = data.get("model") or {}
    if isinstance(model, dict):
        return str(model.get("providerID") or model.get("provider") or "")
    return ""


def _extract_paths(value: Any) -> list[str]:
    text = _dump_json(value)
    paths = []
    seen = set()
    for match in PATH_RE.finditer(text):
        path = match.group("path").rstrip(".,;:")
        if path not in seen:
            seen.add(path)
            paths.append(path)
    return paths


@dataclass(frozen=True)
class SessionDiff:
    file: str
    status: str
    additions: int
    deletions: int
    patch: str


class OpenCodeStore:
    """Read-only access to the opencode SQLite data model."""

    def __init__(self, db_path: str | Path | None = None, retries: int = 3) -> None:
        self.db_path = resolve_db_path(db_path)
        self.retries = retries

    @property
    def storage_dir(self) -> Path:
        return self.db_path.parent / "storage"

    @property
    def log_dir(self) -> Path:
        return self.db_path.parent / "log"

    @property
    def tool_output_dir(self) -> Path:
        return self.db_path.parent / "tool-output"

    def connect(self) -> sqlite3.Connection:
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")
        conn = sqlite3.connect(_sqlite_uri(self.db_path), uri=True, timeout=5)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only = ON")
        return conn

    def rows(self, query: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
        last_error: sqlite3.OperationalError | None = None
        for attempt in range(self.retries):
            try:
                with self.connect() as conn:
                    return conn.execute(query, tuple(params)).fetchall()
            except sqlite3.OperationalError as exc:
                last_error = exc
                if "locked" not in str(exc).lower() or attempt == self.retries - 1:
                    raise
                time.sleep(0.2 * (attempt + 1))
        if last_error:
            raise last_error
        return []

    def table_names(self) -> list[str]:
        rows = self.rows(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            ORDER BY name
            """
        )
        return [str(row["name"]) for row in rows]

    def has_table(self, table: str) -> bool:
        return table in set(self.table_names())

    def db_overview(self) -> pd.DataFrame:
        records = []
        for table in self.table_names():
            count = int(self.rows(f'SELECT COUNT(*) AS count FROM "{table}"')[0]["count"])
            records.append({"table": table, "rows": count})
        return pd.DataFrame(records)

    def migrations(self) -> pd.DataFrame:
        if not self.has_table("__drizzle_migrations"):
            return pd.DataFrame(columns=["id", "name", "applied_at"])
        rows = self.rows(
            """
            SELECT id, name, applied_at
            FROM __drizzle_migrations
            ORDER BY id
            """
        )
        return pd.DataFrame([dict(row) for row in rows])

    def sessions(self, include_archived: bool = False) -> pd.DataFrame:
        where = "" if include_archived else "WHERE s.time_archived IS NULL"
        rows = self.rows(
            f"""
            SELECT
                s.id,
                s.project_id,
                s.parent_id,
                s.slug,
                s.directory,
                s.title,
                s.version,
                s.share_url,
                s.summary_additions,
                s.summary_deletions,
                s.summary_files,
                s.permission,
                s.time_created,
                s.time_updated,
                s.time_compacting,
                s.time_archived,
                s.workspace_id,
                p.name AS project_name,
                p.vcs AS project_vcs,
                COALESCE(message_counts.message_count, 0) AS message_count,
                COALESCE(part_counts.part_count, 0) AS part_count
            FROM session s
            LEFT JOIN project p ON p.id = s.project_id
            LEFT JOIN (
                SELECT session_id, COUNT(*) AS message_count
                FROM message
                GROUP BY session_id
            ) message_counts ON message_counts.session_id = s.id
            LEFT JOIN (
                SELECT session_id, COUNT(*) AS part_count
                FROM part
                GROUP BY session_id
            ) part_counts ON part_counts.session_id = s.id
            {where}
            ORDER BY s.time_updated DESC
            """
        )
        frame = pd.DataFrame([dict(row) for row in rows])
        if frame.empty:
            return frame

        frame["created"] = frame["time_created"].map(_format_time_ms)
        frame["updated"] = frame["time_updated"].map(_format_time_ms)
        frame["duration_min"] = (
            (frame["time_updated"] - frame["time_created"]).clip(lower=0) / 60000
        ).round(1)

        message_stats = self.message_stats()
        if not message_stats.empty:
            frame = frame.merge(message_stats, left_on="id", right_on="session_id", how="left")
            frame = frame.drop(columns=["session_id"])

        part_stats = self.part_stats()
        if not part_stats.empty:
            frame = frame.merge(part_stats, left_on="id", right_on="session_id", how="left")
            frame = frame.drop(columns=["session_id"])

        child_stats = self.child_session_stats()
        if not child_stats.empty:
            frame = frame.merge(child_stats, left_on="id", right_on="parent_id", how="left")
            frame = frame.drop(columns=["parent_id_y"]).rename(columns={"parent_id_x": "parent_id"})

        numeric_columns = [
            "summary_additions",
            "summary_deletions",
            "summary_files",
            "message_count",
            "part_count",
            "total_cost",
            "token_total",
            "token_input",
            "token_output",
            "token_reasoning",
            "cache_read",
            "cache_write",
            "tool_count",
            "task_count",
            "error_count",
            "child_session_count",
        ]
        for column in numeric_columns:
            if column not in frame:
                frame[column] = 0
            frame[column] = frame[column].fillna(0)

        return frame

    def child_session_stats(self) -> pd.DataFrame:
        rows = self.rows(
            """
            SELECT parent_id, COUNT(*) AS child_session_count
            FROM session
            WHERE parent_id IS NOT NULL
            GROUP BY parent_id
            """
        )
        return pd.DataFrame([dict(row) for row in rows])

    def message_stats(self) -> pd.DataFrame:
        messages = self.messages()
        if messages.empty:
            return pd.DataFrame()
        grouped = messages.groupby("session_id", as_index=False).agg(
            total_cost=("cost", "sum"),
            token_total=("token_total", "sum"),
            token_input=("token_input", "sum"),
            token_output=("token_output", "sum"),
            token_reasoning=("token_reasoning", "sum"),
            cache_read=("cache_read", "sum"),
            cache_write=("cache_write", "sum"),
        )
        grouped["total_cost"] = grouped["total_cost"].round(6)
        return grouped

    def part_stats(self) -> pd.DataFrame:
        parts = self.all_parts()
        if parts.empty:
            return pd.DataFrame()
        grouped = parts.groupby("session_id", as_index=False).agg(
            tool_count=("is_tool", "sum"),
            task_count=("is_task", "sum"),
            error_count=("is_error", "sum"),
        )
        return grouped

    def messages(self, session_id: str | None = None) -> pd.DataFrame:
        where = ""
        params: tuple[str, ...] = ()
        if session_id:
            where = "WHERE session_id = ?"
            params = (session_id,)
        rows = self.rows(
            f"""
            SELECT id, session_id, time_created, time_updated, data
            FROM message
            {where}
            ORDER BY time_created ASC
            """,
            params,
        )
        records = []
        for row in rows:
            data = _load_json(row["data"])
            path = data.get("path") or {}
            records.append(
                {
                    "id": row["id"],
                    "session_id": row["session_id"],
                    "role": data.get("role", ""),
                    "agent": data.get("agent", ""),
                    "mode": data.get("mode", ""),
                    "model_id": _model_id(data),
                    "provider_id": _provider_id(data),
                    "finish": data.get("finish", ""),
                    "cost": float(data.get("cost") or 0),
                    "token_total": _tokens(data, "total"),
                    "token_input": _tokens(data, "input"),
                    "token_output": _tokens(data, "output"),
                    "token_reasoning": _tokens(data, "reasoning"),
                    "cache_read": _tokens(data, "cache_read"),
                    "cache_write": _tokens(data, "cache_write"),
                    "cwd": path.get("cwd", ""),
                    "root": path.get("root", ""),
                    "created": _format_time_ms(row["time_created"]),
                    "updated": _format_time_ms(row["time_updated"]),
                    "raw": data,
                }
            )
        return pd.DataFrame(records)

    def all_parts(self, session_id: str | None = None) -> pd.DataFrame:
        where = ""
        params: tuple[str, ...] = ()
        if session_id:
            where = "WHERE session_id = ?"
            params = (session_id,)
        rows = self.rows(
            f"""
            SELECT id, message_id, session_id, time_created, time_updated, data
            FROM part
            {where}
            ORDER BY time_created ASC
            """,
            params,
        )
        records = []
        for row in rows:
            data = _load_json(row["data"])
            part_type = data.get("type", "")
            state = data.get("state") or {}
            output = state.get("output") or state.get("out")
            error = state.get("error")
            tool = data.get("tool", "")
            records.append(
                {
                    "id": row["id"],
                    "message_id": row["message_id"],
                    "session_id": row["session_id"],
                    "type": part_type,
                    "tool": tool,
                    "status": state.get("status", ""),
                    "input": state.get("input"),
                    "output": output,
                    "error": error,
                    "output_len": len(str(output or "")),
                    "task_id": self._task_id(data),
                    "paths": _extract_paths(state.get("input")) + _extract_paths(output),
                    "created": _format_time_ms(row["time_created"]),
                    "updated": _format_time_ms(row["time_updated"]),
                    "preview": _preview(self._part_content(data)),
                    "content": self._part_content(data),
                    "is_tool": part_type == "tool",
                    "is_task": part_type == "tool" and tool == "task",
                    "is_error": state.get("status") == "error" or bool(error),
                    "raw": data,
                }
            )
        return pd.DataFrame(records)

    def parts(self, session_id: str) -> pd.DataFrame:
        return self.all_parts(session_id)

    def transcript(self, session_id: str) -> pd.DataFrame:
        messages = self.messages(session_id)
        parts = self.parts(session_id)
        if messages.empty or parts.empty:
            return pd.DataFrame()

        message_meta = messages.set_index("id")[["role", "agent", "model_id"]].to_dict("index")
        records = []
        for _, part in parts.iterrows():
            meta = message_meta.get(part["message_id"], {})
            records.append(
                {
                    "created": part["created"],
                    "role": meta.get("role", ""),
                    "agent": meta.get("agent", ""),
                    "model_id": meta.get("model_id", ""),
                    "type": part["type"],
                    "tool": part["tool"],
                    "status": part["status"],
                    "preview": part["preview"],
                    "content": part["content"],
                }
            )
        return pd.DataFrame(records)

    def tools(self, session_id: str) -> pd.DataFrame:
        parts = self.parts(session_id)
        if parts.empty:
            return pd.DataFrame(
                columns=[
                    "created",
                    "tool",
                    "status",
                    "task_id",
                    "output_len",
                    "preview",
                    "input_text",
                    "output_text",
                    "error",
                ]
            )
        tools = parts[parts["type"] == "tool"].copy()
        if tools.empty:
            return tools
        tools["input_text"] = tools["input"].map(_dump_json)
        tools["output_text"] = tools["output"].map(_dump_json)
        tools["path_count"] = tools["paths"].map(len)
        return tools

    def subagents(self, session_id: str) -> pd.DataFrame:
        children = self.rows(
            """
            SELECT id, parent_id, title, slug, time_created, time_updated
            FROM session
            WHERE parent_id = ?
            ORDER BY time_created ASC
            """,
            (session_id,),
        )
        child_records = {
            row["id"]: {
                "child_session_id": row["id"],
                "parent_id": row["parent_id"],
                "child_title": row["title"],
                "child_slug": row["slug"],
                "child_created": _format_time_ms(row["time_created"]),
                "child_updated": _format_time_ms(row["time_updated"]),
            }
            for row in children
        }

        tasks = self.tools(session_id)
        records = []
        if not tasks.empty:
            for _, task in tasks[tasks["tool"] == "task"].iterrows():
                child_id = task.get("task_id") or ""
                if not child_id:
                    continue
                child = child_records.pop(child_id, {})
                records.append(
                    {
                        "created": task.get("created", ""),
                        "status": task.get("status", ""),
                        "task_id": child_id,
                        "description": _task_description(task.get("input")),
                        "result_preview": _preview(task.get("output_text"), 500),
                        **child,
                    }
                )

        for child in child_records.values():
            records.append(
                {
                    "created": child["child_created"],
                    "status": "",
                    "task_id": child["child_session_id"],
                    "description": child["child_title"],
                    "result_preview": "",
                    **child,
                }
            )
        return pd.DataFrame(records)

    def workflow_phases(self, session_id: str) -> pd.DataFrame:
        tools = self.tools(session_id)
        if tools.empty:
            return pd.DataFrame()

        latest_state: dict[str, Any] = {}
        for _, row in tools.iterrows():
            if not str(row.get("tool", "")).startswith("validate_paper_workflow_"):
                continue
            output = _load_json(str(row.get("output") or ""))
            state = output.get("state") if isinstance(output.get("state"), dict) else output
            if isinstance(state, dict) and state.get("phases"):
                latest_state = state

        phases = latest_state.get("phases") or {}
        phase_order = latest_state.get("phase_order") or list(phases)
        records = []
        for position, name in enumerate(phase_order):
            phase = phases.get(name) or {}
            records.append(
                {
                    "position": position + 1,
                    "phase": name,
                    "status": phase.get("status", ""),
                    "attempts": phase.get("attempts", 0),
                    "started_at": phase.get("started_at", ""),
                    "completed_at": phase.get("completed_at", ""),
                    "last_error": phase.get("last_error_message", ""),
                    "artifact_count": len(phase.get("artifact_paths") or []),
                    "artifact_paths": "\n".join(phase.get("artifact_paths") or []),
                }
            )
        return pd.DataFrame(records)

    def artifact_paths(self, session_id: str) -> pd.DataFrame:
        records = []
        for _, row in self.parts(session_id).iterrows():
            for path in row.get("paths") or []:
                records.append(
                    {
                        "created": row.get("created", ""),
                        "source": row.get("tool") or row.get("type", ""),
                        "status": row.get("status", ""),
                        "path": path,
                    }
                )
        if not records:
            return pd.DataFrame(columns=["created", "source", "status", "path"])
        frame = pd.DataFrame(records).drop_duplicates()
        return frame.sort_values(["created", "path"]).reset_index(drop=True)

    def session_diff(self, session_id: str) -> pd.DataFrame:
        path = self.storage_dir / "session_diff" / f"{session_id}.json"
        if not path.exists():
            return pd.DataFrame(columns=["file", "status", "additions", "deletions", "patch"])

        loaded = json.loads(path.read_text())
        records: list[SessionDiff] = []
        for item in loaded if isinstance(loaded, list) else []:
            records.append(
                SessionDiff(
                    file=item.get("file", ""),
                    status=item.get("status", ""),
                    additions=int(item.get("additions") or 0),
                    deletions=int(item.get("deletions") or 0),
                    patch=item.get("patch", ""),
                )
            )
        return pd.DataFrame([record.__dict__ for record in records])

    def session_diff_files(self) -> pd.DataFrame:
        diff_dir = self.storage_dir / "session_diff"
        if not diff_dir.exists():
            return pd.DataFrame(columns=["session_id", "path", "bytes"])
        records = []
        for path in sorted(diff_dir.glob("*.json")):
            records.append(
                {
                    "session_id": path.stem,
                    "path": str(path),
                    "bytes": path.stat().st_size,
                }
            )
        return pd.DataFrame(records)

    def logs(self, session_id: str | None = None) -> pd.DataFrame:
        if not self.log_dir.exists():
            return pd.DataFrame(columns=["file", "line", "level", "timestamp", "service", "text"])
        records = []
        for path in sorted(self.log_dir.glob("*.log")):
            try:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for number, line in enumerate(lines, start=1):
                if session_id and session_id not in line:
                    continue
                records.append(
                    {
                        "file": path.name,
                        "line": number,
                        "level": line.split(maxsplit=1)[0] if line else "",
                        "timestamp": _log_timestamp(line),
                        "service": _log_service(line),
                        "text": line,
                        "preview": _preview(line, 500),
                    }
                )
        return pd.DataFrame(records)

    def tool_output_files(self) -> pd.DataFrame:
        if not self.tool_output_dir.exists():
            return pd.DataFrame(columns=["file", "path", "bytes", "preview", "content"])
        records = []
        for path in sorted(self.tool_output_dir.iterdir()):
            if not path.is_file():
                continue
            content = _read_text_preview(path)
            records.append(
                {
                    "file": path.name,
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "preview": _preview(content, 500),
                    "content": content,
                }
            )
        return pd.DataFrame(records)

    def raw_table(self, table: str, limit: int = 500) -> pd.DataFrame:
        if table not in self.table_names():
            return pd.DataFrame()
        rows = self.rows(f'SELECT * FROM "{table}" LIMIT ?', (limit,))
        frame = pd.DataFrame([dict(row) for row in rows])
        for column in frame.columns:
            if column.lower() in SENSITIVE_COLUMNS or "token" in column.lower():
                frame[column] = frame[column].map(
                    lambda value: "" if value in (None, "") else "[redacted]"
                )
        return frame

    @staticmethod
    def _part_content(data: dict[str, Any]) -> str:
        part_type = data.get("type")
        if part_type in {"text", "reasoning"}:
            return str(data.get("text") or "")
        if part_type == "tool":
            state = data.get("state") or {}
            payload = {
                "tool": data.get("tool"),
                "status": state.get("status"),
                "input": state.get("input"),
                "output": state.get("output") or state.get("out"),
                "error": state.get("error"),
            }
            return json.dumps(payload, ensure_ascii=False, indent=2, default=str)
        if part_type == "patch":
            return str(data.get("patch") or json.dumps(data, ensure_ascii=False, default=str))
        return json.dumps(data, ensure_ascii=False, indent=2, default=str)

    @staticmethod
    def _task_id(data: dict[str, Any]) -> str:
        if data.get("tool") != "task":
            return ""
        state = data.get("state") or {}
        output = str(state.get("output") or "")
        match = TASK_ID_RE.search(output)
        return match.group(1) if match else ""


def _task_description(input_data: Any) -> str:
    if isinstance(input_data, dict):
        description = input_data.get("description")
        prompt = input_data.get("prompt")
        agent = input_data.get("agent")
        return str(description or prompt or agent or "")
    return _preview(input_data, 240)


def _log_timestamp(line: str) -> str:
    parts = line.split()
    if len(parts) >= 2 and parts[1].startswith("20"):
        return parts[1]
    return ""


def _log_service(line: str) -> str:
    match = re.search(r"\bservice=([^\s]+)", line)
    return match.group(1) if match else ""
