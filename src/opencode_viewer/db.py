from __future__ import annotations

import json
import os
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
        ]
        for column in numeric_columns:
            if column not in frame:
                frame[column] = 0
            frame[column] = frame[column].fillna(0)

        return frame

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
                    "model_id": data.get("modelID", ""),
                    "provider_id": data.get("providerID", ""),
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

    def parts(self, session_id: str) -> pd.DataFrame:
        rows = self.rows(
            """
            SELECT id, message_id, session_id, time_created, time_updated, data
            FROM part
            WHERE session_id = ?
            ORDER BY time_created ASC
            """,
            (session_id,),
        )
        records = []
        for row in rows:
            data = _load_json(row["data"])
            part_type = data.get("type", "")
            state = data.get("state") or {}
            records.append(
                {
                    "id": row["id"],
                    "message_id": row["message_id"],
                    "session_id": row["session_id"],
                    "type": part_type,
                    "tool": data.get("tool", ""),
                    "status": state.get("status", ""),
                    "created": _format_time_ms(row["time_created"]),
                    "updated": _format_time_ms(row["time_updated"]),
                    "preview": _preview(self._part_content(data)),
                    "content": self._part_content(data),
                    "raw": data,
                }
            )
        return pd.DataFrame(records)

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
