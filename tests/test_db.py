from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from opencode_viewer.db import OpenCodeStore


def make_db(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE project (
                id TEXT PRIMARY KEY,
                name TEXT,
                vcs TEXT
            );
            CREATE TABLE session (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                parent_id TEXT,
                slug TEXT NOT NULL,
                directory TEXT NOT NULL,
                title TEXT NOT NULL,
                version TEXT NOT NULL,
                share_url TEXT,
                summary_additions INTEGER,
                summary_deletions INTEGER,
                summary_files INTEGER,
                summary_diffs TEXT,
                revert TEXT,
                permission TEXT,
                time_created INTEGER NOT NULL,
                time_updated INTEGER NOT NULL,
                time_compacting INTEGER,
                time_archived INTEGER,
                workspace_id TEXT
            );
            CREATE TABLE message (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                time_created INTEGER NOT NULL,
                time_updated INTEGER NOT NULL,
                data TEXT NOT NULL
            );
            CREATE TABLE part (
                id TEXT PRIMARY KEY,
                message_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                time_created INTEGER NOT NULL,
                time_updated INTEGER NOT NULL,
                data TEXT NOT NULL
            );
            """
        )
        conn.execute("INSERT INTO project VALUES ('proj_1', 'Example', 'git')")
        conn.execute(
            """
            INSERT INTO session (
                id, project_id, slug, directory, title, version,
                summary_additions, summary_deletions, summary_files,
                time_created, time_updated
            )
            VALUES (
                'ses_1', 'proj_1', 'example', '/tmp/example', 'Example session', '1.0',
                3, 1, 2, 1700000000000, 1700000060000
            )
            """
        )
        conn.execute(
            "INSERT INTO message VALUES (?, ?, ?, ?, ?)",
            (
                "msg_1",
                "ses_1",
                1700000001000,
                1700000002000,
                json.dumps(
                    {
                        "role": "assistant",
                        "agent": "general",
                        "mode": "general",
                        "modelID": "gpt-test",
                        "providerID": "openai",
                        "cost": 0.01,
                        "tokens": {
                            "total": 42,
                            "input": 30,
                            "output": 12,
                            "reasoning": 5,
                            "cache": {"read": 10, "write": 2},
                        },
                    }
                ),
            ),
        )
        conn.execute(
            "INSERT INTO part VALUES (?, ?, ?, ?, ?, ?)",
            (
                "prt_1",
                "msg_1",
                "ses_1",
                1700000001000,
                1700000002000,
                json.dumps({"type": "text", "text": "Hello from opencode"}),
            ),
        )


def test_sessions_include_message_and_token_stats(tmp_path: Path) -> None:
    db_path = tmp_path / "opencode.db"
    make_db(db_path)

    sessions = OpenCodeStore(db_path).sessions()

    assert len(sessions) == 1
    row = sessions.iloc[0]
    assert row["id"] == "ses_1"
    assert row["message_count"] == 1
    assert row["part_count"] == 1
    assert row["token_total"] == 42
    assert row["total_cost"] == 0.01


def test_transcript_reads_text_parts(tmp_path: Path) -> None:
    db_path = tmp_path / "opencode.db"
    make_db(db_path)

    transcript = OpenCodeStore(db_path).transcript("ses_1")

    assert len(transcript) == 1
    assert transcript.iloc[0]["role"] == "assistant"
    assert transcript.iloc[0]["type"] == "text"
    assert "Hello from opencode" in transcript.iloc[0]["content"]


def test_session_diff_reads_storage_file(tmp_path: Path) -> None:
    db_path = tmp_path / "opencode.db"
    make_db(db_path)
    diff_dir = tmp_path / "storage" / "session_diff"
    diff_dir.mkdir(parents=True)
    (diff_dir / "ses_1.json").write_text(
        json.dumps([{"file": "example.py", "status": "modified", "additions": 2, "patch": "diff"}])
    )

    diffs = OpenCodeStore(db_path).session_diff("ses_1")

    assert len(diffs) == 1
    assert diffs.iloc[0]["file"] == "example.py"
    assert diffs.iloc[0]["additions"] == 2
