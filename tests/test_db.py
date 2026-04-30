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
            CREATE TABLE account (
                id TEXT PRIMARY KEY,
                email TEXT NOT NULL,
                access_token TEXT NOT NULL,
                refresh_token TEXT NOT NULL
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
        conn.execute(
            "INSERT INTO account VALUES (?, ?, ?, ?)",
            ("acc_1", "user@example.com", "secret-access", "secret-refresh"),
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


def test_messages_read_nested_model_shape(tmp_path: Path) -> None:
    db_path = tmp_path / "opencode.db"
    make_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE message SET data = ? WHERE id = 'msg_1'",
            (
                json.dumps(
                    {
                        "role": "assistant",
                        "agent": "general",
                        "model": {"providerID": "openai", "modelID": "gpt-new"},
                    }
                ),
            ),
        )

    messages = OpenCodeStore(db_path).messages("ses_1")

    assert messages.iloc[0]["provider_id"] == "openai"
    assert messages.iloc[0]["model_id"] == "gpt-new"


def test_subagents_link_task_output_to_child_session(tmp_path: Path) -> None:
    db_path = tmp_path / "opencode.db"
    make_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO session (
                id, project_id, parent_id, slug, directory, title, version,
                summary_additions, summary_deletions, summary_files,
                time_created, time_updated
            )
            VALUES (
                'ses_child', 'proj_1', 'ses_1', 'child', '/tmp/example',
                'Child subagent', '1.0', 0, 0, 0, 1700000003000, 1700000004000
            )
            """
        )
        conn.execute(
            "INSERT INTO part VALUES (?, ?, ?, ?, ?, ?)",
            (
                "prt_task",
                "msg_1",
                "ses_1",
                1700000003000,
                1700000004000,
                json.dumps(
                    {
                        "type": "tool",
                        "tool": "task",
                        "state": {
                            "status": "completed",
                            "input": {"description": "Run child work"},
                            "output": "task_id: ses_child\n\n<task_result>done</task_result>",
                        },
                    }
                ),
            ),
        )

    subagents = OpenCodeStore(db_path).subagents("ses_1")
    sessions = OpenCodeStore(db_path).sessions(include_archived=True)

    assert subagents.iloc[0]["task_id"] == "ses_child"
    assert subagents.iloc[0]["child_title"] == "Child subagent"
    assert sessions[sessions["id"] == "ses_1"].iloc[0]["child_session_count"] == 1
    assert sessions[sessions["id"] == "ses_1"].iloc[0]["task_count"] == 1


def test_subagents_ignore_task_without_persisted_task_id(tmp_path: Path) -> None:
    db_path = tmp_path / "opencode.db"
    make_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO part VALUES (?, ?, ?, ?, ?, ?)",
            (
                "prt_task_without_id",
                "msg_1",
                "ses_1",
                1700000003000,
                1700000004000,
                json.dumps(
                    {
                        "type": "tool",
                        "tool": "task",
                        "state": {
                            "status": "completed",
                            "input": {"description": "Registered, but not persisted"},
                            "output": "<task_result>done</task_result>",
                        },
                    }
                ),
            ),
        )

    subagents = OpenCodeStore(db_path).subagents("ses_1")

    assert subagents.empty


def test_workflow_phases_parse_latest_tool_output(tmp_path: Path) -> None:
    db_path = tmp_path / "opencode.db"
    make_db(db_path)
    workflow_output = {
        "state": {
            "phase_order": ["input_check", "run_training"],
            "phases": {
                "input_check": {
                    "status": "passed",
                    "attempts": 1,
                    "started_at": "2026-01-01T00:00:00+00:00",
                    "completed_at": "2026-01-01T00:01:00+00:00",
                    "artifact_paths": ["/tmp/artifact.md"],
                },
                "run_training": {
                    "status": "running",
                    "attempts": 2,
                    "artifact_paths": [],
                },
            },
        }
    }
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO part VALUES (?, ?, ?, ?, ?, ?)",
            (
                "prt_workflow",
                "msg_1",
                "ses_1",
                1700000003000,
                1700000004000,
                json.dumps(
                    {
                        "type": "tool",
                        "tool": "validate_paper_workflow_get_status",
                        "state": {
                            "status": "completed",
                            "output": json.dumps(workflow_output),
                        },
                    }
                ),
            ),
        )

    phases = OpenCodeStore(db_path).workflow_phases("ses_1")
    artifacts = OpenCodeStore(db_path).artifact_paths("ses_1")

    assert phases["phase"].tolist() == ["input_check", "run_training"]
    assert phases.iloc[0]["status"] == "passed"
    assert "/tmp/artifact.md" in artifacts["path"].tolist()


def test_sidecars_and_raw_table_redaction(tmp_path: Path) -> None:
    db_path = tmp_path / "opencode.db"
    make_db(db_path)
    log_dir = tmp_path / "log"
    log_dir.mkdir()
    (log_dir / "2026.log").write_text("INFO  2026-01-01T00:00:00 service=test ses_1 hello")
    output_dir = tmp_path / "tool-output"
    output_dir.mkdir()
    (output_dir / "tool_1").write_text("large command output")

    store = OpenCodeStore(db_path)
    logs = store.logs("ses_1")
    outputs = store.tool_output_files()
    raw = store.raw_table("account")

    assert logs.iloc[0]["service"] == "test"
    assert outputs.iloc[0]["file"] == "tool_1"
    assert raw.iloc[0]["access_token"] == "[redacted]"
    assert raw.iloc[0]["refresh_token"] == "[redacted]"
