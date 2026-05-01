from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from opencode_viewer.app import (
    _agent_conversation_draft_markdown,
    _agentic_log_markdown,
    _chat_message_presentation,
    _discover_db_options,
    _has_value,
    _initial_db_inputs,
    _sidecar_log_markdown,
    _tool_part_markdown,
    _tool_response_markdown,
)


def test_discover_db_options_labels_run_folders(tmp_path: Path) -> None:
    db_path = tmp_path / "example-run" / "opencode" / "opencode.db"
    db_path.parent.mkdir(parents=True)
    db_path.touch()

    options = _discover_db_options(str(tmp_path))

    assert options == {"example-run": str(db_path)}


def test_initial_db_inputs_accepts_runs_folder_as_db_argument(tmp_path: Path) -> None:
    db_path = tmp_path / "example-run" / "opencode" / "opencode.db"
    db_path.parent.mkdir(parents=True)
    db_path.touch()

    db_value, root_value, select_first = _initial_db_inputs(tmp_path, None)

    assert db_value == str(db_path)
    assert root_value == str(tmp_path)
    assert select_first is False


def test_tool_part_markdown_summarizes_workflow_json() -> None:
    payload = {
        "tool": "validate_paper_workflow_init_run",
        "status": "completed",
        "input": {"repo_root": "/mnt/example"},
        "output": json.dumps(
            {
                "state": {
                    "overall_status": "active",
                    "current_phase": "run_training",
                    "run_mode": "full",
                    "hp_mode": "reference_only",
                    "phase_order": ["run_training"],
                    "phases": {
                        "run_training": {
                            "status": "running",
                            "attempts": 2,
                            "artifact_paths": ["/mnt/example/artifact.json"],
                        }
                    },
                }
            }
        ),
    }

    text = _tool_part_markdown(json.dumps(payload))

    assert "Overall status: `active`" in text
    assert "Current phase: `run_training`" in text
    assert "| `run_training` | `running` | 2 | 1 |" in text


def test_tool_response_markdown_uses_output_formatter() -> None:
    text = _tool_response_markdown(
        {
            "output": json.dumps({"state": {"overall_status": "done", "phases": {}}}),
            "error": "",
        }
    )

    assert "Overall status: `done`" in text


def test_tool_response_markdown_ignores_missing_nan_error() -> None:
    text = _tool_response_markdown(
        {
            "output": json.dumps({"state": {"overall_status": "done", "phases": {}}}),
            "error": float("nan"),
        }
    )

    assert "**Error**" not in text
    assert "Overall status: `done`" in text


def test_has_value_treats_pandas_missing_values_as_empty() -> None:
    assert not _has_value(None)
    assert not _has_value("")
    assert not _has_value(float("nan"))
    assert _has_value("text")
    assert _has_value({"output": "text"})


def test_agentic_log_markdown_prefers_session_agentic_logs() -> None:
    logs = pd.DataFrame(
        [
            {"service": "session", "text": "session created"},
            {"service": "llm", "text": "llm stream started"},
            {"service": "tool.registry", "text": "tool registry event"},
        ]
    )

    text = _agentic_log_markdown(logs, limit=5)

    assert "### Agentic Logs" in text
    assert "session created" in text
    assert "llm stream started" in text
    assert "tool registry event" not in text


def test_sidecar_log_markdown_prefers_logs_near_selected_part() -> None:
    logs = pd.DataFrame(
        [
            {
                "file": "run.log",
                "line": 10,
                "level": "INFO",
                "timestamp": "2026-05-01T10:00:00",
                "timestamp_ms": 1_000_000,
                "service": "session",
                "text": "INFO 2026-05-01T10:00:00 service=session session start",
            },
            {
                "file": "run.log",
                "line": 20,
                "level": "INFO",
                "timestamp": "2026-05-01T10:01:00",
                "timestamp_ms": 1_060_000,
                "service": "session.processor",
                "text": "INFO 2026-05-01T10:01:00 messageID=msg_1 process",
            },
            {
                "file": "run.log",
                "line": 21,
                "level": "INFO",
                "timestamp": "2026-05-01T10:01:01",
                "timestamp_ms": 1_061_000,
                "service": "llm",
                "text": "INFO 2026-05-01T10:01:01 service=llm stream",
            },
            {
                "file": "run.log",
                "line": 40,
                "level": "INFO",
                "timestamp": "2026-05-01T10:10:00",
                "timestamp_ms": 1_600_000,
                "service": "session",
                "text": "INFO 2026-05-01T10:10:00 far away",
            },
        ]
    )
    part = pd.Series(
        {
            "id": "prt_1",
            "message_id": "msg_1",
            "created": "2026-05-01 12:01:00 CEST",
            "created_ms": 1_060_000,
        }
    )

    text = _sidecar_log_markdown(logs, part, limit=10, window_seconds=2)

    assert "Selected part time: `2026-05-01 12:01:00 CEST`" in text
    assert "run.log:20" in text
    assert "messageID=msg_1 process" in text
    assert "service=llm stream" in text
    assert "far away" not in text


def test_agent_conversation_draft_markdown_shows_child_transcript_parts() -> None:
    transcript = pd.DataFrame(
        [
            {
                "created": "2026-05-01 12:00:00 CEST",
                "role": "user",
                "agent": "",
                "type": "text",
                "tool": "",
                "status": "",
                "preview": "Child prompt preview",
                "content": "Child prompt text",
            },
            {
                "created": "2026-05-01 12:00:01 CEST",
                "role": "assistant",
                "agent": "worker",
                "type": "reasoning",
                "tool": "",
                "status": "",
                "preview": "Reasoning preview",
                "content": "Assistant reasoning text",
            },
            {
                "created": "2026-05-01 12:00:02 CEST",
                "role": "assistant",
                "agent": "worker",
                "type": "tool",
                "tool": "read",
                "status": "completed",
                "preview": "read preview",
                "content": json.dumps(
                    {
                        "tool": "read",
                        "status": "completed",
                        "input": {"filePath": "/tmp/example.py"},
                        "output": "child tool output text",
                    }
                ),
            },
            {
                "created": "2026-05-01 12:00:03 CEST",
                "role": "assistant",
                "agent": "worker",
                "type": "text",
                "tool": "",
                "status": "",
                "preview": "Assistant response preview",
                "content": "Assistant child response text",
            },
        ]
    )

    text = _agent_conversation_draft_markdown("ses_child", transcript)

    assert "### Agent Conversation Draft `ses_child`" in text
    assert "Child prompt text" in text
    assert "Assistant reasoning text" in text
    assert "tool `read` - completed" in text
    assert "child tool output text" in text
    assert "Assistant child response text" in text


def test_agent_conversation_draft_markdown_handles_missing_transcript() -> None:
    text = _agent_conversation_draft_markdown("ses_missing", pd.DataFrame())

    assert "### Agent Conversation Draft `ses_missing`" in text
    assert "No child transcript found" in text


def test_chat_message_presentation_assigns_distinct_part_classes() -> None:
    assert _chat_message_presentation("assistant", "tool", "bash")[2] == "opencode-chat-tool"
    assert _chat_message_presentation("assistant", "reasoning", "")[2] == (
        "opencode-chat-reasoning"
    )
    assert _chat_message_presentation("assistant", "patch", "")[2] == "opencode-chat-patch"
    assert _chat_message_presentation("assistant", "step-start", "")[2] == (
        "opencode-chat-step-start"
    )
    assert _chat_message_presentation("assistant", "step-finish", "")[2] == (
        "opencode-chat-step-finish"
    )
    assert _chat_message_presentation("user", "text", "")[2] == "opencode-chat-user"
