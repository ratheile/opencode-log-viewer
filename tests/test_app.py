from __future__ import annotations

import json
from pathlib import Path

from opencode_viewer.app import _discover_db_options, _initial_db_inputs, _tool_part_markdown


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
