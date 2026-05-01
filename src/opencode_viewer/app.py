from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import panel as pn

from opencode_viewer.db import OpenCodeStore, resolve_db_path

pn.extension("tabulator", "jsoneditor", sizing_mode="stretch_width")


SESSION_COLUMNS = [
    "updated",
    "title",
    "parent_id",
    "message_count",
    "part_count",
    "tool_count",
    "task_count",
    "child_session_count",
    "error_count",
    "summary_files",
    "token_total",
    "total_cost",
]

MESSAGE_COLUMNS = [
    "created",
    "role",
    "agent",
    "mode",
    "model_id",
    "provider_id",
    "finish",
    "cost",
    "token_total",
    "token_input",
    "token_output",
    "token_reasoning",
]

TRANSCRIPT_COLUMNS = ["created", "role", "agent", "type", "tool", "status", "preview"]

TOOL_COLUMNS = [
    "created",
    "tool",
    "status",
    "task_id",
    "output_len",
    "path_count",
    "preview",
]

SUBAGENT_COLUMNS = [
    "created",
    "status",
    "task_id",
    "description",
    "child_title",
    "child_session_id",
    "result_preview",
]

WORKFLOW_COLUMNS = [
    "position",
    "phase",
    "status",
    "attempts",
    "started_at",
    "completed_at",
    "artifact_count",
    "last_error",
]

DEFAULT_DB_ROOT = Path.cwd().parent / "opencode"

COLUMN_LABELS = {
    "updated": "Updated",
    "created": "Time",
    "title": "Title",
    "parent_id": "Parent session",
    "message_count": "Messages",
    "part_count": "Parts",
    "tool_count": "Tool calls",
    "task_count": "Task calls",
    "child_session_count": "Child sessions",
    "error_count": "Errors",
    "summary_files": "Changed files",
    "token_total": "Tokens",
    "total_cost": "Cost",
    "role": "Role",
    "agent": "Agent",
    "mode": "Mode",
    "model_id": "Model",
    "provider_id": "Provider",
    "finish": "Finish",
    "cost": "Cost",
    "token_input": "Input tokens",
    "token_output": "Output tokens",
    "token_reasoning": "Reasoning tokens",
    "type": "Part type",
    "tool": "Tool",
    "status": "Status",
    "task_id": "Persisted task id",
    "output_len": "Output chars",
    "path_count": "Path refs",
    "preview": "Preview",
    "description": "Task description",
    "child_title": "Child title",
    "child_session_id": "Child session",
    "result_preview": "Result preview",
    "position": "Step",
    "phase": "Workflow phase",
    "attempts": "Attempts",
    "started_at": "Started",
    "completed_at": "Completed",
    "artifact_count": "Artifact paths",
    "last_error": "Last error",
    "source": "Source",
    "path": "Path",
    "file": "File",
    "bytes": "Bytes",
    "session_id": "Session",
    "additions": "+ lines",
    "deletions": "- lines",
    "line": "Line",
    "level": "Level",
    "timestamp": "Timestamp",
    "service": "Service",
    "table": "Table",
    "rows": "Rows",
}

WHAT_AM_I_LOOKING_AT = """
### What am I looking at?

**Sessions** - top-level and child OpenCode runs found in the selected SQLite DB.

**Conversation** - model messages plus the individual text, reasoning, and tool parts
that make up the transcript.

**Tool calls** - commands or tools invoked by the assistant; `task` links only count
as persisted subagent evidence when a child session id was saved.

**Workflow & paths** - validator phase state and artifact paths extracted from tool
payloads. Paths are provenance only; external files are not opened.

**Sidecar files** - DB-adjacent `storage/session_diff`, `tool-output`, and `log`
files, when present.

**Raw DB tables** - read-only SQLite rows for debugging, with known token columns redacted.
""".strip()

SESSION_SUMMARY_TEXT = """
This tab summarizes the selected session, the database tables, and any persisted
child-session/task links. If it says no persisted subagents were found, the DB did
not save child sessions or task outputs with `task_id` links for this session.
""".strip()

CONVERSATION_TEXT = """
Messages are model-level turns with token and cost metadata. Transcript rows are
the lower-level parts inside those turns, followed by a readable preview of the
first parts.
""".strip()

TOOLS_TEXT = """
Tool calls show assistant actions, statuses, output sizes, and extracted path
references. The detail preview expands the first calls without requiring the Raw
tables.
""".strip()

WORKFLOW_TEXT = """
Workflow phases are parsed from `validate_paper_workflow_*` tool outputs when
present. Artifact paths are displayed only as provenance; `/mnt/...` and other
external paths are not read.
""".strip()

OUTPUTS_TEXT = """
Tool outputs are read only from the DB-adjacent `tool-output` directory. This does
not follow artifact paths recorded inside tool payloads.
""".strip()

DIFFS_TEXT = """
Stored diffs come from DB-adjacent `storage/session_diff` JSON files. They are
session snapshots, not live git diffs.
""".strip()

LOGS_TEXT = """
Logs are DB-adjacent `.log` lines filtered to the selected session id when possible.
""".strip()

RAW_TEXT = """
Raw tables preserve access to the underlying SQLite data. Use these when the
summarized views hide a field you need.
""".strip()

CHAT_MESSAGE_STYLESHEET = """
.message.opencode-chat-user {
  background-color: #e8f1ff;
  border-left: 4px solid #2563eb;
}

.message.opencode-chat-assistant {
  background-color: #f8fafc;
  border-left: 4px solid #94a3b8;
}

.message.opencode-chat-reasoning {
  background-color: #fff7d6;
  border-left: 4px solid #b7791f;
}

.message.opencode-chat-tool {
  background-color: #e3f4f4;
  border-left: 4px solid #0f766e;
}

.message.opencode-chat-tool-response {
  background-color: #edf7f2;
  border-left: 4px solid #15803d;
}

.message.opencode-chat-patch {
  background-color: #e9f8ee;
  border-left: 4px solid #2f855a;
}

.message.opencode-chat-step-start {
  background-color: #f0e8ff;
  border-left: 4px solid #7c3aed;
}

.message.opencode-chat-step-finish {
  background-color: #eceff3;
  border-left: 4px solid #475569;
}

.message.opencode-chat-other {
  background-color: #f6f0e8;
  border-left: 4px solid #a16207;
}

.message[class*="opencode-chat-"] {
  align-items: flex-start;
  padding: 8px 10px;
}
"""

STEP_MESSAGE_TYPES = {"step-start", "step-finish"}
CHAT_VIEW_HEIGHT = 850
AGENTIC_LOG_SERVICES = {"session", "session.prompt", "session.processor", "llm"}


class OpenCodeDashboard:
    def __init__(
        self,
        db_path: str | Path | None = None,
        db_root: str | Path | None = None,
    ) -> None:
        initial_db_path, initial_db_root, select_first_db = _initial_db_inputs(db_path, db_root)
        self.db_root_input = pn.widgets.TextInput(
            name="OpenCode runs folder",
            value=initial_db_root,
            placeholder="Folder containing run folders, e.g. ../opencode",
            sizing_mode="stretch_width",
        )
        self.db_select = pn.widgets.Select(
            name="Run database",
            options={},
            sizing_mode="stretch_width",
        )
        self.scan_button = pn.widgets.Button(
            name="Scan runs",
            icon="folder-open",
            button_type="light",
        )
        self.db_input = pn.widgets.TextInput(
            name="Database",
            value=initial_db_path,
            placeholder="Path to opencode.db",
            sizing_mode="stretch_width",
        )
        self.include_archived = pn.widgets.Checkbox(name="Include archived", value=False)
        self.refresh_button = pn.widgets.Button(
            name="Refresh",
            icon="refresh",
            button_type="primary",
        )
        self.session_select = pn.widgets.Select(
            name="Session",
            options={},
            sizing_mode="stretch_width",
        )
        self.raw_table_select = pn.widgets.Select(
            name="Raw table",
            options=[],
            sizing_mode="stretch_width",
        )

        self.status = pn.pane.Alert("", alert_type="light", visible=False)
        self.summary = pn.pane.Markdown("")
        self.explainer = pn.pane.Markdown(WHAT_AM_I_LOOKING_AT)
        self.db_overview_table = self._table(page_size=50)
        self.sessions_table = self._table(page_size=50)
        self.session_meta = pn.pane.Markdown("")
        self.messages_table = self._table(page_size=50)
        self.transcript_table = self._table(page_size=50)
        self.transcript_preview = pn.pane.Markdown("")
        self.chat_feed = pn.Column(
            scroll=True,
            sizing_mode="stretch_width",
            height=CHAT_VIEW_HEIGHT,
        )
        self.detail_pane = pn.Column(
            pn.pane.Markdown("_Select a message to see details._"),
            scroll=True,
            sizing_mode="stretch_width",
            height=CHAT_VIEW_HEIGHT,
        )
        self.hide_step_messages = pn.widgets.Checkbox(
            name="Hide step-start/step-finish messages",
            value=False,
        )
        self.show_tool_calls = pn.widgets.Checkbox(
            name="Show tool calls",
            value=True,
        )
        self.show_tool_results = pn.widgets.Checkbox(
            name="Show tool results",
            value=True,
        )
        self._parts_df: pd.DataFrame = pd.DataFrame()
        self._messages_df: pd.DataFrame = pd.DataFrame()
        self._transcript_df: pd.DataFrame = pd.DataFrame()
        self._logs_df: pd.DataFrame = pd.DataFrame()
        self.tools_table = self._table(page_size=50)
        self.tool_detail = pn.pane.Markdown("")
        self.subagents_table = self._table(page_size=50)
        self.subagent_note = pn.pane.Markdown("")
        self.workflow_table = self._table(page_size=50)
        self.artifacts_table = self._table(page_size=50)
        self.outputs_table = self._table(page_size=50)
        self.output_detail = pn.pane.Markdown("")
        self.diff_files_table = self._table(page_size=50)
        self.diff_table = self._table(page_size=50)
        self.diff_preview = pn.pane.Markdown("")
        self.logs_table = self._table(page_size=100)
        self.raw_table = self._table(page_size=100)

        self._sessions = pd.DataFrame()
        self._store = self._new_store()

        self.scan_button.on_click(self.refresh_db_options)
        self.refresh_button.on_click(self.refresh)
        self.include_archived.param.watch(lambda _: self.refresh(), "value")
        self.db_root_input.param.watch(lambda _: self.refresh_db_options(), "value")
        self.db_select.param.watch(self._select_db, "value")
        self.session_select.param.watch(lambda _: self.refresh_detail(), "value")
        self.raw_table_select.param.watch(lambda _: self.refresh_raw_table(), "value")
        self.hide_step_messages.param.watch(lambda _: self._rebuild_chat_feed(), "value")
        self.show_tool_calls.param.watch(lambda _: self._rebuild_chat_feed(), "value")
        self.show_tool_results.param.watch(lambda _: self._rebuild_chat_feed(), "value")

        self.refresh_db_options(select_first=select_first_db)
        self.refresh()

    @staticmethod
    def _table(page_size: int) -> pn.widgets.Tabulator:
        return pn.widgets.Tabulator(
            pd.DataFrame(),
            disabled=True,
            pagination="local",
            page_size=page_size,
            show_index=False,
            sizing_mode="stretch_width",
        )

    def _new_store(self) -> OpenCodeStore:
        return OpenCodeStore(self.db_input.value)

    def refresh_db_options(
        self,
        event: Any | None = None,
        *,
        select_first: bool = False,
    ) -> None:
        del event
        options = _discover_db_options(self.db_root_input.value)
        self.db_select.options = options
        current_path = str(resolve_db_path(self.db_input.value))
        if current_path in options.values():
            self.db_select.value = current_path
        elif select_first and options:
            self.db_select.value = next(iter(options.values()))

    def _select_db(self, event: Any) -> None:
        if not event.new:
            return
        self.db_input.value = str(event.new)
        self.refresh()

    def refresh(self, event: Any | None = None) -> None:
        del event
        self._store = self._new_store()
        try:
            self._sessions = self._store.sessions(include_archived=self.include_archived.value)
            db_overview = self._store.db_overview()
        except Exception as exc:  # noqa: BLE001 - surface DB errors in the dashboard
            self._set_error(str(exc))
            self._sessions = pd.DataFrame()
            self.sessions_table.value = pd.DataFrame()
            self.session_select.options = {}
            self.summary.object = ""
            self.refresh_detail()
            return

        self._set_ok(f"Loaded {len(self._sessions)} sessions from `{self._store.db_path}`.")
        self.db_overview_table.value = self._display_frame(db_overview, ["table", "rows"])
        self.raw_table_select.options = self._store.table_names()
        if self.raw_table_select.options and not self.raw_table_select.value:
            self.raw_table_select.value = self.raw_table_select.options[0]
        self._refresh_summary(db_overview)
        self._refresh_session_selector()
        self.sessions_table.value = self._display_frame(self._sessions, SESSION_COLUMNS)
        self.refresh_detail()
        self.refresh_raw_table()

    def refresh_detail(self) -> None:
        session_id = self.session_select.value
        if not session_id:
            self._clear_detail()
            return

        session = self._session_row(session_id)
        self.session_meta.object = self._session_markdown(session)

        messages = self._store.messages(session_id)
        transcript = self._store.transcript(session_id)
        tools = self._store.tools(session_id)
        subagents = self._store.subagents(session_id)
        workflow = self._store.workflow_phases(session_id)
        artifacts = self._store.artifact_paths(session_id)
        diffs = self._store.session_diff(session_id)
        diff_files = self._store.session_diff_files()
        logs = self._store.logs(session_id)
        outputs = self._store.tool_output_files()

        if not transcript.empty:
            transcript = transcript.copy()
            transcript["preview"] = transcript.apply(_readable_part_preview, axis=1)
        if not tools.empty:
            tools = tools.copy()
            tools["preview"] = tools.apply(_readable_tool_preview, axis=1)

        self.messages_table.value = self._display_frame(messages, MESSAGE_COLUMNS)
        self.transcript_table.value = self._display_frame(transcript, TRANSCRIPT_COLUMNS)
        self.transcript_preview.object = self._transcript_markdown(transcript)
        self._parts_df = self._store.parts(session_id).reset_index(drop=True)
        self._messages_df = messages
        self._transcript_df = transcript
        self._logs_df = logs
        self.detail_pane.objects = [pn.pane.Markdown("_Select a message to see details._")]
        self._rebuild_chat_feed()
        self.tools_table.value = self._display_frame(tools, TOOL_COLUMNS)
        self.tool_detail.object = self._tools_markdown(tools)
        self.subagents_table.value = self._display_frame(subagents, SUBAGENT_COLUMNS)
        self.subagent_note.object = self._subagent_note(subagents, session)
        self.workflow_table.value = self._display_frame(workflow, WORKFLOW_COLUMNS)
        self.artifacts_table.value = self._display_frame(
            artifacts,
            ["created", "source", "status", "path"],
        )
        self.outputs_table.value = self._display_frame(outputs, ["file", "bytes", "preview"])
        self.output_detail.object = self._outputs_markdown(outputs)
        self.diff_files_table.value = self._display_frame(
            diff_files,
            ["session_id", "bytes", "path"],
        )
        self.diff_table.value = self._display_frame(
            diffs,
            ["file", "status", "additions", "deletions"],
        )
        self.diff_preview.object = self._diff_markdown(diffs)
        self.logs_table.value = self._display_frame(
            logs,
            ["file", "line", "level", "timestamp", "service", "preview"],
        )

    def refresh_raw_table(self) -> None:
        table = self.raw_table_select.value
        self.raw_table.value = self._store.raw_table(table) if table else pd.DataFrame()

    def view(self) -> pn.template.FastListTemplate:
        controls = pn.Column(
            self.db_root_input,
            pn.Row(self.scan_button),
            self.db_select,
            self.db_input,
            pn.Row(self.refresh_button, self.include_archived),
            self.session_select,
            self.status,
            self.explainer,
            sizing_mode="stretch_width",
        )
        tabs = pn.Tabs(
            (
                "Session Summary",
                pn.Column(
                    pn.pane.Markdown(SESSION_SUMMARY_TEXT),
                    self.session_meta,
                    self.subagent_note,
                    self.subagents_table,
                    pn.pane.Markdown("### Database Tables"),
                    self.db_overview_table,
                ),
            ),
            (
                "Conversation & Messages",
                pn.Tabs(
                    (
                        "Chat",
                        pn.Column(
                            pn.Row(
                                self.hide_step_messages,
                                self.show_tool_calls,
                                self.show_tool_results,
                                sizing_mode="stretch_width",
                            ),
                            pn.Row(
                                pn.Column(self.chat_feed, sizing_mode="stretch_both"),
                                pn.Column(self.detail_pane, sizing_mode="stretch_both"),
                                sizing_mode="stretch_width",
                                min_height=CHAT_VIEW_HEIGHT,
                            ),
                            sizing_mode="stretch_width",
                        ),
                    ),
                    (
                        "Preview",
                        pn.Column(
                            pn.pane.Markdown(CONVERSATION_TEXT),
                            pn.pane.Markdown("### Model Messages"),
                            self.messages_table,
                            pn.pane.Markdown("### Transcript Parts"),
                            self.transcript_table,
                            self.transcript_preview,
                        ),
                    ),
                    sizing_mode="stretch_width",
                ),
            ),
            (
                "Tool Calls",
                pn.Column(pn.pane.Markdown(TOOLS_TEXT), self.tools_table, self.tool_detail),
            ),
            (
                "Workflow & Paths",
                pn.Column(
                    pn.pane.Markdown(WORKFLOW_TEXT),
                    self.workflow_table,
                    pn.pane.Markdown("### Artifact Paths"),
                    self.artifacts_table,
                ),
            ),
            (
                "Tool Outputs",
                pn.Column(pn.pane.Markdown(OUTPUTS_TEXT), self.outputs_table, self.output_detail),
            ),
            (
                "Stored Diffs",
                pn.Column(
                    pn.pane.Markdown(DIFFS_TEXT),
                    self.diff_files_table,
                    self.diff_table,
                    self.diff_preview,
                ),
            ),
            ("Session Logs", pn.Column(pn.pane.Markdown(LOGS_TEXT), self.logs_table)),
            (
                "Raw DB Tables",
                pn.Column(pn.pane.Markdown(RAW_TEXT), self.raw_table_select, self.raw_table),
            ),
            sizing_mode="stretch_width",
        )
        return pn.template.FastListTemplate(
            title="OpenCode Session Dashboard",
            sidebar=[controls],
            main=[
                self.summary,
                pn.pane.Markdown("## Sessions"),
                self.sessions_table,
                pn.pane.Markdown("## Selected Session"),
                tabs,
            ],
            accent_base_color="#2f6f6d",
            header_background="#263238",
        )

    def _clear_detail(self) -> None:
        self.session_meta.object = "No session selected."
        for table in [
            self.messages_table,
            self.transcript_table,
            self.tools_table,
            self.subagents_table,
            self.workflow_table,
            self.artifacts_table,
            self.outputs_table,
            self.diff_files_table,
            self.diff_table,
            self.logs_table,
        ]:
            table.value = pd.DataFrame()
        self.transcript_preview.object = ""
        self.chat_feed.objects = []
        self.detail_pane.objects = [pn.pane.Markdown("_Select a message to see details._")]
        self._parts_df = pd.DataFrame()
        self._messages_df = pd.DataFrame()
        self._transcript_df = pd.DataFrame()
        self.tool_detail.object = ""
        self.subagent_note.object = ""
        self.output_detail.object = ""
        self.diff_preview.object = ""

    def _refresh_summary(self, db_overview: pd.DataFrame) -> None:
        if self._sessions.empty:
            self.summary.object = "No sessions found."
            return

        total_messages = int(self._sessions["message_count"].sum())
        total_parts = int(self._sessions["part_count"].sum())
        total_tools = int(self._sessions["tool_count"].sum())
        total_tasks = int(self._sessions["task_count"].sum())
        child_sessions = int(self._sessions["child_session_count"].sum())
        total_tokens = int(self._sessions["token_total"].sum())
        cost = float(self._sessions["total_cost"].sum())
        latest = self._sessions.iloc[0]["updated"]
        non_empty = int((db_overview["rows"] > 0).sum()) if not db_overview.empty else 0

        self.summary.object = "\n".join(
            [
                "## Overview",
                "",
                f"- Sessions: **{len(self._sessions):,}**",
                f"- Child sessions / task calls: **{child_sessions:,}** / **{total_tasks:,}**",
                (
                    f"- Messages / parts / tools: **{total_messages:,}** / "
                    f"**{total_parts:,}** / **{total_tools:,}**"
                ),
                f"- Tokens: **{total_tokens:,}**",
                f"- Cost: **{cost:.6f}**",
                f"- Non-empty DB tables: **{non_empty:,}**",
                f"- Latest update: **{latest}**",
            ]
        )

    def _refresh_session_selector(self) -> None:
        current = self.session_select.value
        options = {}
        for _, row in self._sessions.iterrows():
            title = row.get("title") or row["id"]
            marker = "child" if row.get("parent_id") else "root"
            label = f"{row['updated']} | {marker} | {title}"
            options[label] = row["id"]
        self.session_select.options = options
        if current in options.values():
            self.session_select.value = current
        elif options:
            self.session_select.value = next(iter(options.values()))

    def _session_row(self, session_id: str) -> pd.Series:
        matches = self._sessions[self._sessions["id"] == session_id]
        if matches.empty:
            return pd.Series({"id": session_id})
        return matches.iloc[0]

    def _session_markdown(self, session: pd.Series) -> str:
        title = session.get("title") or session.get("id", "")
        parent = session.get("parent_id") or ""
        return "\n".join(
            [
                f"### {title}",
                "",
                f"- ID: `{session.get('id', '')}`",
                f"- Parent: `{parent}`" if parent else "- Parent: root session",
                f"- Directory: `{session.get('directory', '')}`",
                f"- Created: **{session.get('created', '')}**",
                f"- Updated: **{session.get('updated', '')}**",
                f"- Duration: **{session.get('duration_min', 0)} min**",
                f"- Version: `{session.get('version', '')}`",
                (
                    f"- Messages / parts / tools: **{int(session.get('message_count', 0)):,}** / "
                    f"**{int(session.get('part_count', 0)):,}** / "
                    f"**{int(session.get('tool_count', 0)):,}**"
                ),
                (
                    f"- Child sessions / task calls / errors: "
                    f"**{int(session.get('child_session_count', 0)):,}** / "
                    f"**{int(session.get('task_count', 0)):,}** / "
                    f"**{int(session.get('error_count', 0)):,}**"
                ),
                f"- Tokens: **{int(session.get('token_total', 0)):,}**",
            ]
        )

    def _filter_transcript(self, transcript: pd.DataFrame) -> pd.DataFrame:
        if not self.hide_step_messages.value or transcript.empty:
            return transcript
        return transcript[~transcript["type"].isin(STEP_MESSAGE_TYPES)].reset_index(drop=True)

    def _filter_parts(self, parts: pd.DataFrame) -> pd.DataFrame:
        if not self.hide_step_messages.value or parts.empty:
            return parts
        return parts[~parts["type"].isin(STEP_MESSAGE_TYPES)].reset_index(drop=True)

    def _rebuild_chat_feed(self) -> None:
        filtered_transcript = self._filter_transcript(self._transcript_df)
        self.chat_feed.objects = self._build_chat_messages(filtered_transcript)

    def _build_chat_messages(self, transcript: pd.DataFrame) -> list[pn.chat.ChatMessage]:
        if transcript.empty or self._parts_df.empty:
            return []

        parts = self._filter_parts(self._parts_df)
        aligned = len(transcript) == len(parts)
        result = []

        for i, (_, row) in enumerate(transcript.iterrows()):
            role = str(row.get("role") or "")
            kind = str(row.get("type") or "")
            tool = str(row.get("tool") or "")
            user_label, avatar, bubble_class = _chat_message_presentation(role, kind, tool)

            # Use full content for text/reasoning; preview summary for tool/patch
            if kind in ("text", "reasoning"):
                bubble_text = str(row.get("content") or row.get("preview") or "")
            else:
                bubble_text = str(row.get("preview") or "")

            part_row = parts.iloc[i] if aligned and i < len(parts) else None
            msg_row = None
            if part_row is not None:
                msg_id = str(part_row.get("message_id") or "")
                if msg_id:
                    matches = self._messages_df[self._messages_df["id"] == msg_id]
                    msg_row = matches.iloc[0] if not matches.empty else None

            content_md = pn.pane.Markdown(
                bubble_text or "_no content_",
                css_classes=["message", bubble_class],
            )

            if part_row is not None:
                btn = pn.widgets.Button(name="Details", button_type="light", width=80, height=26)

                def make_handler(pr: pd.Series, mr: pd.Series | None) -> Any:
                    def handler(event: Any) -> None:
                        self.detail_pane.objects = self._build_detail_panel(pr, mr)

                    return handler

                btn.on_click(make_handler(part_row, msg_row))
                content: Any = pn.Column(
                    content_md, pn.Row(pn.Spacer(), btn), sizing_mode="stretch_width"
                )
            else:
                content = content_md

            show_tool_call = kind != "tool" or self.show_tool_calls.value
            if show_tool_call:
                result.append(
                    pn.chat.ChatMessage(
                        object=content,
                        user=user_label,
                        avatar=avatar,
                        show_reaction_icons=False,
                        show_copy_icon=False,
                        show_timestamp=False,
                        sizing_mode="stretch_width",
                        stylesheets=[CHAT_MESSAGE_STYLESHEET],
                    )
                )
            if (
                part_row is not None
                and kind == "tool"
                and self.show_tool_calls.value
                and self.show_tool_results.value
            ):
                response_text = _tool_response_markdown(part_row)
                if response_text:
                    result.append(
                        pn.chat.ChatMessage(
                            object=pn.pane.Markdown(
                                response_text,
                                css_classes=["message", "opencode-chat-tool-response"],
                            ),
                            user=f"tool response:{tool}" if tool else "tool response",
                            avatar="R",
                            show_reaction_icons=False,
                            show_copy_icon=False,
                            show_timestamp=False,
                            sizing_mode="stretch_width",
                            stylesheets=[CHAT_MESSAGE_STYLESHEET],
                        )
                    )

        return result

    def _build_detail_panel(self, part_row: pd.Series, msg_row: pd.Series | None) -> list[Any]:
        role = _string_value(msg_row.get("role")) if msg_row is not None else ""
        agent = _string_value(msg_row.get("agent")) if msg_row is not None else ""
        model_id = _string_value(msg_row.get("model_id")) if msg_row is not None else ""
        kind = _string_value(part_row.get("type"))
        tool = _string_value(part_row.get("tool"))
        status = _string_value(part_row.get("status"))
        created = _string_value(part_row.get("created"))
        is_tool = bool(part_row.get("is_tool"))
        content = _string_value(part_row.get("content")).strip()
        error = part_row.get("error")

        header = f"## {role} — {kind}"
        if tool:
            header += f" `{tool}`"
        header += f" @ {created}"
        meta = f"**Agent:** {agent}  |  **Model:** {model_id}  |  **Status:** {status}"

        widgets: list[Any] = [pn.pane.Markdown(f"{header}\n\n{meta}\n\n---")]

        if content and not is_tool:
            widgets.append(pn.pane.Markdown("### Content"))
            parsed_content = _try_parse_json(content)
            if isinstance(parsed_content, (dict, list)):
                widgets.append(
                    pn.widgets.JSONEditor(
                        value=parsed_content, mode="view",
                        sizing_mode="stretch_width", height=300,
                    )
                )
            else:
                widgets.append(pn.pane.Markdown(content))

        if is_tool:
            tool_input = part_row.get("input")
            tool_output = part_row.get("output")
            child_session_id = _string_value(part_row.get("task_id")).strip()

            widgets.append(pn.pane.Markdown("### Input"))
            if _has_value(tool_input):
                parsed = _try_parse_json(tool_input)
                if isinstance(parsed, (dict, list)):
                    widgets.append(
                        pn.widgets.JSONEditor(
                            value=parsed, mode="view",
                            sizing_mode="stretch_width", height=300,
                        )
                    )
                else:
                    widgets.append(pn.pane.Markdown(_compact_value_markdown(tool_input)))
            else:
                widgets.append(pn.pane.Markdown("_No input._"))

            if _has_value(tool_output):
                widgets.append(pn.pane.Markdown("### Output"))
                parsed = _try_parse_json(tool_output)
                if _workflow_state_from_value(tool_output):
                    widgets.append(pn.pane.Markdown(_output_markdown(tool_output)))
                if isinstance(parsed, (dict, list)):
                    widgets.append(pn.pane.Markdown("#### Raw Output"))
                    widgets.append(
                        pn.widgets.JSONEditor(
                            value=parsed, mode="view",
                            sizing_mode="stretch_width", height=400,
                        )
                    )
                else:
                    widgets.append(pn.pane.Markdown(_output_markdown(tool_output)))

            if tool == "task" and child_session_id:
                widgets.append(self._agent_conversation_draft(child_session_id))

        if _has_value(error):
            widgets.append(
                pn.pane.Markdown(f"### Error\n\n```\n{_clip(str(error), 3000)}\n```")
            )

        if is_tool:
            widgets.append(pn.pane.Markdown(_sidecar_log_markdown(self._logs_df, part_row)))

        if msg_row is not None:
            cost = float(msg_row.get("cost") or 0)
            token_total = int(msg_row.get("token_total") or 0)
            token_input = int(msg_row.get("token_input") or 0)
            token_output = int(msg_row.get("token_output") or 0)
            cache_read = int(msg_row.get("cache_read") or 0)
            widgets.append(
                pn.pane.Markdown(
                    f"---\n\n### Message Stats\n\n"
                    f"**Cost:** ${cost:.6f}  |  **Tokens:** {token_total:,} "
                    f"(in: {token_input:,}, out: {token_output:,}, cache read: {cache_read:,})"
                )
            )

        return widgets

    def _transcript_markdown(self, transcript: pd.DataFrame) -> str:
        if transcript.empty:
            return "No transcript parts found."

        lines = ["### Transcript Preview", ""]
        for _, row in transcript.head(40).iterrows():
            role = row.get("role") or "part"
            kind = row.get("type") or ""
            tool = f" `{row['tool']}`" if row.get("tool") else ""
            if kind == "tool":
                content = _tool_part_markdown(row.get("content"))
            else:
                content = _clip(str(row.get("content") or "").strip(), 3000)
            lines.extend(
                [
                    f"**{role}** - {kind}{tool} - {row.get('created', '')}",
                    "",
                    content,
                    "",
                ]
            )
        return "\n".join(lines)

    def _agent_conversation_draft(self, child_session_id: str) -> pn.pane.Markdown:
        transcript = self._store.transcript(child_session_id)
        if not transcript.empty:
            transcript = transcript.copy()
            transcript["preview"] = transcript.apply(_readable_part_preview, axis=1)
        return pn.pane.Markdown(_agent_conversation_draft_markdown(child_session_id, transcript))

    def _tools_markdown(self, tools: pd.DataFrame) -> str:
        if tools.empty:
            return "No tool calls found."
        lines = ["### Tool Detail Preview", ""]
        for _, row in tools.head(8).iterrows():
            lines.extend(
                [
                    f"#### `{row.get('tool', '')}` - {row.get('status', '')}",
                    "",
                    _tool_row_markdown(row),
                    "",
                ]
            )
        return "\n".join(lines)

    def _outputs_markdown(self, outputs: pd.DataFrame) -> str:
        if outputs.empty:
            return "No DB-adjacent `tool-output` files found."
        lines = ["### Tool Output Files", ""]
        for _, row in outputs.head(5).iterrows():
            lines.extend(
                [
                    f"#### `{row.get('file', '')}` ({int(row.get('bytes', 0)):,} bytes)",
                    "",
                    f"```\n{_clip(row.get('content') or '')}\n```",
                    "",
                ]
            )
        return "\n".join(lines)

    def _subagent_note(self, subagents: pd.DataFrame, session: pd.Series) -> str:
        if subagents.empty:
            return (
                "### Persisted Task/Subagent Links\n\n"
                "No persisted subagent records found for this session: there are no child "
                "sessions and no `task` tool outputs with saved `task_id` links. This is "
                "expected for DBs such as the ICA run when they only log available "
                "task-capable agents."
            )
        return (
            "### Persisted Task/Subagent Links\n\n"
            f"Found **{len(subagents):,}** persisted child-session/task links for "
            f"`{session.get('id', '')}`."
        )

    def _diff_markdown(self, diffs: pd.DataFrame) -> str:
        if diffs.empty:
            return "No stored session diff found."
        lines = []
        for _, diff in diffs.head(3).iterrows():
            patch = str(diff.get("patch") or "")
            lines.extend(
                [
                    f"### Diff: `{diff.get('file', '')}`",
                    "",
                    f"```diff\n{_clip(patch, 8000)}\n```",
                    "",
                ]
            )
        return "\n".join(lines)

    def _display_frame(self, frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame(columns=[COLUMN_LABELS.get(column, column) for column in columns])
        present = [column for column in columns if column in frame.columns]
        return frame[present].copy().rename(columns=COLUMN_LABELS)

    def _set_ok(self, message: str) -> None:
        self.status.object = message
        self.status.alert_type = "success"
        self.status.visible = True

    def _set_error(self, message: str) -> None:
        self.status.object = f"Database error: `{message}`"
        self.status.alert_type = "danger"
        self.status.visible = True


def _try_parse_json(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            print(f"Value is not valid JSON: {value}")
            pass
    return value


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value == ""
    if isinstance(value, dict | list | tuple | set):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _has_value(value: Any) -> bool:
    return not _is_missing(value)


def _string_value(value: Any) -> str:
    return "" if _is_missing(value) else str(value)


def _numeric_ms(value: Any) -> int | None:
    if _is_missing(value):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return None


def _clip(value: Any, limit: int = 5000) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n..."


def _discover_db_options(root_value: str) -> dict[str, str]:
    if not root_value.strip():
        return {}

    root = Path(root_value).expanduser()
    if not root.exists():
        return {}

    candidates = [root / "opencode.db", root / "opencode" / "opencode.db"]
    candidates.extend(child / "opencode" / "opencode.db" for child in sorted(root.iterdir()))

    options = {}
    seen_paths = set()
    for db_path in candidates:
        if not db_path.is_file() or db_path in seen_paths:
            continue
        seen_paths.add(db_path)
        run_dir = db_path.parent.parent if db_path.parent.name == "opencode" else db_path.parent
        label = run_dir.name or str(db_path)
        if label in options:
            label = str(db_path)
        options[label] = str(db_path)
    return options


def _initial_db_inputs(
    db_path: str | Path | None,
    db_root: str | Path | None,
) -> tuple[str, str, bool]:
    root = Path(db_root).expanduser() if db_root is not None else DEFAULT_DB_ROOT.expanduser()
    db_value = str(resolve_db_path(db_path))
    select_first = db_path is None

    if db_path is not None:
        candidate = Path(db_path).expanduser()
        if candidate.is_dir():
            root = candidate
            options = _discover_db_options(str(root))
            if options:
                db_value = next(iter(options.values()))
                select_first = False
            else:
                db_value = str(candidate)
    elif db_root is not None:
        options = _discover_db_options(str(root))
        if options:
            db_value = next(iter(options.values()))
            select_first = False

    return db_value, str(root), select_first


def _decode_jsonish(value: Any, max_depth: int = 3) -> Any:
    current = value
    for _ in range(max_depth):
        if not isinstance(current, str):
            return current
        stripped = current.strip()
        if not stripped or stripped[0] not in "[{\"":
            return current
        try:
            current = json.loads(stripped)
        except json.JSONDecodeError:
            return current
    return current


def _readable_part_preview(row: pd.Series) -> str:
    if row.get("type") != "tool":
        return str(row.get("preview") or "")
    payload = _decode_jsonish(row.get("content"))
    if not isinstance(payload, dict):
        return str(row.get("preview") or "")
    return _readable_tool_preview(payload)


def _readable_tool_preview(row: pd.Series | dict[str, Any]) -> str:
    tool = str(row.get("tool") or "tool")
    status = str(row.get("status") or "")
    output = row.get("output") if "output" in row else row.get("output_text")
    summary = _tool_output_summary(output)
    prefix = f"{tool} {status}".strip()
    return _clip(f"{prefix}: {summary}" if summary else prefix, 280)


def _chat_message_presentation(role: str, kind: str, tool: str) -> tuple[str, str, str]:
    if kind == "tool":
        label = f"tool:{tool}" if tool else "tool"
        return label, "🔧", "opencode-chat-tool"
    if kind == "reasoning":
        return "reasoning", "💭", "opencode-chat-reasoning"
    if kind == "patch":
        return "patch", "📝", "opencode-chat-patch"
    if kind == "step-start":
        return "step:start", "S", "opencode-chat-step-start"
    if kind == "step-finish":
        return "step:finish", "F", "opencode-chat-step-finish"
    if role == "user":
        return "user", "👤", "opencode-chat-user"
    if role == "assistant" or kind == "text":
        return "assistant", "🤖", "opencode-chat-assistant"
    return kind or role or "part", "?", "opencode-chat-other"


def _tool_part_markdown(content: Any) -> str:
    payload = _decode_jsonish(content)
    if not isinstance(payload, dict):
        return f"```\n{_clip(content, 3000)}\n```"
    return _tool_payload_markdown(payload)


def _tool_row_markdown(row: pd.Series) -> str:
    payload = {
        "tool": row.get("tool"),
        "status": row.get("status"),
        "input": row.get("input"),
        "output": row.get("output"),
        "error": row.get("error"),
    }
    return _tool_payload_markdown(payload)


def _tool_response_markdown(row: pd.Series | dict[str, Any]) -> str:
    error = row.get("error")
    if _has_value(error):
        return f"**Error**\n\n```\n{_clip(str(error), 3000)}\n```"

    output = row.get("output")
    if not _has_value(output):
        return ""
    return _output_markdown(output)


def _tool_payload_markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"**Tool:** `{payload.get('tool') or ''}`",
        f"**Status:** `{payload.get('status') or ''}`",
    ]
    if _has_value(payload.get("input")):
        lines.extend(["", "**Input**", "", _compact_value_markdown(payload.get("input"))])
    if _has_value(payload.get("error")):
        lines.extend(["", "**Error**", "", f"```\n{_clip(payload.get('error'), 3000)}\n```"])
    elif _has_value(payload.get("output")):
        lines.extend(["", "**Output**", "", _output_markdown(payload.get("output"))])
    return "\n".join(lines)


def _compact_value_markdown(value: Any) -> str:
    decoded = _decode_jsonish(value)
    if isinstance(decoded, dict):
        lines = []
        for key, item in decoded.items():
            if isinstance(item, dict | list):
                lines.append(f"- `{key}`: `{_preview_json(item)}`")
            else:
                lines.append(f"- `{key}`: `{item}`")
        return "\n".join(lines) if lines else "_Empty object._"
    if isinstance(decoded, list):
        return "\n".join(f"- `{item}`" for item in decoded[:20])
    return f"```\n{_clip(decoded, 3000)}\n```"


def _workflow_state_from_value(value: Any) -> dict[str, Any] | None:
    decoded = _decode_jsonish(value)
    if not isinstance(decoded, dict):
        return None
    state = decoded.get("state")
    if isinstance(state, dict) and isinstance(state.get("phases"), dict):
        return state
    if isinstance(decoded.get("phases"), dict):
        return decoded
    return None


def _output_markdown(value: Any) -> str:
    decoded = _decode_jsonish(value)
    state = _workflow_state_from_value(decoded)
    if state is not None:
        return _workflow_state_markdown(state)
    if isinstance(decoded, dict | list):
        return f"```json\n{_clip(_preview_json(decoded, pretty=True), 5000)}\n```"
    return f"```\n{_clip(decoded, 3000)}\n```"


def _tool_output_summary(value: Any) -> str:
    if not _has_value(value):
        return ""
    decoded = _decode_jsonish(value)
    state = _workflow_state_from_value(decoded)
    if isinstance(state, dict):
        phases = state.get("phases") if isinstance(state.get("phases"), dict) else {}
        current = state.get("current_phase") or "unknown phase"
        status = state.get("overall_status") or state.get("final_status") or "unknown status"
        if phases:
            return f"workflow {status}, current {current}, {len(phases)} phases"
        return f"workflow {status}, current {current}"
    if isinstance(decoded, dict):
        return _preview_json(decoded)
    if isinstance(decoded, list):
        return f"{len(decoded)} items"
    return str(decoded or "")


def _agentic_log_markdown(logs: pd.DataFrame, limit: int = 8) -> str:
    if logs.empty or "service" not in logs.columns or "text" not in logs.columns:
        return "### Agentic Logs\n\n_No session agentic logs found._"

    agentic_logs = logs[logs["service"].astype(str).isin(AGENTIC_LOG_SERVICES)]
    if agentic_logs.empty:
        return "### Agentic Logs\n\n_No session agentic logs found._"

    lines = ["### Agentic Logs", ""]
    for _, row in agentic_logs.head(limit).iterrows():
        service = str(row.get("service") or "")
        preview = _clip(row.get("text") or "", 700)
        lines.append(f"- `{service}`: {preview}")
    return "\n".join(lines)


def _sidecar_log_markdown(
    logs: pd.DataFrame,
    part_row: pd.Series | None,
    *,
    limit: int = 24,
    window_seconds: int = 90,
) -> str:
    if logs.empty or "text" not in logs.columns:
        return "### Sidecar Logs\n\n_No sidecar logs found for this session._"

    scoped = logs.copy()
    scoped_text = scoped["text"].astype(str)
    exact_mask = pd.Series(False, index=scoped.index)
    selected_ids: list[str] = []
    selected_at = ""

    if part_row is not None:
        selected_at = _string_value(part_row.get("created"))
        for key in ("id", "message_id"):
            value = _string_value(part_row.get(key)).strip()
            if value:
                selected_ids.append(value)
        if selected_ids:
            exact_mask = scoped_text.apply(lambda text: any(item in text for item in selected_ids))

    target_ms = _numeric_ms(part_row.get("created_ms")) if part_row is not None else None
    note = "_Showing the first sidecar log lines for this session._"
    selected = pd.DataFrame()

    if target_ms is not None and "timestamp_ms" in scoped.columns:
        timestamp_ms = pd.to_numeric(scoped["timestamp_ms"], errors="coerce")
        scoped["_distance_ms"] = (timestamp_ms - target_ms).abs()
        time_mask = timestamp_ms.notna() & (scoped["_distance_ms"] <= window_seconds * 1000)
        selected = scoped[time_mask | exact_mask].copy()
        if selected.empty and timestamp_ms.notna().any():
            selected = scoped[timestamp_ms.notna()].nsmallest(limit, "_distance_ms").copy()
            nearest = int((selected["_distance_ms"].min() or 0) / 1000) if not selected.empty else 0
            note = (
                f"_No sidecar logs within +/-{window_seconds}s of the selected part; "
                f"showing nearest lines ({nearest}s away)._"
            )
        else:
            note = f"_Showing sidecar log lines within +/-{window_seconds}s of the selected part"
            if exact_mask.any():
                note += " plus exact message/part id matches"
            note += "._"
    elif exact_mask.any():
        selected = scoped[exact_mask].copy()
        note = "_Showing sidecar log lines that mention the selected message or part id._"
    else:
        selected = scoped.head(limit).copy()

    if selected.empty:
        return "### Sidecar Logs\n\n_No matching sidecar log lines found for this selected part._"

    if {"file", "line"}.issubset(selected.columns):
        selected = selected.drop_duplicates(subset=["file", "line"])
    if "_distance_ms" in selected.columns and len(selected) > limit:
        exact_selected = selected[exact_mask.reindex(selected.index, fill_value=False)]
        if exact_selected.empty:
            selected = selected.nsmallest(limit, "_distance_ms")
        elif len(exact_selected) >= limit:
            selected = exact_selected.nsmallest(limit, "_distance_ms")
        else:
            nearby_selected = selected.drop(exact_selected.index).nsmallest(
                limit - len(exact_selected),
                "_distance_ms",
            )
            selected = pd.concat([exact_selected, nearby_selected])
    elif len(selected) > limit:
        selected = selected.head(limit)

    sort_columns = [column for column in ("timestamp_ms", "file", "line") if column in selected]
    if sort_columns:
        selected = selected.sort_values(sort_columns, na_position="last")

    lines = ["### Sidecar Logs", ""]
    if selected_at:
        lines.extend([f"_Selected part time: `{selected_at}`._", ""])
    lines.extend([note, ""])
    for _, row in selected.iterrows():
        location = f"{_string_value(row.get('file'))}:{_string_value(row.get('line'))}".strip(":")
        timestamp = _string_value(row.get("timestamp"))
        service = _string_value(row.get("service"))
        level = _string_value(row.get("level"))
        label = " | ".join(item for item in [timestamp, level, service, location] if item)
        lines.extend(
            [
                f"#### {label or 'log line'}",
                "",
                f"```text\n{_clip(_string_value(row.get('text')), 1600)}\n```",
                "",
            ]
        )
    return "\n".join(lines).strip()


def _agent_conversation_draft_markdown(
    child_session_id: str,
    transcript: pd.DataFrame,
    *,
    limit: int = 40,
) -> str:
    header = f"### Agent Conversation Draft `{child_session_id}`"
    if transcript.empty:
        return f"{header}\n\n_No child transcript found._"

    lines = [header, ""]
    for _, row in transcript.head(limit).iterrows():
        role = str(row.get("role") or "part")
        agent = str(row.get("agent") or "")
        kind = str(row.get("type") or "")
        tool = str(row.get("tool") or "")
        status = str(row.get("status") or "")
        created = str(row.get("created") or "")

        if kind == "tool":
            label = f"tool `{tool}`" if tool else "tool"
            if status:
                label = f"{label} - {status}"
            content = _clip(_readable_part_preview(row), 1200)
        else:
            label_parts = [role]
            if agent:
                label_parts.append(agent)
            if kind:
                label_parts.append(kind)
            label = " - ".join(label_parts)
            content = _clip(str(row.get("content") or row.get("preview") or "").strip(), 1500)

        if created:
            label = f"{label} - {created}"
        lines.extend([f"#### {label}", "", content or "_No content._", ""])

    hidden = max(len(transcript) - limit, 0)
    if hidden:
        lines.append(f"_Showing first {limit} parts; {hidden} later parts omitted._")
    return "\n".join(lines).strip()


def _workflow_state_markdown(state: dict[str, Any]) -> str:
    phases = state.get("phases") or {}
    phase_order = state.get("phase_order") or list(phases)
    status_counts: dict[str, int] = {}
    for phase in phases.values():
        status = str(phase.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1

    lines = [
        f"- Overall status: `{state.get('overall_status') or state.get('final_status') or ''}`",
        f"- Current phase: `{state.get('current_phase') or ''}`",
        f"- Run mode: `{state.get('run_mode') or ''}`",
        f"- HP mode: `{state.get('hp_mode') or ''}`",
        f"- Phase counts: `{', '.join(f'{key}: {value}' for key, value in status_counts.items())}`",
        "",
        "| Phase | Status | Attempts | Artifacts |",
        "| --- | --- | ---: | ---: |",
    ]
    for name in phase_order:
        phase = phases.get(name) or {}
        artifact_count = len(phase.get("artifact_paths") or [])
        lines.append(
            f"| `{name}` | `{phase.get('status') or ''}` | "
            f"{phase.get('attempts') or 0} | {artifact_count} |"
        )
    if state.get("blocker"):
        lines.extend(["", f"**Blocker:** `{state.get('blocker')}`"])
    return "\n".join(lines)


def _preview_json(value: Any, *, pretty: bool = False) -> str:
    indent = 2 if pretty else None
    return json.dumps(value, ensure_ascii=False, indent=indent, default=str)


def build_app(
    db_path: str | Path | None = None,
    db_root: str | Path | None = None,
) -> pn.template.FastListTemplate:
    return OpenCodeDashboard(db_path, db_root).view()


def _served_app_inputs(argv: list[str] | None = None) -> tuple[str | None, str | None]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--db", default=None)
    parser.add_argument("--db-root", default=None)
    args, _ = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    return args.db, args.db_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the OpenCode Panel dashboard.")
    parser.add_argument(
        "--db",
        default=None,
        help="Path to opencode.db, a run folder, or a folder containing run folders",
    )
    parser.add_argument(
        "--db-root",
        default=None,
        help="Folder containing run folders with opencode/opencode.db files",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", default=5006, type=int, help="Port to bind")
    parser.add_argument("--show", action="store_true", help="Open the dashboard in a browser")
    args = parser.parse_args()

    pn.serve(
        {"/": build_app(args.db, args.db_root)},
        address=args.host,
        port=args.port,
        show=args.show,
        title="OpenCode Session Dashboard",
    )


app = build_app(*_served_app_inputs())
app.servable(title="OpenCode Session Dashboard")


if __name__ == "__main__":
    main()
