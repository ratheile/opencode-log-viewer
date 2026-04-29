from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import panel as pn

from opencode_viewer.db import OpenCodeStore, resolve_db_path

pn.extension("tabulator", sizing_mode="stretch_width")


SESSION_COLUMNS = [
    "updated",
    "title",
    "directory",
    "message_count",
    "part_count",
    "summary_files",
    "summary_additions",
    "summary_deletions",
    "token_total",
    "total_cost",
]


MESSAGE_COLUMNS = [
    "created",
    "role",
    "agent",
    "mode",
    "model_id",
    "finish",
    "cost",
    "token_total",
    "token_input",
    "token_output",
    "token_reasoning",
]


TRANSCRIPT_COLUMNS = ["created", "role", "agent", "type", "tool", "status", "preview"]


class OpenCodeDashboard:
    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_input = pn.widgets.TextInput(
            name="Database",
            value=str(resolve_db_path(db_path)),
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

        self.status = pn.pane.Alert(
            "",
            alert_type="light",
            visible=False,
            sizing_mode="stretch_width",
        )
        self.summary = pn.pane.Markdown("", sizing_mode="stretch_width")
        self.sessions_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            height=360,
            disabled=True,
            pagination="local",
            page_size=12,
            show_index=False,
            sizing_mode="stretch_width",
        )
        self.session_meta = pn.pane.Markdown("", sizing_mode="stretch_width")
        self.messages_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            height=320,
            disabled=True,
            pagination="local",
            page_size=10,
            show_index=False,
            sizing_mode="stretch_width",
        )
        self.transcript_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            height=380,
            disabled=True,
            pagination="local",
            page_size=12,
            show_index=False,
            sizing_mode="stretch_width",
        )
        self.transcript_preview = pn.pane.Markdown("", sizing_mode="stretch_width")
        self.diff_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            height=240,
            disabled=True,
            pagination="local",
            page_size=8,
            show_index=False,
            sizing_mode="stretch_width",
        )
        self.diff_preview = pn.pane.Markdown("", sizing_mode="stretch_width")

        self._sessions = pd.DataFrame()
        self._store = self._new_store()

        self.refresh_button.on_click(self.refresh)
        self.include_archived.param.watch(lambda _: self.refresh(), "value")
        self.session_select.param.watch(lambda _: self.refresh_detail(), "value")

        self.refresh()

    def _new_store(self) -> OpenCodeStore:
        return OpenCodeStore(self.db_input.value)

    def refresh(self, event: Any | None = None) -> None:
        del event
        self._store = self._new_store()
        try:
            self._sessions = self._store.sessions(include_archived=self.include_archived.value)
        except Exception as exc:  # noqa: BLE001 - surface DB errors in the dashboard
            self._set_error(str(exc))
            self._sessions = pd.DataFrame()
            self.sessions_table.value = pd.DataFrame()
            self.session_select.options = {}
            self.summary.object = ""
            self.refresh_detail()
            return

        self._set_ok(f"Loaded {len(self._sessions)} sessions from `{self._store.db_path}`.")
        self._refresh_summary()
        self._refresh_session_selector()
        self._refresh_sessions_table()
        self.refresh_detail()

    def refresh_detail(self) -> None:
        session_id = self.session_select.value
        if not session_id:
            self.session_meta.object = "No session selected."
            self.messages_table.value = pd.DataFrame()
            self.transcript_table.value = pd.DataFrame()
            self.transcript_preview.object = ""
            self.diff_table.value = pd.DataFrame()
            self.diff_preview.object = ""
            return

        session = self._session_row(session_id)
        self.session_meta.object = self._session_markdown(session)

        messages = self._store.messages(session_id)
        self.messages_table.value = self._display_frame(messages, MESSAGE_COLUMNS)

        transcript = self._store.transcript(session_id)
        self.transcript_table.value = self._display_frame(transcript, TRANSCRIPT_COLUMNS)
        self.transcript_preview.object = self._transcript_markdown(transcript)

        diffs = self._store.session_diff(session_id)
        self.diff_table.value = self._display_frame(
            diffs,
            ["file", "status", "additions", "deletions"],
        )
        self.diff_preview.object = self._diff_markdown(diffs)

    def view(self) -> pn.template.FastListTemplate:
        controls = pn.Column(
            self.db_input,
            pn.Row(self.refresh_button, self.include_archived),
            self.session_select,
            self.status,
            sizing_mode="stretch_width",
        )
        tabs = pn.Tabs(
            ("Messages", self.messages_table),
            ("Transcript", pn.Column(self.transcript_table, self.transcript_preview)),
            ("Diffs", pn.Column(self.diff_table, self.diff_preview)),
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
                self.session_meta,
                tabs,
            ],
            accent_base_color="#2f6f6d",
            header_background="#263238",
        )

    def _refresh_summary(self) -> None:
        if self._sessions.empty:
            self.summary.object = "No sessions found."
            return

        total_messages = int(self._sessions["message_count"].sum())
        total_parts = int(self._sessions["part_count"].sum())
        additions = int(self._sessions["summary_additions"].sum())
        deletions = int(self._sessions["summary_deletions"].sum())
        total_tokens = int(self._sessions["token_total"].sum())
        cost = float(self._sessions["total_cost"].sum())
        latest = self._sessions.iloc[0]["updated"]

        self.summary.object = "\n".join(
            [
                "## Overview",
                "",
                f"- Sessions: **{len(self._sessions):,}**",
                f"- Messages / parts: **{total_messages:,}** / **{total_parts:,}**",
                f"- Files changed: **{int(self._sessions['summary_files'].sum()):,}**",
                f"- Additions / deletions: **{additions:,}** / **{deletions:,}**",
                f"- Tokens: **{total_tokens:,}**",
                f"- Cost: **{cost:.6f}**",
                f"- Latest update: **{latest}**",
            ]
        )

    def _refresh_session_selector(self) -> None:
        current = self.session_select.value
        options = {}
        for _, row in self._sessions.iterrows():
            title = row.get("title") or row["id"]
            label = f"{row['updated']} | {title}"
            options[label] = row["id"]
        self.session_select.options = options
        if current in options.values():
            self.session_select.value = current
        elif options:
            self.session_select.value = next(iter(options.values()))

    def _refresh_sessions_table(self) -> None:
        self.sessions_table.value = self._display_frame(self._sessions, SESSION_COLUMNS)

    def _session_row(self, session_id: str) -> pd.Series:
        matches = self._sessions[self._sessions["id"] == session_id]
        if matches.empty:
            return pd.Series({"id": session_id})
        return matches.iloc[0]

    def _session_markdown(self, session: pd.Series) -> str:
        title = session.get("title") or session.get("id", "")
        return "\n".join(
            [
                f"### {title}",
                "",
                f"- ID: `{session.get('id', '')}`",
                f"- Directory: `{session.get('directory', '')}`",
                f"- Created: **{session.get('created', '')}**",
                f"- Updated: **{session.get('updated', '')}**",
                f"- Duration: **{session.get('duration_min', 0)} min**",
                f"- Version: `{session.get('version', '')}`",
                (
                    f"- Messages / parts: **{int(session.get('message_count', 0)):,}** / "
                    f"**{int(session.get('part_count', 0)):,}**"
                ),
                f"- Tokens: **{int(session.get('token_total', 0)):,}**",
            ]
        )

    def _transcript_markdown(self, transcript: pd.DataFrame) -> str:
        if transcript.empty:
            return ""

        lines = ["### Transcript Preview", ""]
        for _, row in transcript.head(12).iterrows():
            role = row.get("role") or "part"
            kind = row.get("type") or ""
            tool = f" `{row['tool']}`" if row.get("tool") else ""
            content = str(row.get("content") or "").strip()
            if len(content) > 1400:
                content = f"{content[:1400]}..."
            lines.extend([f"**{role}** · {kind}{tool} · {row.get('created', '')}", "", content, ""])
        return "\n".join(lines)

    def _diff_markdown(self, diffs: pd.DataFrame) -> str:
        if diffs.empty:
            return "No stored session diff found."
        first = diffs.iloc[0]
        patch = str(first.get("patch") or "")
        if len(patch) > 8000:
            patch = f"{patch[:8000]}\n..."
        return f"### First Diff: `{first.get('file', '')}`\n\n```diff\n{patch}\n```"

    def _display_frame(self, frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame(columns=columns)
        present = [column for column in columns if column in frame.columns]
        return frame[present].copy()

    def _set_ok(self, message: str) -> None:
        self.status.object = message
        self.status.alert_type = "success"
        self.status.visible = True

    def _set_error(self, message: str) -> None:
        self.status.object = f"Database error: `{message}`"
        self.status.alert_type = "danger"
        self.status.visible = True


def build_app(db_path: str | Path | None = None) -> pn.template.FastListTemplate:
    return OpenCodeDashboard(db_path).view()


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the OpenCode Panel dashboard.")
    parser.add_argument("--db", default=None, help="Path to opencode.db")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", default=5006, type=int, help="Port to bind")
    parser.add_argument("--show", action="store_true", help="Open the dashboard in a browser")
    args = parser.parse_args()

    pn.serve(
        {"/": build_app(args.db)},
        address=args.host,
        port=args.port,
        show=args.show,
        title="OpenCode Session Dashboard",
    )


app = build_app()
app.servable(title="OpenCode Session Dashboard")


if __name__ == "__main__":
    main()
