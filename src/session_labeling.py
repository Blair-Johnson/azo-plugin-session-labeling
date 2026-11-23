"""Session labeling add-on for the orchestrated agent.

Adds tools that let the agent name and describe the current persisted session so
users can recognize sessions while browsing or resuming work.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

_PLUGIN_NAME = "azo-plugin-session-labeling"
_AGENT_UTILS_MIN_VERSION = "0.4.0"


class _PluginCompatibilityError(RuntimeError):
    agent_zoo_plugin_compatibility = True


def _version_tuple(raw: str) -> tuple[int, ...]:
    return tuple(int(part) for part in str(raw).split(".") if part.isdigit())


try:
    from agent_utils import RENDER_CHANNEL_API_VERSION
    from agent_utils.core.events import Event, RenderRequested
    from agent_utils.core.pipeline import EmitFn, Feature
    from agent_utils.core.state import Entry, MODEL_RENDER_CHANNEL, State
    from agent_utils.core.tool import Tool
    from agent_utils.features.core import (
        ExcludeForgotten,
        MessageRenderer,
        RenderTransformRegistrar,
    )
except (ImportError, AttributeError) as exc:
    raise _PluginCompatibilityError(
        f"{_PLUGIN_NAME} requires agent-utils>={_AGENT_UTILS_MIN_VERSION} with "
        "RenderTransformRegistrar, MODEL_RENDER_CHANNEL, and "
        "Entry.append_render_transform; upgrade agent-utils before reloading."
    ) from exc

if (
    _version_tuple(RENDER_CHANNEL_API_VERSION) < _version_tuple(_AGENT_UTILS_MIN_VERSION)
    or MODEL_RENDER_CHANNEL != "model"
    or not hasattr(Entry, "append_render_transform")
):
    raise _PluginCompatibilityError(
        f"{_PLUGIN_NAME} requires agent-utils>={_AGENT_UTILS_MIN_VERSION} with "
        "RenderTransformRegistrar, MODEL_RENDER_CHANNEL, and "
        "Entry.append_render_transform; upgrade agent-utils before reloading."
    )

from agent_zoo.storage.projects import list_projects, list_sessions, register_session

_TITLE_MAX_CHARS = 80
_DESCRIPTION_MAX_CHARS = 240
_DESCRIPTION_MAX_WORDS = 7
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WHITESPACE_RE = re.compile(r"\s+")
_SESSION_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_]*$")

_SYSTEM_PROMPT_BLOCK = """

# Session labeling
Once the conversation has started and the line of work has crystallized, use
`name_session` to set a short, stable session name and/or concise description.
Session names must be one token with no spaces; use underscores as delimiters,
e.g. `session_labeling_addon`. Keep names fairly stable unless the focus changes
substantially. Descriptions should describe the current work, status, or next
step in fewer than 8 words, and may be updated as the work changes.
""".rstrip()


def _normalize_text(
    value: str, *, field: str, max_chars: int
) -> tuple[str | None, str | None]:
    text = str(value or "")
    if _CONTROL_RE.search(text):
        return None, f"ERROR: {field} contains unsupported control characters."
    text = _WHITESPACE_RE.sub(" ", text).strip()
    if not text:
        return None, f"ERROR: {field} must be non-empty."
    if len(text) > max_chars:
        return (
            None,
            f"ERROR: {field} is {len(text)} characters; keep it at or below "
            f"{max_chars} characters.",
        )
    return text, None


def _validate_session_name(name: str) -> tuple[str | None, str | None]:
    title, error = _normalize_text(
        name, field="session name", max_chars=_TITLE_MAX_CHARS
    )
    if error:
        return None, error
    if " " in title or not _SESSION_NAME_RE.fullmatch(title):
        return (
            None,
            "ERROR: session name must use only letters, numbers, and underscores; "
            "use underscores instead of spaces.",
        )
    return title, None


def _validate_description(description: str) -> tuple[str | None, str | None]:
    desc, error = _normalize_text(
        description,
        field="session description",
        max_chars=_DESCRIPTION_MAX_CHARS,
    )
    if error:
        return None, error
    words = desc.split()
    if len(words) > _DESCRIPTION_MAX_WORDS:
        return (
            None,
            f"ERROR: session description is {len(words)} words; keep it under 8 words.",
        )
    return desc, None


def _current_session_id(state: Any) -> str:
    run_db = getattr(state, "run_db", None)
    sid = str(getattr(run_db, "session_id", "") or "").strip()
    if sid:
        return sid
    return str(getattr(state, "_session_id", "") or "").strip()


def _project_from_state(state: Any) -> str:
    """Best-effort project name for the currently running session."""
    if state is None:
        return ""

    for attr in ("_project_name", "project_name"):
        value = str(getattr(state, attr, "") or "").strip()
        if value:
            return value

    context = getattr(state, "_agent_zoo_context", None)
    if isinstance(context, dict):
        value = str(context.get("project_name") or "").strip()
        if value:
            return value

    run_db = getattr(state, "run_db", None)
    agent_db = getattr(run_db, "agent_db", None)
    db_path = str(getattr(agent_db, "path", "") or "").strip()
    if db_path:
        path = Path(db_path).expanduser()
        if path.name == "db.sqlite" and path.parent.name:
            return path.parent.name

    return ""


def _project_for_session(
    session_id: str, state: Any = None
) -> tuple[str | None, str | None]:
    if not session_id:
        return None, "ERROR: Could not determine the current session id."

    runtime_project = _project_from_state(state)
    if runtime_project:
        return runtime_project, None

    matches: list[str] = []
    for project in list_projects():
        name = str(project.get("name", "") or "")
        if not name:
            continue
        try:
            sessions = list_sessions(name)
        except Exception:
            continue
        if any(
            str(entry.get("session_id", "") or "") == session_id for entry in sessions
        ):
            matches.append(name)

    if len(matches) == 1:
        return matches[0], None
    if len(matches) > 1:
        return (
            None,
            "ERROR: Current session id appears in multiple projects: "
            + ", ".join(matches)
            + ". Use the /session command manually for this unusual case.",
        )
    return (
        None,
        "ERROR: Could not determine the current project for this session. "
        "The session id is available, but no runtime project metadata or project DB path "
        "was found. Use /session name manually for this unusual case.",
    )


class NameSession(Tool):
    """Set the current session's name and/or description."""

    name = "name_session"
    optional_reads = {"_session_id", "_project_name", "_agent_zoo_context", "run_db"}
    writes = {"_session_title", "_session_description"}

    def call(
        self,
        state: State,
        name: str | None = None,
        description: str | None = None,
    ) -> str:
        """Set the current session's name and/or description.

        Use this after the user's goal has become clear, not at the very beginning
        of a conversation. Choose a stable name that helps the user identify the
        session in session lists and resume menus. Session names must be one token
        with no spaces; use underscores as delimiters, such as
        "session_labeling_addon" or "debug_login_retry_loop". Avoid vague names
        like "work_session", avoid changing the name for minor pivots, and do not
        include newlines or control characters.

        Use the description to keep the session browser in sync with the current
        line of work. Descriptions may be updated periodically as the scope, status,
        or next steps change. Keep descriptions concrete and under 8 words.

        Args:
            name: Optional session name/title to show in session browsers. Use only
                letters, numbers, and underscores; no spaces.
            description: Optional concise description of the current work, status,
                or next step for the session browser. Must be fewer than 8 words.
        """
        if name is None and description is None:
            return "ERROR: provide at least one of name or description."

        updates: dict[str, str] = {}
        messages: list[str] = []

        if name is not None:
            title, error = _validate_session_name(name)
            if error:
                return error
            updates["title"] = title
            messages.append(f"Session name set to: {title}")

        if description is not None:
            desc, error = _validate_description(description)
            if error:
                return error
            updates["description"] = desc
            messages.append(f"Session description set to: {desc}")

        sid = _current_session_id(state)
        project, error = _project_for_session(sid, state)
        if error:
            return error

        register_session(project, sid, **updates)
        if state is not None:
            if "title" in updates:
                state._session_title = updates["title"]
            if "description" in updates:
                state._session_description = updates["description"]
        return "; ".join(messages)


name_session = NameSession()


class SessionLabelingPromptInstructions(RenderTransformRegistrar):
    """Append session-labeling guidance to the system prompt's model channel."""

    reads = {"entries"}
    writes = {"entries"}

    @staticmethod
    def _make_transform(entry: Entry):
        def transform(
            messages: list[dict[str, Any]],
            state: State,
            *,
            _entry: Entry = entry,
        ) -> list[dict[str, Any]]:
            del state
            if not messages or _entry.role != "system":
                return messages
            first = dict(messages[0])
            if first.get("role") != "system":
                return messages
            content = str(first.get("content") or "")
            if "# Session labeling" in content:
                return messages
            first["content"] = content + _SYSTEM_PROMPT_BLOCK
            return [first, *messages[1:]]

        return transform

    def handle(self, event: Event, state: State, emit: EmitFn) -> None:
        del emit
        assert isinstance(event, RenderRequested)
        if event.channels is not None and MODEL_RENDER_CHANNEL not in event.channels:
            return
        if not getattr(state, "entries", None):
            return
        entry = state.entries[0]
        entry.append_render_transform(
            MODEL_RENDER_CHANNEL,
            self._make_transform(entry),
            name="session-labeling system prompt",
        )


def register_features(builder: Any, *, session: Any, config: Any) -> None:
    del session, config
    builder.add(
        Feature(
            name="session_labeling_addon",
            components=[
                NameSession(),
                SessionLabelingPromptInstructions(),
            ],
            order=[
                SessionLabelingPromptInstructions,
                ExcludeForgotten,
                MessageRenderer,
            ],
        )
    )
