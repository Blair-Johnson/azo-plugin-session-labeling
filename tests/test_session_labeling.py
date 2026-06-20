from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import src.session_labeling as plugin


def test_name_session_registers_first_turn_session_from_run_db_path(tmp_path):
    db_path = tmp_path / "projects" / "demo" / "db.sqlite"
    state = SimpleNamespace(
        run_db=SimpleNamespace(
            session_id="session-1",
            agent_db=SimpleNamespace(path=str(db_path)),
        )
    )
    calls = []

    with patch.object(plugin, "list_projects", return_value=[]), patch.object(
        plugin,
        "register_session",
        side_effect=lambda *args, **kwargs: calls.append((args, kwargs)),
    ):
        result = plugin.name_session.fn(
            name="first_turn_label",
            description="Initial investigation",
            state=state,
        )

    assert result == (
        "Session name set to: first_turn_label; "
        "Session description set to: Initial investigation"
    )
    assert calls == [(
        ("demo", "session-1"),
        {"title": "first_turn_label", "description": "Initial investigation"},
    )]
    assert state._session_title == "first_turn_label"
    assert state._session_description == "Initial investigation"


def test_project_for_session_falls_back_to_registered_session_index():
    with patch.object(
        plugin,
        "list_projects",
        return_value=[{"name": "alpha"}, {"name": "beta"}],
    ), patch.object(
        plugin,
        "list_sessions",
        side_effect=lambda project: (
            [{"session_id": "target-session"}] if project == "beta" else []
        ),
    ):
        project, error = plugin._project_for_session("target-session")

    assert project == "beta"
    assert error is None


def test_project_for_session_errors_without_runtime_or_registered_project():
    with patch.object(plugin, "list_projects", return_value=[]):
        project, error = plugin._project_for_session("missing-session")

    assert project is None
    assert "Could not determine the current project" in error
