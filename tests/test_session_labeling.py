from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from agent_utils import PipelineBuilder, Session, core
from agent_utils.components import (
    ExcludeForgotten,
    MessageRenderer,
    RenderPlanFinalizer,
    RenderPlanInitializer,
    RenderTransformRegistrar,
)
from agent_utils.types import MODEL_RENDER_CHANNEL

import src.session_labeling as plugin


def _build_session(*, system_prompt: str = "Base system") -> Session:
    session = Session(system_prompt=system_prompt)
    builder = PipelineBuilder(session).add(core())
    plugin.register_features(builder, session=session, config=SimpleNamespace())
    session.pipeline = builder.build()
    return session


def test_prompt_instructions_build_as_an_ordered_model_registrar():
    session = _build_session()
    component_types = [type(component) for component in session.pipeline]

    assert issubclass(
        plugin.SessionLabelingPromptInstructions,
        RenderTransformRegistrar,
    )
    assert component_types.index(RenderPlanInitializer) < component_types.index(
        plugin.SessionLabelingPromptInstructions
    )
    assert component_types.index(
        plugin.SessionLabelingPromptInstructions
    ) < component_types.index(ExcludeForgotten)
    assert component_types.index(ExcludeForgotten) < component_types.index(
        RenderPlanFinalizer
    ) < component_types.index(MessageRenderer)


def test_system_prompt_guidance_is_model_only_and_storage_stays_raw():
    session = _build_session(system_prompt="Base system")
    entry = session.state.entries[0]
    stored = [{**message} for message in entry.messages]

    rendered = session.rerender()

    assert rendered == [{
        "role": "system",
        "content": "Base system" + plugin._SYSTEM_PROMPT_BLOCK,
    }]
    assert entry.render(session.state, channel=MODEL_RENDER_CHANNEL) == rendered
    assert session.state.rendered_messages == rendered
    assert entry.messages == stored
    assert entry.messages[0]["content"] == "Base system"
    assert rendered[0] is not entry.messages[0]


def test_repeated_preparation_rebuilds_without_accumulating_guidance():
    session = _build_session()
    entry = session.state.entries[0]

    first = session.rerender()
    second = session.rerender()

    assert first == second
    assert entry.render_transform_names() == (
        "session-labeling system prompt",
        "forgotten exclusion",
    )
    assert second[0]["content"].count("# Session labeling") == 1


def test_existing_session_labeling_guidance_is_not_duplicated():
    system_prompt = "Base system" + plugin._SYSTEM_PROMPT_BLOCK
    session = _build_session(system_prompt=system_prompt)

    rendered = session.rerender()

    assert rendered[0]["content"] == system_prompt
    assert rendered[0]["content"].count("# Session labeling") == 1


def test_guidance_only_applies_to_a_system_prompt_entry():
    session = _build_session(system_prompt="")

    rendered = session.run({"role": "user", "content": "hello"})
    entry = session.state.entries[0]

    assert rendered == [{"role": "user", "content": "hello"}]
    assert entry.messages == rendered
    assert "# Session labeling" not in rendered[0]["content"]


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
