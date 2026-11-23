from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from agent_utils.core.events import InputReceived
from agent_utils.core.pipeline import PipelineBuilder
from agent_utils.core.session import Session
from agent_utils.core.state import MODEL_RENDER_CHANNEL
from agent_utils.features.core import (
    ExcludeForgotten,
    MessageRenderer,
    RenderPlanFinalizer,
    RenderPlanInitializer,
    RenderTransformRegistrar,
    core_feature,
)
from agent_zoo.pipelines.default import build_default_builder

import session_labeling as plugin


def _build_session(*, system_prompt: str = "Base system") -> Session:
    session = Session(system_prompt=system_prompt)
    builder = PipelineBuilder().add(core_feature())
    plugin.register_features(builder, session=session, config={})
    session.pipeline = builder.build(suppress_warnings=True)
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
    assert (
        component_types.index(ExcludeForgotten)
        < component_types.index(RenderPlanFinalizer)
        < component_types.index(MessageRenderer)
    )


def test_system_prompt_guidance_is_model_only_and_storage_stays_raw():
    session = _build_session(system_prompt="Base system")
    entry = session.state.entries[0]
    stored = [{**message} for message in entry.messages]

    rendered = session.render()

    assert rendered == [
        {
            "role": "system",
            "content": "Base system" + plugin._SYSTEM_PROMPT_BLOCK,
        }
    ]
    assert entry.render(session.state, channel=MODEL_RENDER_CHANNEL) == rendered
    assert session.state.rendered_messages == rendered
    assert entry.messages == stored
    assert entry.messages[0]["content"] == "Base system"
    assert rendered[0] is not entry.messages[0]


def test_repeated_preparation_rebuilds_without_accumulating_guidance():
    session = _build_session()
    entry = session.state.entries[0]

    first = session.render()
    second = session.render()

    assert first == second
    assert entry.render_transform_names() == (
        "session-labeling system prompt",
        "forgotten exclusion",
    )
    assert second[0]["content"].count("# Session labeling") == 1


def test_existing_session_labeling_guidance_is_not_duplicated():
    system_prompt = "Base system" + plugin._SYSTEM_PROMPT_BLOCK
    session = _build_session(system_prompt=system_prompt)

    rendered = session.render()

    assert rendered[0]["content"] == system_prompt
    assert rendered[0]["content"].count("# Session labeling") == 1


def test_guidance_only_applies_to_a_system_prompt_entry():
    session = _build_session(system_prompt="")

    session.step(InputReceived({"role": "user", "content": "hello"}))
    rendered = session.render()
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
        result = plugin.name_session.call(
            state,
            name="first_turn_label",
            description="Initial investigation",
        )

    assert result == (
        "Session name set to: first_turn_label; "
        "Session description set to: Initial investigation"
    )
    assert calls == [
        (
            ("demo", "session-1"),
            {"title": "first_turn_label", "description": "Initial investigation"},
        )
    ]
    assert state._session_title == "first_turn_label"
    assert state._session_description == "Initial investigation"


def test_name_session_tool_executes_through_pipeline():
    session = _build_session()
    session.state._session_id = "sess-42"
    session.state._project_name = "demo"
    calls = []

    with patch.object(
        plugin,
        "register_session",
        side_effect=lambda *args, **kwargs: calls.append((args, kwargs)),
    ):
        session.step(
            InputReceived(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "name_session",
                                "arguments": '{"name": "bug_triage", "description": "Triaging event loop bug"}',
                            },
                        }
                    ],
                }
            )
        )

    assert calls == [
        (
            ("demo", "sess-42"),
            {"title": "bug_triage", "description": "Triaging event loop bug"},
        )
    ]
    assert session.state._session_title == "bug_triage"
    assert session.state._session_description == "Triaging event loop bug"
    assert (
        session.state.entries[-1].messages[-1]["content"]
        == "Session name set to: bug_triage; Session description set to: Triaging event loop bug"
    )


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


def test_composes_onto_default_builder():
    session = Session()
    builder = build_default_builder(
        session,
        config={},
        include_plugins=False,
        include_llm_drivers=False,
    )
    plugin.register_features(builder, session=session, config={})
    pipeline = builder.build()
    assert any(isinstance(comp, plugin.NameSession) for comp in pipeline)
    assert any(
        isinstance(comp, plugin.SessionLabelingPromptInstructions) for comp in pipeline
    )
