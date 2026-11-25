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
    plugin.register_features(builder, config={})
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
    renderer_index = next(
        index
        for index, component in enumerate(session.pipeline)
        if isinstance(component, MessageRenderer)
    )
    assert (
        component_types.index(ExcludeForgotten)
        < component_types.index(RenderPlanFinalizer)
        < renderer_index
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


def test_name_session_updates_session_state_and_submits_checkpoint():
    from agent_zoo.core.feature_state import ZooState, session_state
    from agent_zoo.events import SaveCheckpoint

    session = _build_session()
    session.state = ZooState()
    smeta = session_state(session.state)
    smeta.session_id = "session-1"
    smeta.instance_id = "inst-1"
    smeta.project_name = "demo"

    result = plugin.name_session.call(
        session.state,
        name="first_turn_label",
        description="Initial investigation",
    )

    assert result == (
        "Session name set to: first_turn_label; "
        "Session description set to: Initial investigation"
    )
    assert smeta.title == "first_turn_label"
    assert smeta.description == "Initial investigation"
    outbox = session.state.io.take_outbox()
    save_reqs = [eff.request for eff in outbox if isinstance(eff.request, SaveCheckpoint)]
    assert len(save_reqs) == 1
    assert save_reqs[0].session_id == "session-1"
    assert save_reqs[0].sync_catalog is True


def test_name_session_tool_executes_through_pipeline():
    from agent_zoo.core.feature_state import ZooState, session_state
    from agent_zoo.events import SaveCheckpoint

    session = _build_session()
    session.state = ZooState()
    session.ensure_initialized()
    smeta = session_state(session.state)
    smeta.session_id = "sess-42"
    smeta.project_name = "demo"

    effects = session.step(
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

    save_reqs = [
        eff.request
        for eff in effects
        if isinstance(getattr(eff, "request", None), SaveCheckpoint)
    ]
    assert len(save_reqs) == 1
    assert save_reqs[0].session_id == "sess-42"
    assert save_reqs[0].sync_catalog is True
    assert smeta.title == "bug_triage"
    assert smeta.description == "Triaging event loop bug"
    assert (
        session.state.entries[-1].messages[-1]["content"]
        == "Session name set to: bug_triage; Session description set to: Triaging event loop bug"
    )


def test_name_session_errors_without_session_id():
    from agent_zoo.core.feature_state import ZooState

    result = plugin.name_session.call(ZooState(), name="bug_triage")
    assert "Could not determine the current session id" in result


def test_composes_onto_default_builder():
    from agent_zoo.testing import assert_pipeline_sans_io

    session = Session()
    builder = build_default_builder(
        session,
        config={},
        include_plugins=False,
        include_llm_drivers=False,
    )
    plugin.register_features(builder, config={})
    pipeline = builder.build()
    assert any(isinstance(comp, plugin.NameSession) for comp in pipeline)
    assert any(
        isinstance(comp, plugin.SessionLabelingPromptInstructions) for comp in pipeline
    )
    assert_pipeline_sans_io(pipeline)
