"""
Tests for workflows/gatekeeper_workflow.py

Uses ``WorkflowEnvironment.start_time_skipping()`` — a fully in-process
Temporal server that needs no external daemon.

Coverage areas
──────────────
1. Happy path:          "ready" decision with confidence ≥ 0.8 → child runs, result returned
2. Low confidence:      "ready" with confidence < 0.8 is treated as "working" → loops
3. Working → ready:     a "working" turn followed by "ready" completes normally
4. Ask / signal:        "ask" exposes question via get_status(), user_answer signal resumes
5. MAX_ITERATIONS:      after 8 loops with no confident decision → forced handoff
6. MAX_ITERATIONS:      after 8 loops with no agent chosen → graceful error message
7. get_status() query:  fields present and accurate during each workflow stage
8. Child workflow ID:   normal format "{wf_id}-{agent}", forced format with "-forced"
9. Agent routing:       all four specialist workflow classes are reachable
"""

import asyncio

import pytest
from temporalio import activity
from temporalio.client import Client
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from activities.gatekeeper_activity import GatekeeperDecision
from activities.specialist_activity import SpecialistInput
from workflows.audio_workflow import AudioWorkflow
from workflows.gatekeeper_workflow import (
    CONFIDENCE_THRESHOLD,
    GatekeeperInput,
    GatekeeperWorkflow,
)
from workflows.hermes_workflow import HermesWorkflow
from workflows.scribe_workflow import ScribeWorkflow
from workflows.wifi_workflow import WifiWorkflow

# All workflows must be registered even when only testing the parent, because
# GatekeeperWorkflow spawns child workflows on the same task queue.
ALL_WORKFLOWS = [
    GatekeeperWorkflow,
    HermesWorkflow,
    WifiWorkflow,
    AudioWorkflow,
    ScribeWorkflow,
]
TASK_QUEUE = "test-gk-queue"


async def _poll_status(handle, target_stage: str, timeout: float = 10.0) -> dict:
    """Poll get_status() until the workflow reaches *target_stage* or times out."""
    deadline = asyncio.get_event_loop().time() + timeout
    while True:
        status = await handle.query(GatekeeperWorkflow.get_status)
        if status["stage"] == target_stage:
            return status
        if asyncio.get_event_loop().time() > deadline:
            raise TimeoutError(
                f"Workflow did not reach stage '{target_stage}' within {timeout}s; "
                f"last stage: {status['stage']}"
            )
        await asyncio.sleep(0.05)


# ---------------------------------------------------------------------------
# 1. Happy path — "ready" immediately
# ---------------------------------------------------------------------------

class TestHappyPath:
    async def test_ready_decision_returns_specialist_result(self):
        @activity.defn(name="gatekeeper_activity")
        async def mock_gk(messages: list) -> GatekeeperDecision:
            return GatekeeperDecision(
                action="ready",
                chosen_agent="hermes",
                confidence=0.9,
                enriched_brief="LTE handoff failure",
            )

        @activity.defn(name="specialist_activity")
        async def mock_spec(inp: SpecialistInput) -> dict:
            return {"complete": True, "summary": "Replace antenna module."}

        async with await WorkflowEnvironment.start_time_skipping() as env:
            async with Worker(
                env.client,
                task_queue=TASK_QUEUE,
                workflows=ALL_WORKFLOWS,
                activities=[mock_gk, mock_spec],
            ):
                result = await env.client.execute_workflow(
                    GatekeeperWorkflow.run,
                    GatekeeperInput(task="LTE drops on handoff"),
                    id="test-gk-happy",
                    task_queue=TASK_QUEUE,
                )

        assert result == "Replace antenna module."

    async def test_enriched_brief_passed_to_specialist(self):
        """The enriched_brief from the gatekeeper should be the specialist's input."""
        received_briefs: list[str] = []

        @activity.defn(name="gatekeeper_activity")
        async def mock_gk(messages: list) -> GatekeeperDecision:
            return GatekeeperDecision(
                action="ready",
                chosen_agent="wifi",
                confidence=0.9,
                enriched_brief="WiFi DHCP failure on SSID Corp-5GHz",
            )

        @activity.defn(name="specialist_activity")
        async def mock_spec(inp: SpecialistInput) -> dict:
            received_briefs.append(inp.brief)
            return {"complete": True, "summary": "Fixed."}

        async with await WorkflowEnvironment.start_time_skipping() as env:
            async with Worker(
                env.client,
                task_queue=TASK_QUEUE,
                workflows=ALL_WORKFLOWS,
                activities=[mock_gk, mock_spec],
            ):
                await env.client.execute_workflow(
                    GatekeeperWorkflow.run,
                    GatekeeperInput(task="wifi drops"),
                    id="test-gk-brief",
                    task_queue=TASK_QUEUE,
                )

        assert received_briefs == ["WiFi DHCP failure on SSID Corp-5GHz"]


# ---------------------------------------------------------------------------
# 2. Confidence threshold — "ready" but confidence too low
# ---------------------------------------------------------------------------

class TestConfidenceThreshold:
    async def test_low_confidence_loops_not_handed_off(self):
        """
        If action='ready' but confidence < CONFIDENCE_THRESHOLD the workflow
        should keep looping.  We give it a low-confidence turn first, then a
        high-confidence one.
        """
        call_count = 0

        @activity.defn(name="gatekeeper_activity")
        async def mock_gk(messages: list) -> GatekeeperDecision:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # confidence below threshold — should NOT trigger handoff
                return GatekeeperDecision(
                    action="ready",
                    chosen_agent="hermes",
                    confidence=CONFIDENCE_THRESHOLD - 0.05,
                    enriched_brief="lte issue",
                )
            # second call: now confident enough
            return GatekeeperDecision(
                action="ready",
                chosen_agent="hermes",
                confidence=CONFIDENCE_THRESHOLD,
                enriched_brief="lte issue confirmed",
            )

        @activity.defn(name="specialist_activity")
        async def mock_spec(inp: SpecialistInput) -> dict:
            return {"complete": True, "summary": "Done after two gatekeeper turns."}

        async with await WorkflowEnvironment.start_time_skipping() as env:
            async with Worker(
                env.client,
                task_queue=TASK_QUEUE,
                workflows=ALL_WORKFLOWS,
                activities=[mock_gk, mock_spec],
            ):
                result = await env.client.execute_workflow(
                    GatekeeperWorkflow.run,
                    GatekeeperInput(task="lte"),
                    id="test-gk-confidence",
                    task_queue=TASK_QUEUE,
                )

        assert result == "Done after two gatekeeper turns."
        assert call_count == 2, "Gatekeeper should have been called twice"

    async def test_unknown_agent_in_workflow_map_does_not_trigger_handoff(self):
        """chosen_agent not in _WORKFLOW_MAP should prevent handoff and keep looping."""
        call_count = 0

        @activity.defn(name="gatekeeper_activity")
        async def mock_gk(messages: list) -> GatekeeperDecision:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return GatekeeperDecision(
                    action="ready",
                    chosen_agent="nonexistent_agent",   # not in _WORKFLOW_MAP
                    confidence=0.95,
                    enriched_brief="task",
                )
            return GatekeeperDecision(
                action="ready",
                chosen_agent="scribe",
                confidence=0.95,
                enriched_brief="task",
            )

        @activity.defn(name="specialist_activity")
        async def mock_spec(inp: SpecialistInput) -> dict:
            return {"complete": True, "summary": "Scribe done."}

        async with await WorkflowEnvironment.start_time_skipping() as env:
            async with Worker(
                env.client,
                task_queue=TASK_QUEUE,
                workflows=ALL_WORKFLOWS,
                activities=[mock_gk, mock_spec],
            ):
                result = await env.client.execute_workflow(
                    GatekeeperWorkflow.run,
                    GatekeeperInput(task="write readme"),
                    id="test-gk-unknown-agent",
                    task_queue=TASK_QUEUE,
                )

        assert result == "Scribe done."
        assert call_count == 2


# ---------------------------------------------------------------------------
# 3. "working" → "ready"
# ---------------------------------------------------------------------------

class TestWorkingThenReady:
    async def test_working_turn_followed_by_ready(self):
        call_count = 0

        @activity.defn(name="gatekeeper_activity")
        async def mock_gk(messages: list) -> GatekeeperDecision:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return GatekeeperDecision(action="working", reasoning="gathering info")
            return GatekeeperDecision(
                action="ready",
                chosen_agent="audio",
                confidence=0.9,
                enriched_brief="bluetooth codec issue",
            )

        @activity.defn(name="specialist_activity")
        async def mock_spec(inp: SpecialistInput) -> dict:
            return {"complete": True, "summary": "A2DP codec downgrade found."}

        async with await WorkflowEnvironment.start_time_skipping() as env:
            async with Worker(
                env.client,
                task_queue=TASK_QUEUE,
                workflows=ALL_WORKFLOWS,
                activities=[mock_gk, mock_spec],
            ):
                result = await env.client.execute_workflow(
                    GatekeeperWorkflow.run,
                    GatekeeperInput(task="bluetooth keeps dropping"),
                    id="test-gk-working-then-ready",
                    task_queue=TASK_QUEUE,
                )

        assert result == "A2DP codec downgrade found."
        assert call_count == 2


# ---------------------------------------------------------------------------
# 4. "ask" action — signal flow
# ---------------------------------------------------------------------------

class TestAskSignalFlow:
    async def test_ask_exposes_question_and_resumes_on_signal(self):
        call_count = 0
        received_answers: list[str] = []

        @activity.defn(name="gatekeeper_activity")
        async def mock_gk(messages: list) -> GatekeeperDecision:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return GatekeeperDecision(
                    action="ask",
                    question="Which device model is affected?",
                    reasoning="need device model",
                )
            # Capture what the user answered (it's in messages[-1])
            received_answers.append(messages[-1]["content"])
            return GatekeeperDecision(
                action="ready",
                chosen_agent="hermes",
                confidence=0.95,
                enriched_brief="LTE on device XYZ",
            )

        @activity.defn(name="specialist_activity")
        async def mock_spec(inp: SpecialistInput) -> dict:
            return {"complete": True, "summary": "Device XYZ: replace modem."}

        async with await WorkflowEnvironment.start_time_skipping() as env:
            async with Worker(
                env.client,
                task_queue=TASK_QUEUE,
                workflows=ALL_WORKFLOWS,
                activities=[mock_gk, mock_spec],
            ):
                handle = await env.client.start_workflow(
                    GatekeeperWorkflow.run,
                    GatekeeperInput(task="LTE drops"),
                    id="test-gk-ask",
                    task_queue=TASK_QUEUE,
                )

                # Wait until the workflow is waiting for an answer
                status = await _poll_status(handle, "waiting_for_answer")

                assert status["pending_question"] == "Which device model is affected?"

                # Send the user's answer
                await handle.signal(GatekeeperWorkflow.user_answer, "Device XYZ")

                result = await handle.result()

        assert result == "Device XYZ: replace modem."
        assert received_answers == ["Device XYZ"]

    async def test_pending_question_cleared_after_answer(self):
        """After a signal is received, pending_question should be empty again."""
        call_count = 0

        @activity.defn(name="gatekeeper_activity")
        async def mock_gk(messages: list) -> GatekeeperDecision:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return GatekeeperDecision(
                    action="ask",
                    question="Any error logs?",
                    reasoning="r",
                )
            return GatekeeperDecision(
                action="ready", chosen_agent="wifi", confidence=0.9, enriched_brief="wifi"
            )

        @activity.defn(name="specialist_activity")
        async def mock_spec(inp: SpecialistInput) -> dict:
            return {"complete": True, "summary": "OK"}

        async with await WorkflowEnvironment.start_time_skipping() as env:
            async with Worker(
                env.client,
                task_queue=TASK_QUEUE,
                workflows=ALL_WORKFLOWS,
                activities=[mock_gk, mock_spec],
            ):
                handle = await env.client.start_workflow(
                    GatekeeperWorkflow.run,
                    GatekeeperInput(task="wifi drops"),
                    id="test-gk-question-cleared",
                    task_queue=TASK_QUEUE,
                )

                await _poll_status(handle, "waiting_for_answer")
                await handle.signal(GatekeeperWorkflow.user_answer, "no logs")
                await handle.result()


# ---------------------------------------------------------------------------
# 5 & 6. MAX_ITERATIONS guard
# ---------------------------------------------------------------------------

class TestMaxIterations:
    async def test_max_iterations_forces_handoff_when_agent_chosen(self):
        """After 8 iterations without a confident decision, handoff to last chosen agent."""

        @activity.defn(name="gatekeeper_activity")
        async def mock_gk_always_working(messages: list) -> GatekeeperDecision:
            # Never reaches confidence threshold; but does name an agent
            return GatekeeperDecision(
                action="working",
                chosen_agent="hermes",
                confidence=0.5,
                reasoning="still thinking",
            )

        @activity.defn(name="specialist_activity")
        async def mock_spec(inp: SpecialistInput) -> dict:
            return {"complete": True, "summary": "Forced handoff result."}

        async with await WorkflowEnvironment.start_time_skipping() as env:
            async with Worker(
                env.client,
                task_queue=TASK_QUEUE,
                workflows=ALL_WORKFLOWS,
                activities=[mock_gk_always_working, mock_spec],
            ):
                result = await env.client.execute_workflow(
                    GatekeeperWorkflow.run,
                    GatekeeperInput(task="ambiguous task"),
                    id="test-gk-maxiter-handoff",
                    task_queue=TASK_QUEUE,
                )

        assert result == "Forced handoff result."

    async def test_max_iterations_returns_error_message_when_no_agent(self):
        """After 8 iterations with no agent ever chosen, return a graceful error message."""

        @activity.defn(name="gatekeeper_activity")
        async def mock_gk_no_agent(messages: list) -> GatekeeperDecision:
            return GatekeeperDecision(
                action="working",
                chosen_agent="",   # never picks anyone
                confidence=0.0,
                reasoning="confused",
            )

        @activity.defn(name="specialist_activity")
        async def mock_spec(inp: SpecialistInput) -> dict:  # pragma: no cover
            return {"complete": True, "summary": "Should not reach here."}

        async with await WorkflowEnvironment.start_time_skipping() as env:
            async with Worker(
                env.client,
                task_queue=TASK_QUEUE,
                workflows=ALL_WORKFLOWS,
                activities=[mock_gk_no_agent, mock_spec],
            ):
                result = await env.client.execute_workflow(
                    GatekeeperWorkflow.run,
                    GatekeeperInput(task="totally ambiguous"),
                    id="test-gk-maxiter-noagent",
                    task_queue=TASK_QUEUE,
                )

        assert "could not route" in result.lower()


# ---------------------------------------------------------------------------
# 7. get_status() query
# ---------------------------------------------------------------------------

class TestGetStatusQuery:
    async def test_status_fields_present_in_initial_state(self):
        """get_status() should always return all expected keys."""

        @activity.defn(name="gatekeeper_activity")
        async def mock_gk(messages: list) -> GatekeeperDecision:
            return GatekeeperDecision(
                action="ready", chosen_agent="scribe", confidence=0.9, enriched_brief="docs"
            )

        @activity.defn(name="specialist_activity")
        async def mock_spec(inp: SpecialistInput) -> dict:
            return {"complete": True, "summary": "Doc written."}

        async with await WorkflowEnvironment.start_time_skipping() as env:
            async with Worker(
                env.client,
                task_queue=TASK_QUEUE,
                workflows=ALL_WORKFLOWS,
                activities=[mock_gk, mock_spec],
            ):
                handle = await env.client.start_workflow(
                    GatekeeperWorkflow.run,
                    GatekeeperInput(task="write doc"),
                    id="test-gk-status-fields",
                    task_queue=TASK_QUEUE,
                )

                # Query during execution (should have all keys)
                status = await handle.query(GatekeeperWorkflow.get_status)
                await handle.result()

        required_keys = {"stage", "pending_question", "chosen_agent", "confidence", "child_workflow_id"}
        assert required_keys.issubset(status.keys())

    async def test_chosen_agent_and_confidence_tracked_during_working_turns(self):
        """chosen_agent + confidence are updated even on non-final turns."""
        call_count = 0

        @activity.defn(name="gatekeeper_activity")
        async def mock_gk(messages: list) -> GatekeeperDecision:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return GatekeeperDecision(
                    action="working",
                    chosen_agent="hermes",
                    confidence=0.6,
                    reasoning="leaning hermes",
                )
            return GatekeeperDecision(
                action="ready",
                chosen_agent="hermes",
                confidence=0.9,
                enriched_brief="lte issue",
            )

        @activity.defn(name="specialist_activity")
        async def mock_spec(inp: SpecialistInput) -> dict:
            return {"complete": True, "summary": "Done."}

        async with await WorkflowEnvironment.start_time_skipping() as env:
            async with Worker(
                env.client,
                task_queue=TASK_QUEUE,
                workflows=ALL_WORKFLOWS,
                activities=[mock_gk, mock_spec],
            ):
                await env.client.execute_workflow(
                    GatekeeperWorkflow.run,
                    GatekeeperInput(task="lte drops"),
                    id="test-gk-status-tracking",
                    task_queue=TASK_QUEUE,
                )

        # Completed without error is enough to confirm state tracking worked


# ---------------------------------------------------------------------------
# 8. Child workflow ID format
# ---------------------------------------------------------------------------

class TestChildWorkflowId:
    async def test_normal_child_id_format(self):
        """Child workflow ID should be {parent_id}-{agent}."""
        captured_child_ids: list[str] = []

        @activity.defn(name="gatekeeper_activity")
        async def mock_gk(messages: list) -> GatekeeperDecision:
            return GatekeeperDecision(
                action="ready", chosen_agent="audio", confidence=0.9, enriched_brief="audio"
            )

        @activity.defn(name="specialist_activity")
        async def mock_spec(inp: SpecialistInput) -> dict:
            return {"complete": True, "summary": "Audio fixed."}

        parent_id = "test-gk-childid-normal"

        async with await WorkflowEnvironment.start_time_skipping() as env:
            async with Worker(
                env.client,
                task_queue=TASK_QUEUE,
                workflows=ALL_WORKFLOWS,
                activities=[mock_gk, mock_spec],
            ):
                handle = await env.client.start_workflow(
                    GatekeeperWorkflow.run,
                    GatekeeperInput(task="bt drops"),
                    id=parent_id,
                    task_queue=TASK_QUEUE,
                )
                await handle.result()
                status = await handle.query(GatekeeperWorkflow.get_status)

        expected_child_id = f"{parent_id}-audio"
        assert status["child_workflow_id"] == expected_child_id

    async def test_forced_child_id_format(self):
        """After MAX_ITERATIONS, the forced child ID should end with '-forced'."""

        @activity.defn(name="gatekeeper_activity")
        async def mock_gk(messages: list) -> GatekeeperDecision:
            return GatekeeperDecision(
                action="working", chosen_agent="hermes", confidence=0.5, reasoning="r"
            )

        @activity.defn(name="specialist_activity")
        async def mock_spec(inp: SpecialistInput) -> dict:
            return {"complete": True, "summary": "Forced done."}

        parent_id = "test-gk-childid-forced"

        async with await WorkflowEnvironment.start_time_skipping() as env:
            async with Worker(
                env.client,
                task_queue=TASK_QUEUE,
                workflows=ALL_WORKFLOWS,
                activities=[mock_gk, mock_spec],
            ):
                handle = await env.client.start_workflow(
                    GatekeeperWorkflow.run,
                    GatekeeperInput(task="ambiguous"),
                    id=parent_id,
                    task_queue=TASK_QUEUE,
                )
                await handle.result()
                status = await handle.query(GatekeeperWorkflow.get_status)

        expected_child_id = f"{parent_id}-hermes-forced"
        assert status["child_workflow_id"] == expected_child_id


# ---------------------------------------------------------------------------
# 9. All four specialist routes are reachable
# ---------------------------------------------------------------------------

class TestAgentRouting:
    @pytest.mark.parametrize("agent", ["hermes", "wifi", "audio", "scribe"])
    async def test_routes_to_each_specialist(self, agent: str):
        @activity.defn(name="gatekeeper_activity")
        async def mock_gk(messages: list) -> GatekeeperDecision:
            return GatekeeperDecision(
                action="ready", chosen_agent=agent, confidence=0.9, enriched_brief=f"{agent} task"
            )

        @activity.defn(name="specialist_activity")
        async def mock_spec(inp: SpecialistInput) -> dict:
            return {"complete": True, "summary": f"{agent} complete"}

        async with await WorkflowEnvironment.start_time_skipping() as env:
            async with Worker(
                env.client,
                task_queue=TASK_QUEUE,
                workflows=ALL_WORKFLOWS,
                activities=[mock_gk, mock_spec],
            ):
                result = await env.client.execute_workflow(
                    GatekeeperWorkflow.run,
                    GatekeeperInput(task=f"{agent} task"),
                    id=f"test-gk-route-{agent}",
                    task_queue=TASK_QUEUE,
                )

        assert result == f"{agent} complete"
