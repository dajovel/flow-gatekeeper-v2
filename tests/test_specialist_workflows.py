"""
Tests for the four specialist workflows:
  HermesWorkflow / WifiWorkflow / AudioWorkflow / ScribeWorkflow

All four share the same structure (only AGENT differs), so the full
behavioural suite runs against HermesWorkflow. The other three are covered
by a parametrised smoke-test that confirms:
  - the workflow completes
  - the correct AGENT constant is passed to specialist_activity

Coverage areas
──────────────
1. Complete immediately:    first specialist_activity call returns "complete"
2. Blocker → signal:        first call returns "blocker", user_answer resumes it
3. Multiple blockers:       two consecutive blockers accumulate context correctly
4. Context accumulation:    answer keys are answer_1, answer_2, … sequentially
5. get_status() query:      stage transitions "working" → "blocked" → "working" → "complete"
6. user_answer signal:      sets _answer_ready; wait_condition unblocks
7. Smoke tests:             HermesWorkflow, WifiWorkflow, AudioWorkflow, ScribeWorkflow all run
"""

import asyncio

import pytest
from temporalio import activity
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from activities.specialist_activity import SpecialistInput
from workflows.audio_workflow import AudioWorkflow
from workflows.hermes_workflow import HermesWorkflow
from workflows.scribe_workflow import ScribeWorkflow
from workflows.wifi_workflow import WifiWorkflow

TASK_QUEUE = "test-spec-queue"

# Map agent name → workflow class for parametrised tests
_WORKFLOW_MAP = {
    "hermes": HermesWorkflow,
    "wifi":   WifiWorkflow,
    "audio":  AudioWorkflow,
    "scribe": ScribeWorkflow,
}


async def _poll_specialist_status(handle, workflow_cls, target_stage: str, timeout: float = 10.0) -> dict:
    """Poll the specialist workflow's get_status() until *target_stage* is reached."""
    deadline = asyncio.get_event_loop().time() + timeout
    while True:
        status = await handle.query(workflow_cls.get_status)
        if status["stage"] == target_stage:
            return status
        if asyncio.get_event_loop().time() > deadline:
            raise TimeoutError(
                f"Workflow did not reach stage '{target_stage}' within {timeout}s; "
                f"last stage: {status['stage']}"
            )
        await asyncio.sleep(0.05)


# ---------------------------------------------------------------------------
# 1. Complete on first activity call
# ---------------------------------------------------------------------------

class TestCompleteImmediately:
    async def test_hermes_completes_immediately(self):
        @activity.defn(name="specialist_activity")
        async def mock_spec(inp: SpecialistInput) -> dict:
            return {"complete": True, "summary": "Antenna replacement confirmed."}

        async with await WorkflowEnvironment.start_time_skipping() as env:
            async with Worker(
                env.client,
                task_queue=TASK_QUEUE,
                workflows=[HermesWorkflow],
                activities=[mock_spec],
            ):
                result = await env.client.execute_workflow(
                    HermesWorkflow.run,
                    "LTE signal drops during handoff",
                    id="test-hermes-complete",
                    task_queue=TASK_QUEUE,
                )

        assert result == "Antenna replacement confirmed."

    async def test_default_summary_when_key_missing(self):
        """If specialist_activity returns {} (no summary key), default text is used."""

        @activity.defn(name="specialist_activity")
        async def mock_spec(inp: SpecialistInput) -> dict:
            return {"complete": True}  # summary key missing

        async with await WorkflowEnvironment.start_time_skipping() as env:
            async with Worker(
                env.client,
                task_queue=TASK_QUEUE,
                workflows=[HermesWorkflow],
                activities=[mock_spec],
            ):
                result = await env.client.execute_workflow(
                    HermesWorkflow.run,
                    "LTE issue",
                    id="test-hermes-no-summary",
                    task_queue=TASK_QUEUE,
                )

        assert result == "Task completed."


# ---------------------------------------------------------------------------
# 2. Blocker → user_answer signal → complete
# ---------------------------------------------------------------------------

class TestBlockerSignalFlow:
    async def test_single_blocker_then_complete(self):
        call_count = 0

        @activity.defn(name="specialist_activity")
        async def mock_spec(inp: SpecialistInput) -> dict:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"blocker": True, "question": "What firmware version?"}
            return {"complete": True, "summary": "Firmware v3.2 confirmed — update needed."}

        async with await WorkflowEnvironment.start_time_skipping() as env:
            async with Worker(
                env.client,
                task_queue=TASK_QUEUE,
                workflows=[HermesWorkflow],
                activities=[mock_spec],
            ):
                handle = await env.client.start_workflow(
                    HermesWorkflow.run,
                    "LTE drops",
                    id="test-hermes-blocker",
                    task_queue=TASK_QUEUE,
                )

                # Wait for the workflow to enter "blocked" state
                status = await _poll_specialist_status(handle, HermesWorkflow, "blocked")
                assert status["pending_question"] == "What firmware version?"

                # Send the answer via signal
                await handle.signal(HermesWorkflow.user_answer, "v3.1")

                result = await handle.result()

        assert result == "Firmware v3.2 confirmed — update needed."
        assert call_count == 2

    async def test_pending_question_exposed_via_get_status(self):
        """pending_question should be visible in get_status() while blocked."""

        @activity.defn(name="specialist_activity")
        async def mock_spec(inp: SpecialistInput) -> dict:
            # Return blocker only once
            if not inp.context:
                return {"blocker": True, "question": "Which LTE band?"}
            return {"complete": True, "summary": "Done."}

        async with await WorkflowEnvironment.start_time_skipping() as env:
            async with Worker(
                env.client,
                task_queue=TASK_QUEUE,
                workflows=[HermesWorkflow],
                activities=[mock_spec],
            ):
                handle = await env.client.start_workflow(
                    HermesWorkflow.run,
                    "LTE drops",
                    id="test-hermes-pending-q",
                    task_queue=TASK_QUEUE,
                )

                status = await _poll_specialist_status(handle, HermesWorkflow, "blocked")
                assert status["pending_question"] == "Which LTE band?"
                assert status["stage"] == "blocked"

                await handle.signal(HermesWorkflow.user_answer, "Band 3")
                await handle.result()


# ---------------------------------------------------------------------------
# 3. Multiple blockers in sequence
# ---------------------------------------------------------------------------

class TestMultipleBlockers:
    async def test_two_blockers_context_accumulates(self):
        """Two consecutive blockers should both be answered and passed in context."""
        call_count = 0
        final_context: dict = {}

        @activity.defn(name="specialist_activity")
        async def mock_spec(inp: SpecialistInput) -> dict:
            nonlocal call_count, final_context
            call_count += 1
            if call_count == 1:
                return {"blocker": True, "question": "Firmware version?"}
            if call_count == 2:
                return {"blocker": True, "question": "Device model?"}
            # Third call: record the full context and complete
            final_context = dict(inp.context)
            return {"complete": True, "summary": "Full analysis done."}

        async with await WorkflowEnvironment.start_time_skipping() as env:
            async with Worker(
                env.client,
                task_queue=TASK_QUEUE,
                workflows=[HermesWorkflow],
                activities=[mock_spec],
            ):
                handle = await env.client.start_workflow(
                    HermesWorkflow.run,
                    "LTE drops",
                    id="test-hermes-multi-blocker",
                    task_queue=TASK_QUEUE,
                )

                # First blocker
                await _poll_specialist_status(handle, HermesWorkflow, "blocked")
                await handle.signal(HermesWorkflow.user_answer, "v3.1")

                # Second blocker
                await _poll_specialist_status(handle, HermesWorkflow, "blocked")
                await handle.signal(HermesWorkflow.user_answer, "XYZ-Pro")

                result = await handle.result()

        assert result == "Full analysis done."
        assert call_count == 3
        # Both answers should be in the context passed to the third call
        assert "answer_1" in final_context
        assert "answer_2" in final_context
        assert final_context["answer_1"] == "v3.1"
        assert final_context["answer_2"] == "XYZ-Pro"


# ---------------------------------------------------------------------------
# 4. Context key naming
# ---------------------------------------------------------------------------

class TestContextKeyNaming:
    async def test_context_keys_are_sequential_answer_n(self):
        """Keys must be answer_1, answer_2, answer_3 — not answer_0 or other schemes."""
        call_count = 0
        received_contexts: list[dict] = []

        @activity.defn(name="specialist_activity")
        async def mock_spec(inp: SpecialistInput) -> dict:
            nonlocal call_count
            call_count += 1
            received_contexts.append(dict(inp.context))
            if call_count <= 2:
                return {"blocker": True, "question": f"Question {call_count}"}
            return {"complete": True, "summary": "Done"}

        async with await WorkflowEnvironment.start_time_skipping() as env:
            async with Worker(
                env.client,
                task_queue=TASK_QUEUE,
                workflows=[HermesWorkflow],
                activities=[mock_spec],
            ):
                handle = await env.client.start_workflow(
                    HermesWorkflow.run,
                    "LTE issue",
                    id="test-hermes-context-keys",
                    task_queue=TASK_QUEUE,
                )

                await _poll_specialist_status(handle, HermesWorkflow, "blocked")
                await handle.signal(HermesWorkflow.user_answer, "answer-one")

                await _poll_specialist_status(handle, HermesWorkflow, "blocked")
                await handle.signal(HermesWorkflow.user_answer, "answer-two")

                await handle.result()

        # Call 1: empty context
        assert received_contexts[0] == {}
        # Call 2: one answer
        assert set(received_contexts[1].keys()) == {"answer_1"}
        # Call 3: two answers
        assert set(received_contexts[2].keys()) == {"answer_1", "answer_2"}


# ---------------------------------------------------------------------------
# 5. Stage transitions visible via get_status()
# ---------------------------------------------------------------------------

class TestStageTransitions:
    async def test_stage_moves_from_working_to_blocked_to_complete(self):
        stages_observed: list[str] = []

        @activity.defn(name="specialist_activity")
        async def mock_spec(inp: SpecialistInput) -> dict:
            stages_observed.append("activity_called")
            if not inp.context:
                return {"blocker": True, "question": "Info?"}
            return {"complete": True, "summary": "Done"}

        async with await WorkflowEnvironment.start_time_skipping() as env:
            async with Worker(
                env.client,
                task_queue=TASK_QUEUE,
                workflows=[HermesWorkflow],
                activities=[mock_spec],
            ):
                handle = await env.client.start_workflow(
                    HermesWorkflow.run,
                    "LTE issue",
                    id="test-hermes-stages",
                    task_queue=TASK_QUEUE,
                )

                # Must reach "blocked"
                blocked_status = await _poll_specialist_status(handle, HermesWorkflow, "blocked")
                assert blocked_status["stage"] == "blocked"

                await handle.signal(HermesWorkflow.user_answer, "yes")

                await handle.result()

                final_status = await handle.query(HermesWorkflow.get_status)
                assert final_status["stage"] == "complete"


# ---------------------------------------------------------------------------
# 6. user_answer signal clears state correctly
# ---------------------------------------------------------------------------

class TestUserAnswerSignal:
    async def test_answer_ready_flag_cleared_between_blockers(self):
        """
        After the first blocker is answered, _answer_ready must be reset to False
        so the second blocker can also pause properly.
        """
        call_count = 0

        @activity.defn(name="specialist_activity")
        async def mock_spec(inp: SpecialistInput) -> dict:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return {"blocker": True, "question": f"Q{call_count}"}
            return {"complete": True, "summary": "Both questions answered."}

        async with await WorkflowEnvironment.start_time_skipping() as env:
            async with Worker(
                env.client,
                task_queue=TASK_QUEUE,
                workflows=[HermesWorkflow],
                activities=[mock_spec],
            ):
                handle = await env.client.start_workflow(
                    HermesWorkflow.run,
                    "LTE issue",
                    id="test-hermes-signal-cleared",
                    task_queue=TASK_QUEUE,
                )

                # Both blockers must each independently pause and receive a signal
                await _poll_specialist_status(handle, HermesWorkflow, "blocked")
                await handle.signal(HermesWorkflow.user_answer, "a1")

                await _poll_specialist_status(handle, HermesWorkflow, "blocked")
                await handle.signal(HermesWorkflow.user_answer, "a2")

                result = await handle.result()

        assert result == "Both questions answered."


# ---------------------------------------------------------------------------
# 7. Smoke tests — all four specialist workflows
# ---------------------------------------------------------------------------

class TestAllSpecialistsSmoke:
    @pytest.mark.parametrize("agent,workflow_cls", list(_WORKFLOW_MAP.items()))
    async def test_specialist_completes(self, agent: str, workflow_cls):
        """Each specialist workflow should complete and return the summary."""
        received_agents: list[str] = []

        @activity.defn(name="specialist_activity")
        async def mock_spec(inp: SpecialistInput) -> dict:
            received_agents.append(inp.agent)
            return {"complete": True, "summary": f"{agent} done"}

        async with await WorkflowEnvironment.start_time_skipping() as env:
            async with Worker(
                env.client,
                task_queue=TASK_QUEUE,
                workflows=[workflow_cls],
                activities=[mock_spec],
            ):
                result = await env.client.execute_workflow(
                    workflow_cls.run,
                    f"task for {agent}",
                    id=f"test-smoke-{agent}",
                    task_queue=TASK_QUEUE,
                )

        assert result == f"{agent} done"
        assert received_agents == [agent], (
            f"Expected specialist_activity to receive agent='{agent}' "
            f"but got {received_agents}"
        )
