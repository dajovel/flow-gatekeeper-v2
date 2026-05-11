"""
GatekeeperWorkflow — triage parent that hands off to a specialist child workflow.

What this demonstrates:
  - Long-running loop driven by LLM decisions (all non-determinism in activities)
  - Human-in-the-loop: user answers questions via Temporal signals
  - wait_condition: durable pause — survives worker restarts
  - Child workflow handoff: specialist gets its own event history in Temporal UI

Pipeline:
  1. Receives a free-form task description (GatekeeperInput.task)
  2. Loops calling gatekeeper_activity — LLM analyses and decides:
       "working" → append reasoning, loop again
       "ask"     → expose question via query, wait for user_answer signal
       "ready"   → spawn the right specialist as a child workflow, return result
  3. Child workflow id:  "{workflow_id}-{chosen_agent}"  (visible in Temporal UI)

Signals:
  user_answer(answer: str)  — user responds to a pending gatekeeper question

Queries:
  get_status() → {stage, pending_question, chosen_agent, confidence, child_workflow_id}
"""

from dataclasses import dataclass
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from activities.gatekeeper_activity import gatekeeper_activity, GatekeeperDecision
    from workflows.hermes_workflow import HermesWorkflow
    from workflows.wifi_workflow import WifiWorkflow
    from workflows.audio_workflow import AudioWorkflow
    from workflows.scribe_workflow import ScribeWorkflow

# Hand off only when confidence reaches this threshold
CONFIDENCE_THRESHOLD = 0.8

# Maps the LLM's chosen_agent string to the right workflow class
_WORKFLOW_MAP = {
    "hermes": HermesWorkflow,
    "wifi":   WifiWorkflow,
    "audio":  AudioWorkflow,
    "scribe": ScribeWorkflow,
}


@dataclass
class GatekeeperInput:
    task: str
    task_id: str = ""


@workflow.defn
class GatekeeperWorkflow:
    """
    Parent triage workflow.
    Loops with the LLM until confident, then spawns the right specialist child.
    """

    def __init__(self):
        self._messages: list = []
        self._stage: str = "analyzing"
        self._pending_question: str = ""
        self._answer_ready: bool = False
        self._user_answer: str = ""
        self._chosen_agent: str = ""
        self._confidence: float = 0.0
        self._child_workflow_id: str = ""
        self._iterations: int = 0

    # ── Signals ──────────────────────────────────────────────────────────────
    # Signals are how the terminal client sends user answers into the workflow.
    # Temporal durably records them in the event history.

    @workflow.signal
    async def user_answer(self, answer: str) -> None:
        """User responds to a pending gatekeeper question."""
        self._user_answer = answer
        self._answer_ready = True

    # ── Queries ──────────────────────────────────────────────────────────────
    # Queries read workflow state without mutating it.
    # The terminal client polls this every 0.5 s to know what to print.

    @workflow.query
    def get_status(self) -> dict:
        return {
            "stage":              self._stage,
            "pending_question":   self._pending_question,
            "chosen_agent":       self._chosen_agent,
            "confidence":         self._confidence,
            "child_workflow_id":  self._child_workflow_id,
        }

    # ── Main loop ─────────────────────────────────────────────────────────────

    @workflow.run
    async def run(self, inp: GatekeeperInput) -> str:
        retry = RetryPolicy(maximum_attempts=3, initial_interval=timedelta(seconds=5))
        # Seed the conversation with the user's original task
        self._messages = [{"role": "user", "content": inp.task}]

        # Safety cap: if the LLM loops without deciding after this many turns,
        # force a handoff to whichever agent it last picked (or fail gracefully).
        MAX_ITERATIONS = 8

        while self._iterations < MAX_ITERATIONS:
            self._iterations += 1

            # ── Step 1: ask the LLM what to do next ──────────────────────────
            # execute_activity is the only safe way to call the LLM from a workflow.
            # Temporal will replay this from history on worker restart — it will NOT
            # re-call the LLM; it replays the stored result.
            decision: GatekeeperDecision = await workflow.execute_activity(
                gatekeeper_activity,
                args=[self._messages],
                start_to_close_timeout=timedelta(seconds=120),
                retry_policy=retry,
            )

            # Track the agent + confidence even on intermediate turns
            if decision.chosen_agent:
                self._chosen_agent = decision.chosen_agent
            self._confidence = decision.confidence

            # Append the LLM's reasoning so it remembers its own prior turns
            self._messages.append({
                "role": "assistant",
                "content": decision.reasoning or decision.enriched_brief or "(thinking)",
            })

            # ── Step 2: act on the decision ──────────────────────────────────

            if decision.action == "ask" and decision.question:
                # Expose the question via query, then durably pause until the
                # terminal client sends a user_answer signal.
                self._stage = "waiting_for_answer"
                self._pending_question = decision.question
                self._answer_ready = False

                await workflow.wait_condition(lambda: self._answer_ready)

                # Feed the answer back into the message history and loop
                self._messages.append({"role": "user", "content": self._user_answer})
                self._pending_question = ""
                self._stage = "analyzing"
                self._answer_ready = False

            elif (
                decision.action == "ready"
                and decision.confidence >= CONFIDENCE_THRESHOLD
                and decision.chosen_agent in _WORKFLOW_MAP
            ):
                # ── Hand off to specialist child workflow ─────────────────────
                # The child gets its own workflow ID and its own event history.
                # In Temporal UI you'll see both rows, linked via
                # ChildWorkflowExecutionStarted in the parent's history.
                self._stage = "handing_off"
                child_id = f"{workflow.info().workflow_id}-{decision.chosen_agent}"
                self._child_workflow_id = child_id

                specialist_cls = _WORKFLOW_MAP[decision.chosen_agent]

                result: str = await workflow.execute_child_workflow(
                    specialist_cls.run,
                    args=[decision.enriched_brief or inp.task],
                    id=child_id,
                    task_queue=workflow.info().task_queue,
                    execution_timeout=timedelta(hours=1),
                )

                self._stage = "complete"
                return result

            else:
                # "working" or confidence too low — keep gathering info
                self._stage = "analyzing"

        # MAX_ITERATIONS reached without a confident decision.
        # Force handoff to the best candidate so far, or fail with a message.
        if self._chosen_agent and self._chosen_agent in _WORKFLOW_MAP:
            self._stage = "handing_off"
            child_id = f"{workflow.info().workflow_id}-{self._chosen_agent}-forced"
            self._child_workflow_id = child_id
            specialist_cls = _WORKFLOW_MAP[self._chosen_agent]
            result = await workflow.execute_child_workflow(
                specialist_cls.run,
                args=[inp.task],
                id=child_id,
                task_queue=workflow.info().task_queue,
                execution_timeout=timedelta(hours=1),
            )
            self._stage = "complete"
            return result

        self._stage = "complete"
        return "Gatekeeper could not route this task after the maximum number of iterations."
