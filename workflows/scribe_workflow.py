"""
ScribeWorkflow — documentation specialist.

Spawned as a child by GatekeeperWorkflow when the task is classified as a
documentation task (writing, reviewing, summarising technical content).

Signals:  user_answer(answer: str)
Queries:  get_status() → {stage, pending_question}
"""

from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from activities.specialist_activity import specialist_activity, SpecialistInput

AGENT = "scribe"


@workflow.defn
class ScribeWorkflow:
    """Documentation specialist. Owns its own event history in Temporal UI."""

    def __init__(self):
        self._stage: str = "working"
        self._pending_question: str = ""
        self._answer_ready: bool = False
        self._user_answer: str = ""
        self._context: dict = {}

    @workflow.signal
    async def user_answer(self, answer: str) -> None:
        self._user_answer = answer
        self._answer_ready = True

    @workflow.query
    def get_status(self) -> dict:
        return {
            "stage":            self._stage,
            "pending_question": self._pending_question,
        }

    @workflow.run
    async def run(self, brief: str) -> str:
        while True:
            result = await workflow.execute_activity(
                specialist_activity,
                SpecialistInput(agent=AGENT, brief=brief, context=self._context),
                start_to_close_timeout=timedelta(minutes=10),
                retry_policy=RetryPolicy(maximum_attempts=1),
            )

            if result.get("blocker"):
                self._stage = "blocked"
                self._pending_question = result["question"]
                self._answer_ready = False

                await workflow.wait_condition(lambda: self._answer_ready)

                self._context[f"answer_{len(self._context) + 1}"] = self._user_answer
                self._pending_question = ""
                self._stage = "working"
                self._answer_ready = False
                continue

            self._stage = "complete"
            return result.get("summary", "Task completed.")
