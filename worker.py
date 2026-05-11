"""
Worker — run this before run.py.

    python worker.py

Registers GatekeeperWorkflow plus all four specialist workflows and both
activities on a single task queue. One worker process handles everything.

── Why one task queue? ──────────────────────────────────────────────────
The parent workflow spawns child workflows on the same task queue. If you
split workflows across queues you'd need separate workers for each. One queue
keeps the dev setup simple.

── Persistence note ─────────────────────────────────────────────────────
The Temporal dev server runs in-memory by default — history is lost on
restart. To persist locally:

    temporal server start-dev --db-filename temporal.db
─────────────────────────────────────────────────────────────────────────
"""

import asyncio
import os

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.worker import Worker
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner, SandboxRestrictions

from activities.gatekeeper_activity import gatekeeper_activity
from activities.specialist_activity import specialist_activity
from workflows.gatekeeper_workflow import GatekeeperWorkflow
from workflows.hermes_workflow import HermesWorkflow
from workflows.wifi_workflow import WifiWorkflow
from workflows.audio_workflow import AudioWorkflow
from workflows.scribe_workflow import ScribeWorkflow

TASK_QUEUE = "gatekeeper-queue"


async def main() -> None:
    client = await Client.connect(
        os.getenv("TEMPORAL_ADDRESS", "localhost:7233"),
        data_converter=pydantic_data_converter,
    )

    # Tell Temporal's workflow sandbox to let pydantic-related modules through.
    # Without this, annotated_types (a pydantic dependency) gets imported inside
    # the sandbox on second runs, triggering a warning and potentially causing
    # workflow task failures on replay.
    _sandbox = SandboxedWorkflowRunner(
        restrictions=SandboxRestrictions.default.with_passthrough_modules(
            "annotated_types",
            "pydantic",
            "pydantic_core",
        )
    )

    worker = Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[
            GatekeeperWorkflow,
            HermesWorkflow,
            WifiWorkflow,
            AudioWorkflow,
            ScribeWorkflow,
        ],
        activities=[gatekeeper_activity, specialist_activity],
        workflow_runner=_sandbox,
    )
    print(f"Worker running — polling '{TASK_QUEUE}' ...")
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
