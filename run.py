"""
Terminal client for flow-gatekeeper-v2.

Starts a GatekeeperWorkflow, handles question/answer exchanges in the
terminal, then waits for the specialist child workflow to complete.

Usage:
    python run.py "describe your task here"
    python run.py "LTE signal drops on device XYZ during tower handoff"

Watch execution in Temporal UI: http://localhost:8233
"""

import asyncio
import os
import sys
import uuid
from datetime import timedelta

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter

from workflows.gatekeeper_workflow import GatekeeperWorkflow, GatekeeperInput
from workflows.hermes_workflow import HermesWorkflow
from workflows.wifi_workflow import WifiWorkflow
from workflows.audio_workflow import AudioWorkflow
from workflows.scribe_workflow import ScribeWorkflow

# Maps the LLM's chosen_agent string to the right workflow class.
# Used to query/signal the correct specialist child workflow.
_WORKFLOW_MAP = {
    "hermes": HermesWorkflow,
    "wifi":   WifiWorkflow,
    "audio":  AudioWorkflow,
    "scribe": ScribeWorkflow,
}

TASK_QUEUE = "gatekeeper-queue"
POLL_INTERVAL = 0.5  # seconds


async def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python run.py \"describe your task here\"")
        sys.exit(1)

    task = " ".join(sys.argv[1:])

    client = await Client.connect(
        os.getenv("TEMPORAL_ADDRESS", "localhost:7233"),
        data_converter=pydantic_data_converter,
    )

    workflow_id = f"gatekeeper-{uuid.uuid4().hex[:8]}"

    print(f"\n{'='*60}")
    print(f"[Apollo] Task:        {task}")
    print(f"[Apollo] Workflow ID: {workflow_id}")
    print(f"[Apollo] Temporal UI: http://localhost:8233")
    print(f"{'='*60}\n")
    print("[Apollo] Gatekeeper analyzing...\n")

    handle = await client.start_workflow(
        GatekeeperWorkflow.run,
        GatekeeperInput(task=task),
        id=workflow_id,
        task_queue=TASK_QUEUE,
        execution_timeout=timedelta(hours=2),
    )

    child_handle = None
    last_stage = ""
    last_child_question = ""
    status = {}
    poll_count = 0

    while True:
        # ── Poll gatekeeper ──────────────────────────────────────────────────
        try:
            status = await handle.query(GatekeeperWorkflow.get_status)
        except Exception:
            break  # workflow finished or errored

        stage = status["stage"]
        poll_count += 1

        # Print a dot every ~5 seconds while the LLM is thinking so the
        # terminal doesn't look frozen.
        if stage == "analyzing" and poll_count % 10 == 0:
            print(".", end="", flush=True)

        if stage != last_stage:
            last_stage = stage
            if stage == "analyzing":
                print("[Apollo] Analyzing...")
            elif stage == "handing_off":
                agent = (status.get("chosen_agent") or "unknown").upper()
                conf  = status.get("confidence", 0)
                print(f"\n[Apollo] Confident ({conf:.0%}) — routing to {agent}")
                child_id = status.get("child_workflow_id", "")
                if child_id:
                    child_handle = client.get_workflow_handle(child_id)
                    print(f"[Apollo] Child workflow: {child_id}\n")
            elif stage == "complete":
                break

        # ── Gatekeeper question ──────────────────────────────────────────────
        # When the LLM returned action="ask", the workflow is paused here
        # waiting for a user_answer signal.
        if stage == "waiting_for_answer":
            q = status.get("pending_question", "")
            if q:
                print(f"[Apollo] Question: {q}")
                answer = input("Your answer: ").strip()
                print()
                await handle.signal(GatekeeperWorkflow.user_answer, answer)
                last_stage = ""  # force re-print of next stage transition

        # ── Specialist blocker ───────────────────────────────────────────────
        # Once the child workflow is running, poll it for blockers.
        # We look up the right workflow class from _WORKFLOW_MAP so the
        # query/signal use the correct method references.
        if child_handle:
            chosen = status.get("chosen_agent", "")
            specialist_cls = _WORKFLOW_MAP.get(chosen)
            if specialist_cls:
                try:
                    cs = await child_handle.query(specialist_cls.get_status)
                    cq = cs.get("pending_question", "")
                    if cs.get("stage") == "blocked" and cq and cq != last_child_question:
                        last_child_question = cq
                        agent_label = (chosen or "agent").upper()
                        print(f"[{agent_label}] Needs info: {cq}")
                        answer = input("Your answer: ").strip()
                        print()
                        await child_handle.signal(specialist_cls.user_answer, answer)
                    elif cs.get("stage") == "complete":
                        child_handle = None
                except Exception:
                    child_handle = None

        await asyncio.sleep(POLL_INTERVAL)

    # ── Final result ─────────────────────────────────────────────────────────
    try:
        result = await handle.result()
        agent = (status.get("chosen_agent") or "specialist").upper()
        print(f"\n{'='*60}")
        print(f"[{agent}] Result")
        print(f"{'='*60}")
        print(f"\n{result}\n")
        print(f"{'='*60}\n")
    except Exception as e:
        print(f"\n[Apollo] Ended: {e}\n")


if __name__ == "__main__":
    asyncio.run(main())
