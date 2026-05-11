# flow-gatekeeper-v2

A durable, terminal-first AI triage system built on Temporal + Ollama.

A gatekeeper agent receives a free-form task, loops with an LLM to build
confidence, asks the user clarifying questions when needed, then hands the
task off to the right specialist agent — which runs as a separate, linked
child workflow with its own durable event history.

---

## Prerequisites

Install these before anything else.

### 1. Temporal CLI

```bash
# macOS
brew install temporal

# Linux (amd64)
curl -sSf https://temporal.download/cli/archive/latest?platform=linux&arch=amd64 \
  -o temporal.tar.gz && tar -xzf temporal.tar.gz && mv temporal /usr/local/bin/
```

Verify: `temporal --version`

### 2. Ollama + model

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Pull the model used by this project
ollama pull qwen3-coder:latest

# Verify it responds
ollama run qwen3-coder:latest "say hi"
```

### 3. Python 3.11+ and uv

```bash
# macOS
brew install python@3.11 uv

# Linux
pip install uv
```

---

## Quick Start (3 terminals)

```bash
# ── Terminal 1: Temporal dev server ──────────────────────────────────
temporal server start-dev
# UI available at http://localhost:8233
# Add --db-filename temporal.db to persist history across restarts

# ── Terminal 2: Worker (keep this running) ───────────────────────────
cd /path/to/flow-gatekeeper-v2
uv venv && uv pip install -r requirements.txt
uv run python worker.py

# ── Terminal 3: Send a task ───────────────────────────────────────────
uv run python run.py "LTE signal drops on device XYZ during tower handoff"
uv run python run.py "bluetooth keeps disconnecting from my headphones"
uv run python run.py "write a README for this project"
```

Multiple tasks can run simultaneously — open as many Terminal 3 sessions
as you want. Each gets its own unique workflow ID and isolated history.

---

## How It Works

### High-level flow

```
Terminal 3 (run.py)
    │
    │  start_workflow(GatekeeperInput)
    ▼
┌─────────────────────────────────────────────────────────┐
│  GatekeeperWorkflow  (parent)                           │
│                                                         │
│  loop (max 8 iterations):                               │
│    ┌──────────────────────────────────────────────┐     │
│    │  gatekeeper_activity  ← calls Ollama LLM     │     │
│    │  returns GatekeeperDecision:                 │     │
│    │    action = "working"  → append, loop again  │     │
│    │    action = "ask"      → pause, wait signal  │     │
│    │    action = "ready"    → spawn child workflow │     │
│    └──────────────────────────────────────────────┘     │
│                          │                              │
│         confidence >= 0.8 and chosen_agent set          │
│                          │                              │
│              execute_child_workflow()                   │
└─────────────────────────────────────────────────────────┘
                           │
         ┌─────────────────┼─────────────────────┐
         ▼                 ▼                     ▼
  HermesWorkflow    WifiWorkflow          AudioWorkflow
  (cellular)        (wifi)                (audio/BT)
         \                                      /
          └──────────── ScribeWorkflow ─────────┘
                        (documentation)

Each specialist:
  loop:
    specialist_activity  ← calls Ollama with loaded skills/*.md
      → complete  → return summary string
      → blocker   → pause, wait signal, loop again
```

### Temporal concepts in this project

| Concept | Where it is used |
|---|---|
| `@workflow.defn` | `GatekeeperWorkflow`, `HermesWorkflow`, etc. — durable state machines |
| `@activity.defn` | `gatekeeper_activity`, `specialist_activity` — all LLM calls |
| `@workflow.signal` | `user_answer(answer)` — how terminal sends human input into the workflow |
| `@workflow.query` | `get_status()` — how terminal reads workflow state without mutating it |
| `wait_condition` | Durable pause — workflow sleeps until signal arrives, survives restarts |
| `execute_child_workflow` | Spawns specialist as a child — own ID, own history, own retry policy |
| `execute_activity` | Only safe way to call the LLM from workflow code (non-determinism boundary) |
| `pydantic_data_converter` | Type-safe serialisation of dataclasses into Temporal history |
| `SandboxedWorkflowRunner` | Lets pydantic modules pass through Temporal's determinism sandbox |

### Why everything non-deterministic goes in activities

Temporal achieves durability by replaying workflow code from its event
history. If you call the LLM directly inside a workflow, every replay
would hit the network again and potentially return different results —
Temporal would detect the mismatch and throw a non-determinism error.

Activities are the escape hatch: they run once, their result is stored
in the event history, and on replay Temporal returns the stored result
without re-running the activity.

```
Workflow code (deterministic)        Activity (non-deterministic OK)
─────────────────────────────        ────────────────────────────────
Pure Python logic only               LLM calls (Ollama / OpenAI)
Signal/query handlers                File I/O
wait_condition                       Network requests
execute_child_workflow               Database reads
```

### Signal / query flow (how the terminal talks to a running workflow)

```
run.py poll loop (every 0.5 s)
  │
  ├─ query GatekeeperWorkflow.get_status()
  │     returns: {stage, pending_question, chosen_agent, confidence, child_workflow_id}
  │
  ├─ if stage == "waiting_for_answer":
  │     print question → read input → signal GatekeeperWorkflow.user_answer(text)
  │
  └─ if child_handle exists:
        query specialist_cls.get_status()
        if stage == "blocked":
            print question → read input → signal specialist_cls.user_answer(text)
```

Signals are durable — if the worker crashes after you send an answer
but before the workflow processes it, Temporal re-delivers the signal
when the worker restarts.

### Gatekeeper decision loop (detail)

```
messages = [{"role": "user", "content": task}]

iteration 1:
  gatekeeper_activity(messages)
    → LLM returns JSON: {action, question, reasoning, enriched_brief, chosen_agent, confidence}
    
  if action == "working":
    append reasoning to messages
    loop (max 8 iterations before forced handoff)

  if action == "ask":
    set stage = "waiting_for_answer"
    wait_condition(answer_ready)      ← durable pause
    append answer to messages
    set stage = "analyzing"
    loop

  if action == "ready" and confidence >= 0.8:
    spawn child workflow → return result
```

### Child workflow handoff (detail)

```python
child_id = f"{workflow_id}-{chosen_agent}"   # e.g. "gatekeeper-abc123-hermes"

result = await workflow.execute_child_workflow(
    HermesWorkflow.run,
    args=[enriched_brief],
    id=child_id,
    task_queue="gatekeeper-queue",
    execution_timeout=timedelta(hours=1),
)
```

In Temporal UI at http://localhost:8233 you will see:
- `gatekeeper-abc123` — parent row, status Completed
- `gatekeeper-abc123-hermes` — child row, linked, status Completed

### Specialist blocker protocol

When the specialist LLM needs more information it outputs a tagged line:

```
SPECIALIST_BLOCKER: {"question": "What firmware version is running on the device?"}
```

`specialist_activity` scans every line of the raw LLM output for this tag.
If found, it returns `{"blocker": True, "question": "..."}` instead of a summary.
The specialist workflow then:
1. Sets `stage = "blocked"`, stores the question
2. `wait_condition(answer_ready)` — durable pause
3. Appends answer to `self._context` dict
4. Loops — next `specialist_activity` call includes all prior answers

### Skills system

Drop any `.md` file into `skills/<agent>/` and the specialist picks it up
automatically on the next call — no code changes needed.

```
skills/
├── hermes/
│   ├── cereg_states.md        ← CEREG registration codes reference
│   └── ofono_commands.md      ← useful ofono debugging commands
├── wifi/
│   └── dhcp_debug.md
├── audio/
│   └── a2dp_codec_guide.md
└── scribe/
    └── style_guide.md
```

`specialist_activity` loads all `.md` files for the active agent and
injects them into the system prompt before calling the LLM.

---

## File Structure

```
flow-gatekeeper-v2/
│
├── activities/
│   ├── __init__.py
│   ├── gatekeeper_activity.py   ← LLM triage → GatekeeperDecision
│   │     @activity.defn gatekeeper_activity(messages: list) -> GatekeeperDecision
│   │     Calls Ollama, strips <think> blocks, parses JSON
│   │
│   └── specialist_activity.py   ← LLM specialist work, loads skills
│         @activity.defn specialist_activity(inp: SpecialistInput) -> dict
│         Returns {"complete": True, "summary": ...} or {"blocker": True, "question": ...}
│
├── workflows/
│   ├── __init__.py
│   ├── gatekeeper_workflow.py   ← parent workflow, triage loop
│   │     @workflow.defn GatekeeperWorkflow
│   │     signals: user_answer(str)
│   │     queries: get_status() -> dict
│   │     max 8 iterations, confidence threshold 0.8
│   │
│   ├── hermes_workflow.py       ← cellular specialist (AGENT = "hermes")
│   ├── wifi_workflow.py         ← wifi specialist   (AGENT = "wifi")
│   ├── audio_workflow.py        ← audio specialist  (AGENT = "audio")
│   └── scribe_workflow.py       ← docs specialist   (AGENT = "scribe")
│         All four: signals user_answer, queries get_status, blocker loop
│
├── skills/
│   ├── hermes/    ← drop .md skill files here
│   ├── wifi/
│   ├── audio/
│   └── scribe/
│
├── worker.py        ← registers all 5 workflows + 2 activities, sandbox config
├── run.py           ← terminal client, poll loop, handles I/O
├── requirements.txt
└── pyproject.toml
```

---

## Rebuilding from Scratch

Follow this order. Each step depends on the one before it.

### Step 1 — Scaffold directories

```bash
mkdir -p flow-gatekeeper-v2/{activities,workflows,skills/{hermes,wifi,audio,scribe}}
touch flow-gatekeeper-v2/activities/__init__.py
touch flow-gatekeeper-v2/workflows/__init__.py
```

### Step 2 — requirements.txt and pyproject.toml

```toml
# pyproject.toml
[project]
name = "flow-gatekeeper-v2"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["temporalio", "openai"]
```

```text
# requirements.txt
temporalio
openai
```

Install:
```bash
cd flow-gatekeeper-v2
uv venv && uv pip install -r requirements.txt
```

### Step 3 — activities/gatekeeper_activity.py

Key pieces to get right:
- `AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama", max_retries=0)`
- Model: `qwen3-coder:latest`
- System prompt must name all 4 agents with their exact `chosen_agent` strings
- Strip `<think>.*?</think>` (re.DOTALL) and markdown fences before `json.loads()`
- On `JSONDecodeError`, return `GatekeeperDecision(action="working")` — never raise

```python
@dataclass
class GatekeeperDecision:
    action: str = "working"   # "working" | "ask" | "ready"
    question: str = ""
    reasoning: str = ""
    enriched_brief: str = ""
    chosen_agent: str = ""    # "hermes" | "wifi" | "audio" | "scribe"
    confidence: float = 0.0
```

### Step 4 — activities/specialist_activity.py

Key pieces:
- Load `skills/<agent>/*.md` at runtime via `Path(__file__).parent.parent / "skills" / agent`
- Detect `SPECIALIST_BLOCKER: {"question": "..."}` by scanning each output line
- Return `{"blocker": True, "question": ...}` or `{"complete": True, "summary": ...}`
- `timeout=300.0` — specialists take longer than the gatekeeper

```python
@dataclass
class SpecialistInput:
    agent: str
    brief: str
    context: dict = field(default_factory=dict)
```

### Step 5 — workflows/gatekeeper_workflow.py

Key pieces:
- Import workflow classes inside `workflow.unsafe.imports_passed_through()` block
- `_WORKFLOW_MAP = {"hermes": HermesWorkflow, "wifi": WifiWorkflow, ...}`
- `CONFIDENCE_THRESHOLD = 0.8`
- `MAX_ITERATIONS = 8` — guard against infinite LLM loops
- On `action == "ask"`: set `_pending_question`, `await workflow.wait_condition(lambda: self._answer_ready)`
- On `action == "ready"`: `await workflow.execute_child_workflow(specialist_cls.run, args=[brief], id=child_id, ...)`
- After MAX_ITERATIONS: force handoff to `self._chosen_agent` if one was identified

### Step 6 — four specialist workflows

Each is identical except for the `AGENT` constant and class name.
Template (copy, change `AGENT` and class name):

```python
AGENT = "hermes"   # change per specialist

@workflow.defn
class HermesWorkflow:
    def __init__(self):
        self._stage = "working"
        self._pending_question = ""
        self._answer_ready = False
        self._user_answer = ""
        self._context = {}

    @workflow.signal
    async def user_answer(self, answer: str) -> None:
        self._user_answer = answer
        self._answer_ready = True

    @workflow.query
    def get_status(self) -> dict:
        return {"stage": self._stage, "pending_question": self._pending_question}

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
```

### Step 7 — worker.py

Critical: include the `SandboxedWorkflowRunner` configuration or you will
get `annotated_types` import warnings on the second workflow run.

```python
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner, SandboxRestrictions

_sandbox = SandboxedWorkflowRunner(
    restrictions=SandboxRestrictions.default.with_passthrough_modules(
        "annotated_types", "pydantic", "pydantic_core",
    )
)

worker = Worker(
    client,
    task_queue="gatekeeper-queue",
    workflows=[GatekeeperWorkflow, HermesWorkflow, WifiWorkflow, AudioWorkflow, ScribeWorkflow],
    activities=[gatekeeper_activity, specialist_activity],
    workflow_runner=_sandbox,
)
```

### Step 8 — run.py

Critical section — the blocker polling must look up the specialist class
dynamically from `_WORKFLOW_MAP`, not use a hardcoded class:

```python
_WORKFLOW_MAP = {"hermes": HermesWorkflow, "wifi": WifiWorkflow,
                 "audio": AudioWorkflow, "scribe": ScribeWorkflow}

# inside the poll loop:
chosen = status.get("chosen_agent", "")
specialist_cls = _WORKFLOW_MAP.get(chosen)
if specialist_cls and child_handle:
    cs = await child_handle.query(specialist_cls.get_status)
    if cs.get("stage") == "blocked":
        answer = input("Your answer: ").strip()
        await child_handle.signal(specialist_cls.user_answer, answer)
```

---

## Adding a New Specialist Agent

1. Create `workflows/bluetooth_workflow.py` — copy any existing specialist,
   change `AGENT = "bluetooth"` and `class BluetoothWorkflow`.

2. Create `skills/bluetooth/` directory and drop `.md` skill files in.

3. Add a base prompt in `activities/specialist_activity.py`:
   ```python
   _BASE_PROMPTS["bluetooth"] = "You are Bluetooth-Agent, ..."
   ```

4. Add to `_WORKFLOW_MAP` in both `gatekeeper_workflow.py` and `run.py`:
   ```python
   "bluetooth": BluetoothWorkflow,
   ```

5. Register in `worker.py`:
   ```python
   workflows=[..., BluetoothWorkflow],
   ```

6. Update the gatekeeper system prompt in `gatekeeper_activity.py` to
   describe the new agent and its domain.

---

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `TEMPORAL_ADDRESS` | `localhost:7233` | Temporal cluster gRPC address |

Set them inline or export before running:
```bash
OLLAMA_HOST=http://192.168.1.5:11434 uv run python worker.py
```

---

## Temporal UI Guide

Open http://localhost:8233 while running.

| What you see | What it means |
|---|---|
| `GatekeeperWorkflow` row — Running | Gatekeeper is still analyzing or waiting for your answer |
| `GatekeeperWorkflow` row — Completed | Task fully done, specialist finished |
| `gatekeeper-abc123-hermes` row | Specialist child, spawned on confident handoff |
| `WorkflowExecutionSignaled` event | A `user_answer` signal was received |
| `ActivityTaskScheduled` event | An Ollama call was dispatched |
| `ActivityTaskCompleted` event | Ollama returned a result (stored durably) |
| `ChildWorkflowExecutionStarted` event | Parent spawned the specialist child |

Click any workflow row → History tab to see every event in order.

---

## Troubleshooting

**Worker shows `annotated_types` sandbox warning**
Ensure `SandboxedWorkflowRunner` with pydantic passthrough is in `worker.py` (Step 7 above).

**Gatekeeper stuck on "Analyzing..." forever**
The LLM is returning `action="working"` in a loop. `MAX_ITERATIONS = 8` will
force a handoff after 8 turns. If it happens repeatedly, check that the
gatekeeper system prompt includes the right agent domains for your task type.

**Ollama connection refused**
Verify Ollama is running: `ollama list`. Restart: `ollama serve`.

**Workflow ID already exists error**
The Temporal dev server remembers workflow IDs within the session. Each
`run.py` invocation generates a new random ID so this should not happen.
If it does, restart `temporal server start-dev`.

**Worker task failures / non-determinism errors**
Stop the worker, restart `temporal server start-dev` (clears all history),
then restart the worker. Non-determinism errors occur when workflow code
changes while a workflow is mid-execution.

**Persist history across Temporal server restarts**
```bash
temporal server start-dev --db-filename temporal.db
```
