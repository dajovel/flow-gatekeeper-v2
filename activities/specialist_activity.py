"""
specialist_activity — the actual LLM work done by each specialist agent.

Receives a SpecialistInput (which agent, the task brief, and any answers
already collected from the user) and returns one of two dicts:

  {"complete": True,  "summary": "..."}          → done, return to workflow
  {"blocker":  True,  "question": "..."}          → needs human input, loop again

Skills are loaded from  skills/<agent>/*.md  at activity runtime. Drop any
.md file into the right subfolder and the agent will pick it up automatically.

The LLM signals a blocker by emitting a tagged line anywhere in its output:
  SPECIALIST_BLOCKER: {"question": "what is the firmware version?"}
"""

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from openai import AsyncOpenAI
from temporalio import activity

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL = "qwen3-coder:latest"

SKILLS_DIR = Path(__file__).parent.parent / "skills"

# Per-agent base prompts describing the specialist's role and domain
_BASE_PROMPTS = {
    "hermes": (
        "You are Hermes, a cellular diagnostics specialist. "
        "You investigate modem, LTE/5G signal, CEREG state, ofono, and handoff issues. "
        "Provide a thorough technical analysis and clear recommendations."
    ),
    "wifi": (
        "You are Wifi-Agent, a WiFi diagnostics specialist. "
        "You investigate DHCP failures, SSID issues, 802.11 protocol problems, "
        "roaming, AP associations, and latency. Provide actionable recommendations."
    ),
    "audio": (
        "You are Audio-Agent, an audio diagnostics specialist. "
        "You investigate A2DP, Bluetooth audio, codec negotiation, PCM pipeline, "
        "ALSA, and SCO call quality. Provide a precise root-cause analysis."
    ),
    "scribe": (
        "You are Scribe, a technical documentation specialist. "
        "You write, review, and summarise technical content clearly and accurately. "
        "Produce well-structured output appropriate for the audience."
    ),
}

_BLOCKER_TAG = "SPECIALIST_BLOCKER:"


def _load_skills(agent: str) -> str:
    """Return the concatenated text of all .md files in skills/<agent>/."""
    skill_dir = SKILLS_DIR / agent
    if not skill_dir.exists():
        return ""
    parts = []
    for md_file in sorted(skill_dir.glob("*.md")):
        parts.append(f"--- skill: {md_file.name} ---\n{md_file.read_text()}")
    return "\n\n".join(parts)


@dataclass
class SpecialistInput:
    agent: str
    brief: str
    context: dict = field(default_factory=dict)


@activity.defn
async def specialist_activity(inp: SpecialistInput) -> dict:
    """
    Runs the specialist LLM with the task brief and any previously collected
    blocker answers. Returns a result or a blocker needing human input.
    """
    base = _BASE_PROMPTS.get(inp.agent, "You are a helpful specialist agent.")
    skills_text = _load_skills(inp.agent)

    system_parts = [base]
    if skills_text:
        system_parts.append(f"\nLoaded skills:\n{skills_text}")
    system_parts.append(
        f"\nIf you need information you cannot determine yourself, output exactly:\n"
        f"{_BLOCKER_TAG} {{\"question\": \"your question here\"}}\n"
        f"on its own line. Otherwise produce your complete analysis."
    )
    system_prompt = "\n".join(system_parts)

    # Build the user message from brief + any prior blocker answers
    user_content = inp.brief
    if inp.context:
        answers_text = "\n".join(
            f"Q{i}: {v}" for i, v in enumerate(inp.context.values(), 1)
        )
        user_content += f"\n\nAdditional context provided:\n{answers_text}"

    client = AsyncOpenAI(
        base_url=f"{OLLAMA_HOST}/v1",
        api_key="ollama",
        max_retries=0,
        timeout=300.0,
    )

    response = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )

    raw = response.choices[0].message.content or ""

    # Strip qwen3 thinking blocks
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    # Check for blocker signal
    for line in raw.splitlines():
        if line.strip().startswith(_BLOCKER_TAG):
            payload = line.strip()[len(_BLOCKER_TAG):].strip()
            try:
                data = json.loads(payload)
                return {"blocker": True, "question": data.get("question", payload)}
            except json.JSONDecodeError:
                return {"blocker": True, "question": payload}

    return {"complete": True, "summary": raw}
