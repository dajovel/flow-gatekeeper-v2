"""
gatekeeper_activity — LLM triage that decides which specialist to call.

Receives the running message history (user task + prior assistant turns) and
returns a GatekeeperDecision telling the workflow what to do next:

  action = "working"  → still gathering info, append reasoning to history
  action = "ask"      → needs a human answer, question field is populated
  action = "ready"    → confident enough to hand off (confidence >= 0.8)

The LLM is qwen3-coder running locally via Ollama.
"""

import json
import os
import re
from dataclasses import dataclass, field

from openai import AsyncOpenAI
from temporalio import activity

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL = "qwen3-coder:latest"

# ── Specialist registry ───────────────────────────────────────────────────────
# Shown to the LLM so it knows which agents exist and what they handle.
SYSTEM_PROMPT = """
You are Apollo, a triage agent. Your only job is to gather enough information
about a task and then route it to the right specialist agent.

Available specialists:
  hermes → Cellular / modem issues (CEREG, ofono, signal, handoff, LTE, 5G, SIM)
  wifi   → WiFi issues (DHCP, SSID, roaming, 802.11, latency, AP, WPA)
  audio  → Audio AND Bluetooth connectivity issues (A2DP, Bluetooth pairing, Bluetooth
            disconnecting, codec, PCM, ALSA, SCO, HFP, HSP, BLE, Bluetooth audio)
  scribe → Documentation (writing, reviewing, summarising technical content)

Each turn you MUST reply with ONLY a JSON object — no prose, no markdown fences:
{
  "action":        "working" | "ask" | "ready",
  "question":      "question text if action is ask, else empty string",
  "reasoning":     "your brief analysis of what you know so far",
  "enriched_brief": "a polished task description to hand to the specialist",
  "chosen_agent":  "hermes" | "wifi" | "audio" | "scribe" | "",
  "confidence":    0.0 to 1.0
}

Rules:
- Use "ask" when you need one specific piece of info to be certain.
- Use "ready" only when confidence >= 0.8 and chosen_agent is set.
- Use "working" ONLY once while reasoning. Do not stay in "working" for more than
  one turn — you must either ask a question or commit to a routing decision.
- Never route to a specialist without setting chosen_agent.
- When in doubt between two agents, pick the most likely one and set confidence
  to 0.75 so you can ask one clarifying question to confirm.
""".strip()


@dataclass
class GatekeeperDecision:
    action: str = "working"
    question: str = ""
    reasoning: str = ""
    enriched_brief: str = ""
    chosen_agent: str = ""
    confidence: float = 0.0


@activity.defn
async def gatekeeper_activity(messages: list) -> GatekeeperDecision:
    """
    Calls the LLM with the full conversation history and returns a routing decision.
    Non-deterministic — must be an activity, never called directly from workflow code.
    """
    client = AsyncOpenAI(
        base_url=f"{OLLAMA_HOST}/v1",
        api_key="ollama",
        max_retries=0,
        timeout=120.0,
    )

    response = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
    )

    raw = response.choices[0].message.content or ""

    # qwen3 emits <think>...</think> blocks before its actual answer — strip them
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    # strip markdown code fences if the model wrapped the JSON anyway
    raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("` \n")

    try:
        data = json.loads(raw)
        return GatekeeperDecision(
            action=data.get("action", "working"),
            question=data.get("question", ""),
            reasoning=data.get("reasoning", ""),
            enriched_brief=data.get("enriched_brief", ""),
            chosen_agent=data.get("chosen_agent", ""),
            confidence=float(data.get("confidence", 0.0)),
        )
    except (json.JSONDecodeError, ValueError):
        # If the LLM didn't produce valid JSON, keep looping
        return GatekeeperDecision(action="working", reasoning=raw[:300])
