"""
Tests for activities/gatekeeper_activity.py

All LLM calls are mocked at the AsyncOpenAI constructor level so no
real Ollama instance is needed.

Coverage areas
──────────────
1. JSON parsing:    valid payloads, missing keys, all three action values
2. Sanitisation:    <think> block stripping, markdown-fence stripping
3. Error handling:  JSON decode errors, ValueError on bad confidence, None content
4. LLM call shape:  system prompt is first message, conversation history appended
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from activities.gatekeeper_activity import (
    GatekeeperDecision,
    SYSTEM_PROMPT,
    gatekeeper_activity,
)
from tests.conftest import make_llm_response


# ---------------------------------------------------------------------------
# Fixture: patch AsyncOpenAI so every test gets a fresh mock LLM client
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_openai():
    """
    Patches ``AsyncOpenAI`` inside gatekeeper_activity so no network calls
    are made.  Yields the *instance* mock so tests can set
    ``mock_openai.chat.completions.create.return_value``.
    """
    instance = AsyncMock()
    instance.chat = MagicMock()
    instance.chat.completions = MagicMock()
    instance.chat.completions.create = AsyncMock()

    with patch("activities.gatekeeper_activity.AsyncOpenAI", return_value=instance):
        yield instance


# ---------------------------------------------------------------------------
# 1. JSON parsing — valid payloads
# ---------------------------------------------------------------------------

class TestValidJsonParsing:
    async def test_ready_decision_all_fields(self, mock_openai):
        payload = json.dumps({
            "action": "ready",
            "chosen_agent": "hermes",
            "confidence": 0.9,
            "enriched_brief": "LTE signal drops on handoff",
            "reasoning": "clearly a cellular issue",
            "question": "",
        })
        mock_openai.chat.completions.create.return_value = make_llm_response(payload)

        result = await gatekeeper_activity([{"role": "user", "content": "LTE drops"}])

        assert result.action == "ready"
        assert result.chosen_agent == "hermes"
        assert result.confidence == pytest.approx(0.9)
        assert result.enriched_brief == "LTE signal drops on handoff"
        assert result.reasoning == "clearly a cellular issue"
        assert result.question == ""

    async def test_ask_decision(self, mock_openai):
        payload = json.dumps({
            "action": "ask",
            "question": "Which device model is affected?",
            "chosen_agent": "",
            "confidence": 0.5,
            "reasoning": "need more specifics",
            "enriched_brief": "",
        })
        mock_openai.chat.completions.create.return_value = make_llm_response(payload)

        result = await gatekeeper_activity([{"role": "user", "content": "signal problem"}])

        assert result.action == "ask"
        assert result.question == "Which device model is affected?"

    async def test_working_decision(self, mock_openai):
        payload = json.dumps({
            "action": "working",
            "reasoning": "still analysing",
            "chosen_agent": "",
            "confidence": 0.3,
            "question": "",
            "enriched_brief": "",
        })
        mock_openai.chat.completions.create.return_value = make_llm_response(payload)

        result = await gatekeeper_activity([{"role": "user", "content": "vague task"}])

        assert result.action == "working"
        assert result.reasoning == "still analysing"

    async def test_all_four_agent_values_parsed(self, mock_openai):
        for agent in ("hermes", "wifi", "audio", "scribe"):
            payload = json.dumps({
                "action": "ready",
                "chosen_agent": agent,
                "confidence": 0.85,
                "enriched_brief": f"{agent} task",
                "reasoning": "ok",
                "question": "",
            })
            mock_openai.chat.completions.create.return_value = make_llm_response(payload)

            result = await gatekeeper_activity([{"role": "user", "content": "task"}])
            assert result.chosen_agent == agent

    async def test_confidence_parsed_from_string(self, mock_openai):
        """confidence coming back as a JSON string should still be converted to float."""
        payload = json.dumps({
            "action": "ready",
            "chosen_agent": "scribe",
            "confidence": "0.88",   # string, not number
            "enriched_brief": "write doc",
            "reasoning": "docs",
            "question": "",
        })
        mock_openai.chat.completions.create.return_value = make_llm_response(payload)

        result = await gatekeeper_activity([{"role": "user", "content": "write README"}])

        assert isinstance(result.confidence, float)
        assert result.confidence == pytest.approx(0.88)

    async def test_missing_optional_fields_use_defaults(self, mock_openai):
        """A minimal JSON with only 'action' should not raise."""
        mock_openai.chat.completions.create.return_value = make_llm_response('{"action": "working"}')

        result = await gatekeeper_activity([{"role": "user", "content": "task"}])

        assert result.action == "working"
        assert result.question == ""
        assert result.reasoning == ""
        assert result.chosen_agent == ""
        assert result.confidence == 0.0
        assert result.enriched_brief == ""


# ---------------------------------------------------------------------------
# 2. Output sanitisation — <think> blocks and markdown fences
# ---------------------------------------------------------------------------

class TestOutputSanitisation:
    async def test_single_think_block_stripped(self, mock_openai):
        payload = (
            "<think>internal monologue</think>\n"
            '{"action": "ready", "chosen_agent": "wifi", "confidence": 0.85, '
            '"enriched_brief": "wifi fix", "reasoning": "wifi", "question": ""}'
        )
        mock_openai.chat.completions.create.return_value = make_llm_response(payload)

        result = await gatekeeper_activity([{"role": "user", "content": "wifi drops"}])

        assert result.action == "ready"
        assert result.chosen_agent == "wifi"

    async def test_multiline_think_block_stripped(self, mock_openai):
        payload = (
            "<think>\nline 1\nline 2\nline 3\n</think>\n"
            '{"action": "ask", "question": "Which router?", "chosen_agent": "", '
            '"confidence": 0.5, "reasoning": "r", "enriched_brief": ""}'
        )
        mock_openai.chat.completions.create.return_value = make_llm_response(payload)

        result = await gatekeeper_activity([{"role": "user", "content": "wifi problem"}])

        assert result.action == "ask"
        assert result.question == "Which router?"

    async def test_json_markdown_fence_stripped(self, mock_openai):
        payload = (
            "```json\n"
            '{"action": "ready", "chosen_agent": "audio", "confidence": 0.9, '
            '"enriched_brief": "audio issue", "reasoning": "bluetooth", "question": ""}\n'
            "```"
        )
        mock_openai.chat.completions.create.return_value = make_llm_response(payload)

        result = await gatekeeper_activity([{"role": "user", "content": "BT drops"}])

        assert result.action == "ready"
        assert result.chosen_agent == "audio"

    async def test_bare_markdown_fence_stripped(self, mock_openai):
        """Fence without a language tag (```) should also be removed."""
        payload = (
            "```\n"
            '{"action": "working", "reasoning": "ok", "chosen_agent": "", '
            '"confidence": 0.2, "question": "", "enriched_brief": ""}\n'
            "```"
        )
        mock_openai.chat.completions.create.return_value = make_llm_response(payload)

        result = await gatekeeper_activity([{"role": "user", "content": "task"}])

        assert result.action == "working"

    async def test_think_and_fence_both_stripped(self, mock_openai):
        payload = (
            "<think>hidden</think>\n"
            "```json\n"
            '{"action": "ready", "chosen_agent": "hermes", "confidence": 0.9, '
            '"enriched_brief": "lte", "reasoning": "r", "question": ""}\n'
            "```"
        )
        mock_openai.chat.completions.create.return_value = make_llm_response(payload)

        result = await gatekeeper_activity([{"role": "user", "content": "lte"}])

        assert result.action == "ready"


# ---------------------------------------------------------------------------
# 3. Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    async def test_invalid_json_returns_working_action(self, mock_openai):
        mock_openai.chat.completions.create.return_value = make_llm_response("This is not JSON")

        result = await gatekeeper_activity([{"role": "user", "content": "task"}])

        assert result.action == "working"

    async def test_truncated_json_returns_working(self, mock_openai):
        mock_openai.chat.completions.create.return_value = make_llm_response(
            '{"action": "ready", "chosen_agent":'  # truncated
        )

        result = await gatekeeper_activity([{"role": "user", "content": "task"}])

        assert result.action == "working"

    async def test_invalid_json_reasoning_is_capped_at_300_chars(self, mock_openai):
        """Raw LLM output that isn't JSON gets stored in reasoning, capped to 300 chars."""
        long_junk = "X" * 500
        mock_openai.chat.completions.create.return_value = make_llm_response(long_junk)

        result = await gatekeeper_activity([{"role": "user", "content": "task"}])

        assert result.action == "working"
        assert len(result.reasoning) <= 300

    async def test_none_content_from_llm_handled_gracefully(self, mock_openai):
        """If the LLM returns None content the activity should not raise."""
        mock_openai.chat.completions.create.return_value = make_llm_response(None)

        result = await gatekeeper_activity([{"role": "user", "content": "task"}])

        assert result.action == "working"

    async def test_invalid_confidence_type_falls_back(self, mock_openai):
        """If 'confidence' cannot be cast to float, the whole record falls back to working."""
        payload = json.dumps({
            "action": "ready",
            "chosen_agent": "hermes",
            "confidence": "not-a-number",
            "enriched_brief": "lte",
            "reasoning": "r",
            "question": "",
        })
        mock_openai.chat.completions.create.return_value = make_llm_response(payload)

        result = await gatekeeper_activity([{"role": "user", "content": "lte"}])

        # ValueError from float("not-a-number") → falls back to action="working"
        assert result.action == "working"


# ---------------------------------------------------------------------------
# 4. LLM call shape
# ---------------------------------------------------------------------------

class TestLlmCallShape:
    async def test_system_prompt_is_first_message(self, mock_openai):
        mock_openai.chat.completions.create.return_value = make_llm_response(
            '{"action": "working", "reasoning": "r", "question": "", '
            '"enriched_brief": "", "chosen_agent": "", "confidence": 0.0}'
        )
        user_messages = [{"role": "user", "content": "LTE drops"}]

        await gatekeeper_activity(user_messages)

        call_kwargs = mock_openai.chat.completions.create.call_args.kwargs
        sent = call_kwargs["messages"]
        assert sent[0]["role"] == "system"
        assert sent[0]["content"] == SYSTEM_PROMPT

    async def test_conversation_history_appended_after_system_prompt(self, mock_openai):
        mock_openai.chat.completions.create.return_value = make_llm_response(
            '{"action": "working", "reasoning": "r", "question": "", '
            '"enriched_brief": "", "chosen_agent": "", "confidence": 0.0}'
        )
        user_messages = [
            {"role": "user", "content": "LTE drops"},
            {"role": "assistant", "content": "analysing"},
            {"role": "user", "content": "device XYZ"},
        ]

        await gatekeeper_activity(user_messages)

        call_kwargs = mock_openai.chat.completions.create.call_args.kwargs
        sent = call_kwargs["messages"]
        # System prompt + 3 history messages
        assert len(sent) == 4
        assert sent[1:] == user_messages
