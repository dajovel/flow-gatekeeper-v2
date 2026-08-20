"""
Tests for activities/specialist_activity.py

Coverage areas
──────────────
1. _load_skills():      missing dir, empty dir, single file, multiple files (sorted),
                        only .md files loaded (non-.md ignored)
2. SPECIALIST_BLOCKER:  detection in first / middle / last line, valid JSON payload,
                        invalid JSON payload (raw text fallback)
3. Complete response:   no blocker → {"complete": True, "summary": ...}
4. System prompt build: base prompt included, skills injected when present,
                        unknown agent uses generic fallback prompt
5. User content:        context answers formatted as Q1/Q2/…, empty context has no
                        "Additional context" section
6. <think> stripping:   qwen3-style reasoning blocks removed before blocker scan
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from activities.specialist_activity import (
    SpecialistInput,
    _load_skills,
    specialist_activity,
)
from tests.conftest import make_llm_response


# ---------------------------------------------------------------------------
# Fixture: patch AsyncOpenAI inside specialist_activity
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_openai():
    instance = AsyncMock()
    instance.chat = MagicMock()
    instance.chat.completions = MagicMock()
    instance.chat.completions.create = AsyncMock()

    with patch("activities.specialist_activity.AsyncOpenAI", return_value=instance):
        yield instance


# ---------------------------------------------------------------------------
# 1. _load_skills() — pure filesystem function, no mock needed
# ---------------------------------------------------------------------------

class TestLoadSkills:
    def test_missing_directory_returns_empty_string(self, tmp_path, monkeypatch):
        monkeypatch.setattr("activities.specialist_activity.SKILLS_DIR", tmp_path)

        result = _load_skills("nonexistent_agent")

        assert result == ""

    def test_empty_directory_returns_empty_string(self, tmp_path, monkeypatch):
        monkeypatch.setattr("activities.specialist_activity.SKILLS_DIR", tmp_path)
        (tmp_path / "hermes").mkdir()

        result = _load_skills("hermes")

        assert result == ""

    def test_single_skill_file_loaded(self, tmp_path, monkeypatch):
        monkeypatch.setattr("activities.specialist_activity.SKILLS_DIR", tmp_path)
        agent_dir = tmp_path / "hermes"
        agent_dir.mkdir()
        (agent_dir / "cereg_states.md").write_text("# CEREG states\n0 = not registered")

        result = _load_skills("hermes")

        assert "cereg_states.md" in result
        assert "0 = not registered" in result

    def test_multiple_skill_files_concatenated_sorted(self, tmp_path, monkeypatch):
        monkeypatch.setattr("activities.specialist_activity.SKILLS_DIR", tmp_path)
        agent_dir = tmp_path / "hermes"
        agent_dir.mkdir()
        (agent_dir / "b_commands.md").write_text("# B commands")
        (agent_dir / "a_reference.md").write_text("# A reference")

        result = _load_skills("hermes")

        # Files should appear in alphabetical order
        idx_a = result.index("a_reference.md")
        idx_b = result.index("b_commands.md")
        assert idx_a < idx_b

    def test_non_md_files_are_ignored(self, tmp_path, monkeypatch):
        monkeypatch.setattr("activities.specialist_activity.SKILLS_DIR", tmp_path)
        agent_dir = tmp_path / "wifi"
        agent_dir.mkdir()
        (agent_dir / "notes.txt").write_text("should be ignored")
        (agent_dir / "debug.md").write_text("# debug guide")

        result = _load_skills("wifi")

        assert "notes.txt" not in result
        assert "debug.md" in result
        assert "should be ignored" not in result

    def test_skill_separator_format(self, tmp_path, monkeypatch):
        monkeypatch.setattr("activities.specialist_activity.SKILLS_DIR", tmp_path)
        agent_dir = tmp_path / "audio"
        agent_dir.mkdir()
        (agent_dir / "a2dp.md").write_text("content")

        result = _load_skills("audio")

        assert result.startswith("--- skill: a2dp.md ---")


# ---------------------------------------------------------------------------
# 2. SPECIALIST_BLOCKER detection
# ---------------------------------------------------------------------------

class TestBlockerDetection:
    async def test_blocker_on_first_line(self, mock_openai):
        output = 'SPECIALIST_BLOCKER: {"question": "What firmware version?"}\nsome extra text'
        mock_openai.chat.completions.create.return_value = make_llm_response(output)

        result = await specialist_activity(
            SpecialistInput(agent="hermes", brief="LTE issue")
        )

        assert result == {"blocker": True, "question": "What firmware version?"}

    async def test_blocker_in_middle_of_output(self, mock_openai):
        output = (
            "Here is my analysis so far.\n"
            'SPECIALIST_BLOCKER: {"question": "Which band is used?"}\n'
            "This line would not be reached."
        )
        mock_openai.chat.completions.create.return_value = make_llm_response(output)

        result = await specialist_activity(
            SpecialistInput(agent="hermes", brief="LTE issue")
        )

        assert result["blocker"] is True
        assert result["question"] == "Which band is used?"

    async def test_blocker_on_last_line(self, mock_openai):
        output = "Analysis:\n- checked signal\n" + 'SPECIALIST_BLOCKER: {"question": "Got logs?"}'
        mock_openai.chat.completions.create.return_value = make_llm_response(output)

        result = await specialist_activity(
            SpecialistInput(agent="hermes", brief="LTE issue")
        )

        assert result["blocker"] is True
        assert "logs" in result["question"]

    async def test_blocker_with_invalid_json_payload_returns_raw(self, mock_openai):
        """If the JSON after SPECIALIST_BLOCKER: is malformed, return the raw text."""
        output = "SPECIALIST_BLOCKER: not-valid-json"
        mock_openai.chat.completions.create.return_value = make_llm_response(output)

        result = await specialist_activity(
            SpecialistInput(agent="hermes", brief="LTE issue")
        )

        assert result["blocker"] is True
        assert result["question"] == "not-valid-json"

    async def test_no_blocker_tag_returns_complete(self, mock_openai):
        output = "After thorough analysis: replace the antenna module."
        mock_openai.chat.completions.create.return_value = make_llm_response(output)

        result = await specialist_activity(
            SpecialistInput(agent="hermes", brief="LTE issue")
        )

        assert result["complete"] is True
        assert result["summary"] == output


# ---------------------------------------------------------------------------
# 3. Complete response
# ---------------------------------------------------------------------------

class TestCompleteResponse:
    async def test_summary_contains_llm_output(self, mock_openai):
        output = "Root cause: firmware bug in modem driver v3.1. Upgrade to v3.2."
        mock_openai.chat.completions.create.return_value = make_llm_response(output)

        result = await specialist_activity(
            SpecialistInput(agent="hermes", brief="modem crashes")
        )

        assert result == {"complete": True, "summary": output}


# ---------------------------------------------------------------------------
# 4. System prompt construction
# ---------------------------------------------------------------------------

class TestSystemPromptConstruction:
    async def test_known_agent_uses_base_prompt(self, mock_openai):
        mock_openai.chat.completions.create.return_value = make_llm_response("done")

        await specialist_activity(SpecialistInput(agent="hermes", brief="lte issue"))

        call_kwargs = mock_openai.chat.completions.create.call_args.kwargs
        system_content = call_kwargs["messages"][0]["content"]
        assert "Hermes" in system_content
        assert "cellular" in system_content.lower()

    async def test_unknown_agent_uses_generic_fallback_prompt(self, mock_openai):
        mock_openai.chat.completions.create.return_value = make_llm_response("done")

        await specialist_activity(SpecialistInput(agent="unknown_agent", brief="task"))

        call_kwargs = mock_openai.chat.completions.create.call_args.kwargs
        system_content = call_kwargs["messages"][0]["content"]
        assert "helpful specialist" in system_content.lower()

    async def test_skills_injected_into_system_prompt(self, mock_openai, tmp_path, monkeypatch):
        monkeypatch.setattr("activities.specialist_activity.SKILLS_DIR", tmp_path)
        agent_dir = tmp_path / "wifi"
        agent_dir.mkdir()
        (agent_dir / "dhcp_debug.md").write_text("## DHCP debug steps\n1. Check lease")

        mock_openai.chat.completions.create.return_value = make_llm_response("done")

        await specialist_activity(SpecialistInput(agent="wifi", brief="wifi drops"))

        call_kwargs = mock_openai.chat.completions.create.call_args.kwargs
        system_content = call_kwargs["messages"][0]["content"]
        assert "dhcp_debug.md" in system_content
        assert "DHCP debug steps" in system_content

    async def test_no_skills_dir_prompt_has_no_skill_section(self, mock_openai, tmp_path, monkeypatch):
        monkeypatch.setattr("activities.specialist_activity.SKILLS_DIR", tmp_path)
        # no agent subdirectory created → skills_text = ""

        mock_openai.chat.completions.create.return_value = make_llm_response("done")

        await specialist_activity(SpecialistInput(agent="audio", brief="bt drops"))

        call_kwargs = mock_openai.chat.completions.create.call_args.kwargs
        system_content = call_kwargs["messages"][0]["content"]
        assert "Loaded skills" not in system_content

    async def test_blocker_instruction_always_in_system_prompt(self, mock_openai):
        mock_openai.chat.completions.create.return_value = make_llm_response("done")

        await specialist_activity(SpecialistInput(agent="scribe", brief="write readme"))

        call_kwargs = mock_openai.chat.completions.create.call_args.kwargs
        system_content = call_kwargs["messages"][0]["content"]
        assert "SPECIALIST_BLOCKER" in system_content


# ---------------------------------------------------------------------------
# 5. User content construction (context / brief formatting)
# ---------------------------------------------------------------------------

class TestUserContentConstruction:
    async def test_empty_context_no_additional_section(self, mock_openai):
        mock_openai.chat.completions.create.return_value = make_llm_response("done")

        await specialist_activity(SpecialistInput(agent="hermes", brief="my brief", context={}))

        call_kwargs = mock_openai.chat.completions.create.call_args.kwargs
        user_content = call_kwargs["messages"][1]["content"]
        assert "Additional context" not in user_content
        assert user_content == "my brief"

    async def test_single_context_answer_formatted(self, mock_openai):
        mock_openai.chat.completions.create.return_value = make_llm_response("done")

        await specialist_activity(
            SpecialistInput(
                agent="hermes",
                brief="LTE issue",
                context={"answer_1": "v3.2 firmware"},
            )
        )

        call_kwargs = mock_openai.chat.completions.create.call_args.kwargs
        user_content = call_kwargs["messages"][1]["content"]
        assert "Additional context provided" in user_content
        assert "Q1: v3.2 firmware" in user_content

    async def test_multiple_context_answers_formatted_as_q1_q2(self, mock_openai):
        mock_openai.chat.completions.create.return_value = make_llm_response("done")

        await specialist_activity(
            SpecialistInput(
                agent="hermes",
                brief="LTE issue",
                context={"answer_1": "firmware v3.2", "answer_2": "device XYZ"},
            )
        )

        call_kwargs = mock_openai.chat.completions.create.call_args.kwargs
        user_content = call_kwargs["messages"][1]["content"]
        assert "Q1: firmware v3.2" in user_content
        assert "Q2: device XYZ" in user_content

    async def test_brief_always_included_in_user_content(self, mock_openai):
        mock_openai.chat.completions.create.return_value = make_llm_response("done")

        await specialist_activity(
            SpecialistInput(agent="scribe", brief="Write a README for my project")
        )

        call_kwargs = mock_openai.chat.completions.create.call_args.kwargs
        user_content = call_kwargs["messages"][1]["content"]
        assert "Write a README for my project" in user_content


# ---------------------------------------------------------------------------
# 6. <think> stripping
# ---------------------------------------------------------------------------

class TestThinkStripping:
    async def test_think_block_removed_before_blocker_scan(self, mock_openai):
        """A <think> block should not accidentally match SPECIALIST_BLOCKER."""
        output = (
            "<think>I am thinking about SPECIALIST_BLOCKER internally</think>\n"
            "The actual answer: check the antenna."
        )
        mock_openai.chat.completions.create.return_value = make_llm_response(output)

        result = await specialist_activity(
            SpecialistInput(agent="hermes", brief="lte issue")
        )

        # The blocker tag inside <think> should not trigger a blocker response
        assert result["complete"] is True
        assert "check the antenna" in result["summary"]

    async def test_think_block_stripped_from_summary(self, mock_openai):
        output = "<think>hidden thoughts</think>\nVisible answer here."
        mock_openai.chat.completions.create.return_value = make_llm_response(output)

        result = await specialist_activity(
            SpecialistInput(agent="audio", brief="bt issue")
        )

        assert result["complete"] is True
        assert "<think>" not in result["summary"]
        assert "hidden thoughts" not in result["summary"]
