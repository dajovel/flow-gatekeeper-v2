"""
Shared test helpers used across multiple test modules.
"""

from unittest.mock import MagicMock


def make_llm_response(content: str | None) -> MagicMock:
    """
    Build a minimal mock of an OpenAI ChatCompletion response whose
    ``choices[0].message.content`` returns *content*.
    """
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    return response
