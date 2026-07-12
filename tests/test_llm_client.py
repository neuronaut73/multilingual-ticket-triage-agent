"""
Unit tests for src/infrastructure/llm_client.py.

requests.post is mocked so Ollama is never called.
Tests verify:
  - The correct endpoint URL is called.
  - The payload includes model, prompt, format, and options.
  - The 'response' field from the Ollama JSON is returned.
  - HTTP errors are propagated as requests.HTTPError.
"""

from unittest.mock import MagicMock, patch

import pytest
import requests

from src.infrastructure.llm_client import OllamaClient


def make_client() -> OllamaClient:
    return OllamaClient(
        base_url="http://localhost:11434",
        model_name="qwen3:8b",
        temperature=0.1,
        timeout_seconds=30,
    )


def make_mock_response(response_text: str, status_code: int = 200) -> MagicMock:
    mock = MagicMock()
    mock.status_code = status_code
    mock.json.return_value = {"response": response_text, "done": True}
    if status_code >= 400:
        mock.raise_for_status.side_effect = requests.HTTPError(
            response=mock, request=MagicMock()
        )
    else:
        mock.raise_for_status.return_value = None
    return mock


# ─── URL and payload ──────────────────────────────────────────────────────────

def test_generate_json_calls_correct_url():
    with patch("requests.post") as mock_post:
        mock_post.return_value = make_mock_response('{"topic": "Other"}')
        client = make_client()
        client.generate_json("test prompt")

    called_url = mock_post.call_args[0][0]
    assert called_url == "http://localhost:11434/api/generate"


def test_generate_json_sends_correct_payload():
    with patch("requests.post") as mock_post:
        mock_post.return_value = make_mock_response('{"topic": "Other"}')
        client = make_client()
        client.generate_json("test prompt")

    payload = mock_post.call_args[1]["json"]
    assert payload["model"]          == "qwen3:8b"
    assert payload["prompt"]         == "test prompt"
    assert payload["format"]         == "json"
    assert payload["stream"]         is False
    assert payload["options"]["temperature"] == pytest.approx(0.1)


# ─── Response parsing ─────────────────────────────────────────────────────────

def test_generate_json_returns_response_field():
    expected = '{"topic": "Billing / Payment", "urgency": "Low"}'
    with patch("requests.post") as mock_post:
        mock_post.return_value = make_mock_response(expected)
        client = make_client()
        result = client.generate_json("test prompt")

    assert result == expected


# ─── Error handling ───────────────────────────────────────────────────────────

def test_generate_json_raises_on_http_error():
    with patch("requests.post") as mock_post:
        mock_post.return_value = make_mock_response("", status_code=500)
        client = make_client()
        with pytest.raises(requests.HTTPError):
            client.generate_json("test prompt")


def test_base_url_trailing_slash_is_stripped():
    """base_url with trailing slash should still call the correct URL."""
    client = OllamaClient(
        base_url="http://localhost:11434/",
        model_name="qwen3:8b",
        temperature=0.1,
        timeout_seconds=30,
    )
    with patch("requests.post") as mock_post:
        mock_post.return_value = make_mock_response('{"topic": "Other"}')
        client.generate_json("test prompt")

    called_url = mock_post.call_args[0][0]
    assert called_url == "http://localhost:11434/api/generate"
    assert "//" not in called_url.replace("://", "___")
