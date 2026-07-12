"""
Unit tests for src/application/analyzer.py.

All tests use a FakeLLMClient — Ollama is never called.
Tests verify:
  - Valid JSON is parsed into LLMAnalysis.
  - Invalid JSON triggers one retry.
  - An invalid topic value triggers one retry.
  - A successful retry returns the corrected analysis.
  - A failed retry returns the fallback LLMAnalysis.
  - The prompt includes all assignment topics and neighbor evidence.
  - No openai / anthropic / instructor import is present.
"""

import json

import pytest

from src.application.analyzer import LocalLLMAnalyzer, _build_prompt
from src.domain.enums import Topic, Urgency
from src.domain.models import LLMAnalysis, NeighborEvidence, NeighborPrediction, TicketInput

# ─── Shared fixtures ──────────────────────────────────────────────────────────

VALID_JSON = json.dumps({
    "topic": "Technical / Online Access",
    "urgency": "High",
    "missing_info": False,
    "missing_fields": [],
    "confidence": 0.86,
    "short_note": "Customer reports a login access issue on the portal.",
})

INVALID_JSON = "this is not valid json {"

INVALID_TOPIC_JSON = json.dumps({
    "topic": "Unknown Topic",       # not a valid Topic enum value
    "urgency": "High",
    "missing_info": False,
    "missing_fields": [],
    "confidence": 0.86,
    "short_note": "Customer reports a login access issue.",
})

INVALID_URGENCY_JSON = json.dumps({
    "topic": "Technical / Online Access",
    "urgency": "Critical",          # not a valid Urgency enum value
    "missing_info": False,
    "missing_fields": [],
    "confidence": 0.86,
    "short_note": "Customer reports a login access issue.",
})


class FakeLLMClient:
    """
    Returns responses in sequence.

    All prompts are recorded in self.prompts so tests can inspect them.
    Raises StopIteration if more calls are made than responses provided.
    """

    def __init__(self, *responses: str) -> None:
        self._iter   = iter(responses)
        self.prompts: list[str] = []

    def generate_json(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return next(self._iter)


def make_ticket() -> TicketInput:
    return TicketInput(
        ticket_id="T001",
        subject="Cannot login to the customer portal",
        body="I have been trying to log in for three days. I get a 403 Forbidden error.",
        raw_text="Cannot login to the customer portal I have been trying to log in for three days.",
        cleaned_text="Cannot login to the customer portal I have been trying to log in for three days.",
        representation_text=(
            "Subject: Cannot login to the customer portal\n\n"
            "Body: I have been trying to log in for three days. I get a 403 Forbidden error."
        ),
        text_snippet="Cannot login to the customer portal",
    )


def make_prediction() -> NeighborPrediction:
    neighbor = NeighborEvidence(
        ticket_id="REF001",
        distance=0.15,
        similarity=0.87,
        actual_queue="Technical Support",
        actual_priority="high",
        actual_type="Incident",
        actual_tags=["Login", "Portal"],
        proxy_topic="Technical / Online Access",
        proxy_urgency="High",
        text_snippet="User cannot access the portal after password reset.",
    )
    return NeighborPrediction(
        predicted_queue="Technical Support",
        queue_confidence=0.80,
        predicted_priority="high",
        priority_confidence=0.75,
        predicted_proxy_topic="Technical / Online Access",
        proxy_topic_confidence=0.78,
        predicted_tags=["Login", "Portal", "Authentication"],
        neighbors=[neighbor],
    )


# ─── Happy path ───────────────────────────────────────────────────────────────

def test_valid_json_returns_llm_analysis():
    client   = FakeLLMClient(VALID_JSON)
    analyzer = LocalLLMAnalyzer(llm_client=client, max_retries=1)
    result   = analyzer.analyze(make_ticket(), make_prediction())

    assert isinstance(result, LLMAnalysis)
    assert result.topic    == Topic.TECHNICAL
    assert result.urgency  == Urgency.HIGH
    assert result.confidence == pytest.approx(0.86)
    assert result.missing_info is False
    assert len(client.prompts) == 1


# ─── Retry on invalid JSON ────────────────────────────────────────────────────

def test_invalid_json_triggers_retry():
    client   = FakeLLMClient(INVALID_JSON, VALID_JSON)
    analyzer = LocalLLMAnalyzer(llm_client=client, max_retries=1)
    result   = analyzer.analyze(make_ticket(), make_prediction())

    assert len(client.prompts) == 2, "Expected exactly one retry"
    assert result.topic == Topic.TECHNICAL


def test_invalid_topic_triggers_retry():
    client   = FakeLLMClient(INVALID_TOPIC_JSON, VALID_JSON)
    analyzer = LocalLLMAnalyzer(llm_client=client, max_retries=1)
    result   = analyzer.analyze(make_ticket(), make_prediction())

    assert len(client.prompts) == 2
    assert result.topic == Topic.TECHNICAL


def test_invalid_urgency_triggers_retry():
    client   = FakeLLMClient(INVALID_URGENCY_JSON, VALID_JSON)
    analyzer = LocalLLMAnalyzer(llm_client=client, max_retries=1)
    result   = analyzer.analyze(make_ticket(), make_prediction())

    assert len(client.prompts) == 2
    assert result.urgency == Urgency.HIGH


# ─── Retry success ────────────────────────────────────────────────────────────

def test_retry_success_returns_corrected_analysis():
    client   = FakeLLMClient(INVALID_JSON, VALID_JSON)
    analyzer = LocalLLMAnalyzer(llm_client=client, max_retries=1)
    result   = analyzer.analyze(make_ticket(), make_prediction())

    assert result.topic      == Topic.TECHNICAL
    assert result.confidence == pytest.approx(0.86)
    assert result.short_note == "Customer reports a login access issue on the portal."


# ─── Retry failure → fallback ─────────────────────────────────────────────────

def test_retry_failure_returns_fallback():
    client   = FakeLLMClient(INVALID_JSON, INVALID_JSON)
    analyzer = LocalLLMAnalyzer(llm_client=client, max_retries=1)
    result   = analyzer.analyze(make_ticket(), make_prediction())

    assert result.topic      == Topic.OTHER
    assert result.urgency    == Urgency.MEDIUM
    assert result.confidence == pytest.approx(0.0)
    assert result.missing_info is True
    assert "valid_structured_llm_output" in result.missing_fields


def test_max_retries_zero_skips_retry():
    """With max_retries=0, invalid output goes straight to fallback."""
    client   = FakeLLMClient(INVALID_JSON)
    analyzer = LocalLLMAnalyzer(llm_client=client, max_retries=0)
    result   = analyzer.analyze(make_ticket(), make_prediction())

    assert len(client.prompts) == 1, "No retry should occur with max_retries=0"
    assert result.topic      == Topic.OTHER
    assert result.confidence == pytest.approx(0.0)


# ─── Prompt content ───────────────────────────────────────────────────────────

def test_prompt_contains_all_assignment_topics():
    client   = FakeLLMClient(VALID_JSON)
    analyzer = LocalLLMAnalyzer(llm_client=client, max_retries=1)
    analyzer.analyze(make_ticket(), make_prediction())

    prompt = client.prompts[0]
    assert "Policy / Contract"        in prompt
    assert "Claims / Damage"          in prompt
    assert "Billing / Payment"        in prompt
    assert "Technical / Online Access" in prompt
    assert "Other"                    in prompt


def test_prompt_contains_allowed_urgency_values():
    client   = FakeLLMClient(VALID_JSON)
    analyzer = LocalLLMAnalyzer(llm_client=client, max_retries=1)
    analyzer.analyze(make_ticket(), make_prediction())

    prompt = client.prompts[0]
    assert "Low"    in prompt
    assert "Medium" in prompt
    assert "High"   in prompt


def test_prompt_contains_neighbor_evidence():
    prediction = make_prediction()
    client     = FakeLLMClient(VALID_JSON)
    analyzer   = LocalLLMAnalyzer(llm_client=client, max_retries=1)
    analyzer.analyze(make_ticket(), prediction)

    prompt = client.prompts[0]
    assert "Technical Support" in prompt   # predicted_queue
    assert "high"              in prompt   # predicted_priority


def test_prompt_contains_ticket_subject_and_body():
    ticket   = make_ticket()
    client   = FakeLLMClient(VALID_JSON)
    analyzer = LocalLLMAnalyzer(llm_client=client, max_retries=1)
    analyzer.analyze(ticket, make_prediction())

    prompt = client.prompts[0]
    assert "Cannot login to the customer portal" in prompt
    assert "403 Forbidden"                       in prompt


def test_correction_prompt_contains_original_response_and_error():
    """The correction prompt must echo the original bad response and the error."""
    client   = FakeLLMClient(INVALID_JSON, VALID_JSON)
    analyzer = LocalLLMAnalyzer(llm_client=client, max_retries=1)
    analyzer.analyze(make_ticket(), make_prediction())

    correction_prompt = client.prompts[1]
    assert INVALID_JSON in correction_prompt
    assert "schema" in correction_prompt.lower() or "topic" in correction_prompt


# ─── No cloud LLM dependency ──────────────────────────────────────────────────

def test_no_cloud_llm_dependency():
    """analyzer.py and llm_client.py must not import openai, anthropic, or instructor."""
    import importlib.util

    for module_name in ("src.application.analyzer", "src.infrastructure.llm_client"):
        spec = importlib.util.find_spec(module_name)
        assert spec is not None, f"Module {module_name} not found"
        with open(spec.origin, encoding="utf-8") as f:
            source = f.read()
        for forbidden in ("import openai", "from openai", "import anthropic", "from anthropic", "import instructor"):
            assert forbidden not in source, (
                f"Forbidden import '{forbidden}' found in {module_name}"
            )
