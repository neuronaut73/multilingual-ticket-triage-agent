from __future__ import annotations

from pydantic import BaseModel, Field

from .enums import NextAction, Topic, Urgency


# ─── Prediction-time input ────────────────────────────────────────────────────
# Only subject and body are used as inference input. All derived text fields
# are computed from these two during preprocessing.

class TicketInput(BaseModel):
    ticket_id:           str
    subject:             str
    body:                str
    raw_text:            str
    cleaned_text:        str
    representation_text: str
    text_snippet:        str


# ─── Neighbor retrieval ───────────────────────────────────────────────────────

class NeighborEvidence(BaseModel):
    ticket_id:          str
    distance:           float = 0.0   # raw LanceDB _distance, kept for tracing
    similarity:         float = 0.0   # 1 / (1 + distance)
    actual_queue:       str | None = None
    actual_priority:    str | None = None
    actual_type:        str | None = None
    actual_tags:        list[str] = Field(default_factory=list)
    proxy_topic:        str | None = None
    proxy_urgency:      str | None = None
    proxy_next_action:  str | None = None
    text_snippet:       str = ""


class NeighborPrediction(BaseModel):
    predicted_queue:         str | None = None
    queue_confidence:        float = Field(default=0.0, ge=0.0, le=1.0)
    predicted_priority:      str | None = None
    priority_confidence:     float = Field(default=0.0, ge=0.0, le=1.0)
    predicted_proxy_topic:   str | None = None
    proxy_topic_confidence:  float = Field(default=0.0, ge=0.0, le=1.0)
    predicted_tags:          list[str] = Field(default_factory=list)
    neighbors:               list[NeighborEvidence] = Field(default_factory=list)


# ─── LLM output ───────────────────────────────────────────────────────────────
# Structured output produced by the local LLM.
# topic and urgency must be valid assignment enum values.

class LLMAnalysis(BaseModel):
    topic:          Topic
    urgency:        Urgency
    missing_info:   bool
    missing_fields: list[str] = Field(default_factory=list)
    confidence:     float = Field(ge=0.0, le=1.0)
    short_note:     str


# ─── Validation ───────────────────────────────────────────────────────────────

class ValidationResult(BaseModel):
    is_valid:             bool
    requires_human_review: bool = False
    flags:                list[str] = Field(default_factory=list)
    notes:                list[str] = Field(default_factory=list)


# ─── Action execution ─────────────────────────────────────────────────────────

class ActionExecutionResult(BaseModel):
    selected_action: NextAction
    action_status:   str
    action_note:     str
    target:          str | None = None


# ─── Final triage result ──────────────────────────────────────────────────────
# The complete output written to CSV/JSON for the assignment.

class TriageResult(BaseModel):
    ticket_id:             str
    text_snippet:          str
    topic:                 Topic
    urgency:               Urgency
    next_action:           NextAction
    confidence:            float
    missing_info:          bool
    missing_fields:        list[str] = Field(default_factory=list)
    requires_human_review: bool
    short_note:            str
    action_result:         ActionExecutionResult | None = None


