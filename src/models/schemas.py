from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Tier(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    OPENROUTER = "openrouter"


class ModelConfig(BaseModel):
    name: str
    provider: Provider
    model_id: str
    tier: Tier
    cost_per_input_token: float
    cost_per_output_token: float
    cost_per_cached_token: float = 0.0
    max_context: int
    max_output: int
    supports_streaming: bool = True


# ── OpenAI-compatible chat completion request ──


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    messages: list[ChatMessage]
    model: str | None = None
    temperature: float = 0.7
    max_tokens: int | None = None
    stream: bool = False


# ── Unified LLM response ──


class LLMResponse(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    model_name: str
    provider: Provider
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost: float
    tier: Tier
    routed_by: str = "manual"
    escalated: bool = False
    escalation_reason: str | None = None
    timestamp: float = Field(default_factory=time.time)


# ── OpenAI-compatible chat completion response ──


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class RoutingMetadata(BaseModel):
    routed_model: str
    tier: Tier
    confidence: float | None = None
    routed_by: str
    escalated: bool = False
    cost: float
    latency_ms: float


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageInfo
    routing: RoutingMetadata


# ── Quality verification ──


class QualityScore(BaseModel):
    request_id: str
    rubric_score: float
    judge_model: str
    reference_model: str | None = None
    agreement_score: float | None = None
    is_pass: bool
    failure_reason: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


# ── Router training data ──


class RouterTrainingSample(BaseModel):
    prompt_text: str
    prompt_hash: str
    true_tier: Tier
    predicted_tier: Tier | None = None
    source: str  # "manual", "oracle", "verification_failure", "sampled"
    features: dict[str, float] = Field(default_factory=dict)


# ── Stats ──


class CostStats(BaseModel):
    total_requests: int
    total_cost: float
    baseline_cost: float
    net_savings: float
    savings_pct: float
    verification_cost: float
    escalation_rate: float
    quality_parity_pct: float
    quality_ci_lower: float | None = None
    quality_ci_upper: float | None = None
