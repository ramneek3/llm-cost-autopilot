"""LLM Cost Autopilot — FastAPI gateway.

OpenAI-compatible /v1/chat/completions endpoint that transparently routes
requests to the cheapest capable model.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.feedback.loop import retrain_router
from src.logging.db import get_stats, init_db, log_request
from src.models.registry import ModelRegistry
from src.models.schemas import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    CostStats,
    RoutingMetadata,
    UsageInfo,
)
from src.quality.judge import QualityJudge
from src.quality.verifier import QualityVerifier
from src.router.classifier import LearnedRouter
from src.router.engine import RoutingEngine

load_dotenv()

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

registry: ModelRegistry
engine: RoutingEngine
verifier: QualityVerifier


@asynccontextmanager
async def lifespan(app: FastAPI):
    global registry, engine, verifier

    init_db()
    registry = ModelRegistry()

    router_path = Path("models/router_latest.joblib")
    learned_router = LearnedRouter(router_path) if router_path.exists() else LearnedRouter()

    engine = RoutingEngine(registry, learned_router)

    sample_rate = float(os.environ.get("VERIFICATION_SAMPLE_RATE", "0.05"))
    judge_model = os.environ.get("JUDGE_MODEL", "gpt-4o-mini")
    judge = QualityJudge(registry, judge_model_name=judge_model)
    verifier = QualityVerifier(
        registry, judge, sample_rate=sample_rate,
        reference_model=os.environ.get("REFERENCE_MODEL", "claude-sonnet-5.5"),
    )

    logger.info("LLM Cost Autopilot started (sample_rate=%.2f)", sample_rate)
    yield


app = FastAPI(
    title="LLM Cost Autopilot",
    description="Intelligent LLM routing layer that minimizes cost while maintaining quality",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── OpenAI-compatible chat completion endpoint ──


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    """Route a chat completion to the cheapest capable model.

    Drop-in replacement for OpenAI's /v1/chat/completions.
    The router picks the model — the caller doesn't need to.
    """
    try:
        response = await engine.route_request(
            messages=request.messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
    except Exception as e:
        logger.error("Routing failed: %s", e)
        raise HTTPException(status_code=502, detail=str(e))

    prompt_text = " ".join(m.content for m in request.messages if m.role == "user")
    log_request(response, prompt_text)

    # Async quality verification (non-blocking conceptually; awaited for correctness)
    try:
        await verifier.verify(response, request.messages)
    except Exception as e:
        logger.warning("Verification failed (non-fatal): %s", e)

    return ChatCompletionResponse(
        model=response.model_name,
        choices=[
            ChatCompletionChoice(
                message=ChatMessage(role="assistant", content=response.text),
            )
        ],
        usage=UsageInfo(
            prompt_tokens=response.input_tokens,
            completion_tokens=response.output_tokens,
            total_tokens=response.input_tokens + response.output_tokens,
        ),
        routing=RoutingMetadata(
            routed_model=response.model_name,
            tier=response.tier,
            confidence=None,
            routed_by=response.routed_by,
            escalated=response.escalated,
            cost=response.cost,
            latency_ms=response.latency_ms,
        ),
    )


# ── Management endpoints ──


@app.get("/v1/models")
async def list_models():
    """List available models and their costs."""
    models = registry.list_models()
    return {
        "data": [
            {
                "id": m.name,
                "provider": m.provider.value,
                "tier": m.tier.value,
                "cost_per_1m_input_tokens": m.cost_per_input_token * 1_000_000,
                "cost_per_1m_output_tokens": m.cost_per_output_token * 1_000_000,
                "max_context": m.max_context,
            }
            for m in models
        ]
    }


@app.get("/v1/stats")
async def get_cost_stats():
    """Cost savings summary with net savings (accounting for verification overhead)."""
    raw = get_stats()
    total_cost = raw["total_cost"]
    verification_cost = verifier.total_verification_cost

    # Baseline: what it would have cost to send everything to the reference model
    ref_name = os.environ.get("REFERENCE_MODEL", "claude-sonnet-5.5")
    ref = registry.get(ref_name)
    baseline_cost = raw["total_tokens"] * ref.cost_per_output_token  # rough estimate

    net_cost = total_cost + verification_cost
    net_savings = baseline_cost - net_cost

    return CostStats(
        total_requests=raw["total_requests"],
        total_cost=net_cost,
        baseline_cost=baseline_cost,
        net_savings=max(0, net_savings),
        savings_pct=(net_savings / baseline_cost * 100) if baseline_cost > 0 else 0,
        verification_cost=verification_cost,
        escalation_rate=raw["escalation_rate"],
        quality_parity_pct=raw["quality"]["pass_rate"] * 100,
    )


@app.get("/v1/stats/detailed")
async def get_detailed_stats():
    """Full stats including model distribution and quality breakdown."""
    return get_stats()


@app.post("/v1/router/retrain")
async def retrain():
    """Trigger router retraining with champion/challenger promotion."""
    result = retrain_router()
    return result


@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": time.time()}
