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
from fastapi.responses import HTMLResponse

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


@app.get("/", response_class=HTMLResponse)
async def landing():
    models = registry.list_models()
    model_rows = ""
    for m in models:
        cost_in = m.cost_per_input_token * 1_000_000
        cost_out = m.cost_per_output_token * 1_000_000
        badge_color = {"high": "#ef4444", "medium": "#f59e0b", "low": "#22c55e"}[m.tier.value]
        model_rows += f"""<tr>
            <td><code>{m.name}</code></td>
            <td><span class="badge" style="background:{badge_color}">{m.tier.value}</span></td>
            <td>{m.provider.value}</td>
            <td>{"FREE" if cost_in == 0 else f"${cost_in:.2f}"}</td>
            <td>{"FREE" if cost_out == 0 else f"${cost_out:.2f}"}</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LLM Cost Autopilot</title>
<style>
  :root {{
    --bg: #0f172a; --surface: #1e293b; --border: #334155;
    --text: #f1f5f9; --muted: #94a3b8; --accent: #38bdf8;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: var(--bg); color: var(--text); line-height: 1.6; padding: 16px; }}
  .container {{ max-width: 800px; margin: 0 auto; }}
  h1 {{ font-size: 2rem; margin: 2rem 0 0.5rem; }}
  h1 span {{ color: var(--accent); }}
  .subtitle {{ color: var(--muted); font-size: 1.1rem; margin-bottom: 2rem; }}
  .status {{ display: inline-flex; align-items: center; gap: 6px;
             background: #064e3b; color: #34d399; padding: 4px 12px;
             border-radius: 999px; font-size: 0.85rem; font-weight: 600; }}
  .status::before {{ content: ''; width: 8px; height: 8px;
                     background: #34d399; border-radius: 50%; }}
  .card {{ background: var(--surface); border: 1px solid var(--border);
           border-radius: 12px; padding: 1.5rem; margin: 1.5rem 0; }}
  h2 {{ font-size: 1.2rem; margin-bottom: 1rem; color: var(--accent); }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
  th {{ text-align: left; color: var(--muted); font-weight: 500;
       padding: 8px 12px; border-bottom: 1px solid var(--border); }}
  td {{ padding: 8px 12px; border-bottom: 1px solid var(--border); }}
  code {{ background: #0f172a; padding: 2px 8px; border-radius: 4px;
          font-size: 0.85rem; color: var(--accent); }}
  .badge {{ padding: 2px 10px; border-radius: 999px; font-size: 0.75rem;
            color: white; font-weight: 600; text-transform: uppercase; }}
  .endpoint {{ display: flex; align-items: center; gap: 10px;
               padding: 10px 0; border-bottom: 1px solid var(--border); }}
  .endpoint:last-child {{ border-bottom: none; }}
  .method {{ font-size: 0.75rem; font-weight: 700; padding: 3px 8px;
             border-radius: 4px; min-width: 50px; text-align: center; }}
  .get {{ background: #064e3b; color: #34d399; }}
  .post {{ background: #422006; color: #fbbf24; }}
  .path {{ font-family: monospace; color: var(--text); }}
  .desc {{ color: var(--muted); font-size: 0.85rem; margin-left: auto; }}
  .example {{ background: #0f172a; border-radius: 8px; padding: 1rem;
              overflow-x: auto; font-size: 0.82rem; line-height: 1.5; }}
  .example code {{ background: none; padding: 0; }}
  a {{ color: var(--accent); }}
  footer {{ text-align: center; color: var(--muted); font-size: 0.8rem;
            margin-top: 3rem; padding: 1rem; }}
</style>
</head>
<body>
<div class="container">
  <h1><span>LLM Cost Autopilot</span></h1>
  <p class="subtitle">Intelligent routing layer that sends each request to the cheapest capable model.</p>
  <span class="status">Operational</span>

  <div class="card">
    <h2>API Endpoints</h2>
    <div class="endpoint">
      <span class="method post">POST</span>
      <span class="path">/v1/chat/completions</span>
      <span class="desc">OpenAI-compatible chat</span>
    </div>
    <div class="endpoint">
      <span class="method get">GET</span>
      <span class="path">/v1/models</span>
      <span class="desc">List models &amp; pricing</span>
    </div>
    <div class="endpoint">
      <span class="method get">GET</span>
      <span class="path">/v1/stats</span>
      <span class="desc">Cost savings summary</span>
    </div>
    <div class="endpoint">
      <span class="method get">GET</span>
      <span class="path">/v1/stats/detailed</span>
      <span class="desc">Full stats breakdown</span>
    </div>
    <div class="endpoint">
      <span class="method post">POST</span>
      <span class="path">/v1/router/retrain</span>
      <span class="desc">Retrain the router</span>
    </div>
    <div class="endpoint">
      <span class="method get">GET</span>
      <span class="path">/health</span>
      <span class="desc">Health check</span>
    </div>
  </div>

  <div class="card">
    <h2>Available Models</h2>
    <table>
      <thead><tr><th>Model</th><th>Tier</th><th>Provider</th><th>Input $/1M</th><th>Output $/1M</th></tr></thead>
      <tbody>{model_rows}</tbody>
    </table>
  </div>

  <div class="card">
    <h2>Quick Start</h2>
    <div class="example"><code>curl -X POST {os.environ.get('VERCEL_URL', 'https://llm-cost-autopilot-nine.vercel.app')}/v1/chat/completions \\<br>
&nbsp;&nbsp;-H "Content-Type: application/json" \\<br>
&nbsp;&nbsp;-d '{{"messages": [{{"role": "user", "content": "Hello!"}}]}}'</code></div>
    <p style="color:var(--muted); font-size:0.85rem; margin-top:0.75rem;">
      Drop-in replacement for OpenAI — the router picks the cheapest model that can handle your request.
    </p>
  </div>

  <div class="card">
    <h2>How It Works</h2>
    <table>
      <tr><td style="font-size:1.3rem">1.</td><td><strong>Classify</strong> — TF-IDF + logistic regression scores prompt complexity</td></tr>
      <tr><td style="font-size:1.3rem">2.</td><td><strong>Route</strong> — Maps to the cheapest tier (free &rarr; medium &rarr; high)</td></tr>
      <tr><td style="font-size:1.3rem">3.</td><td><strong>Cascade</strong> — Auto-escalates on bad signals (empty, refusal, JSON fail)</td></tr>
      <tr><td style="font-size:1.3rem">4.</td><td><strong>Verify</strong> — 5% sampled LLM-as-judge quality checks</td></tr>
      <tr><td style="font-size:1.3rem">5.</td><td><strong>Learn</strong> — Champion/challenger retraining on all traffic</td></tr>
    </table>
  </div>

  <footer>
    LLM Cost Autopilot &middot; <a href="https://github.com/ramneek3/llm-cost-autopilot">GitHub</a>
  </footer>
</div>
</body>
</html>"""
