# LLM Cost Autopilot

An intelligent routing layer that sits in front of multiple LLM providers, classifies each request's complexity, routes it to the cheapest model capable of handling it at acceptable quality, and continuously validates routing decisions through sampled verification. Net savings are reported after accounting for verification overhead.

## Architecture

```
Client Request
       │
       ▼
┌─────────────────────┐
│  FastAPI Gateway     │  OpenAI-compatible /v1/chat/completions
│  (src/main.py)       │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Routing Engine      │  TF-IDF + logistic regression classifier
│  (src/router/)       │  with cascade escalation on cheap signals
└─────────┬───────────┘
          │
    ┌─────┼─────┐
    ▼     ▼     ▼
 Gemma   4o-  Sonnet   ← All via OpenRouter (single API key)
 :free   mini   4
 (low)  (med) (high)
          │
          ▼
┌─────────────────────┐
│  Quality Verifier    │  5% sample → rubric judge + pairwise comparison
│  (src/quality/)      │  Cross-family judge to reduce bias
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Feedback Loop       │  Champion/challenger promotion
│  (src/feedback/)     │  Trains on ALL traffic, not just failures
└─────────────────────┘
```

## Setup

```bash
# 1. Clone and install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure your OpenRouter API key (one key covers all models)
#    Get a free key at https://openrouter.ai/keys
cp .env.example .env
# Edit .env with your OPENROUTER_API_KEY

# 3. Initialize the database
python -c "from src.logging.db import init_db; init_db()"

# 4. Start the API server
uvicorn src.main:app --reload --port 8000
```

### With Docker

```bash
cp .env.example .env
# Edit .env with your API keys
docker-compose up --build
```

API: `http://localhost:8000` | Dashboard: `http://localhost:8501`

## Usage

### Send a request (OpenAI-compatible)

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is the capital of France?"}]
  }'
```

The response includes routing metadata showing which model was selected and why:

```json
{
  "routing": {
    "routed_model": "claude-haiku-5-5",
    "tier": "low",
    "routed_by": "learned_router",
    "escalated": false,
    "cost": 0.000012
  }
}
```

### Check savings

```bash
curl http://localhost:8000/v1/stats
```

### View available models

```bash
curl http://localhost:8000/v1/models
```

## Building the Oracle Table (Day 1)

Before training the router, build ground truth by evaluating benchmark prompts against all models:

```bash
# Run with all models (costs a few dollars)
python -m scripts.build_oracle

# Or specify models
python -m scripts.build_oracle --models claude-haiku-5-5,gpt-4o-mini,claude-sonnet-5-5
```

This produces `data/oracle_table.json` — the quality, cost, and latency per (prompt, model) pair.

## Training the Router (Days 4-6)

Train the learned router from oracle table data:

```bash
python -m scripts.train_router --oracle-path data/oracle_table.json
```

This outputs:
- Learned router accuracy vs. baselines (always-big, always-small, random, keyword heuristic)
- Per-class precision/recall/F1
- Saved model at `models/router_latest.joblib`

The server picks up the trained model on next restart.

## Adding New Test Cases

Add prompts to `benchmark/dataset.py` in the `_builtin_dataset()` function, or create a JSON file:

```json
[
  {
    "id": "custom-001",
    "text": "Your prompt here",
    "task_type": "extraction",
    "expected_tier": "low",
    "quality_check": {"type": "exact_fields", "fields": ["expected", "values"]}
  }
]
```

Quality check types:
- `exact_fields`: checks that all listed strings appear in the response
- `valid_json`: checks that the response is parseable JSON
- `classification`: checks that one of the valid labels appears
- `min_words`: checks that the response has at least N words

## Adjusting Thresholds

All routing and verification config is in `config/routing.yaml`:

- `confidence_threshold`: below this, the router escalates to the next tier (default: 0.75)
- `sample_rate`: fraction of requests verified by the judge (default: 0.05)
- `rubric_score_threshold`: quality score below this counts as a routing failure (default: 3/5)
- `promotion_threshold`: new router must beat current by this margin to go live (default: 0.02)

Model pricing and tier assignments are in `config/models.yaml`. Update prices here when providers change rates.

## Load Testing

```bash
# Start the server, then:
python -m scripts.load_test --url http://localhost:8000 --count 100 --concurrency 10
```

Generates a report at `data/load_test_report.json` with cost savings, model distribution, and escalation rates.

## Design Decisions

**Why OpenRouter?** Single API key, unified billing, access to free models for the low tier and paid models from multiple providers for higher tiers. Free models (`:free` suffix) handle simple extraction/formatting at zero cost. Swap to direct provider keys by uncommenting models in `config/models.yaml`.

**Why TF-IDF + logistic regression instead of embeddings?** The router needs to be fast (< 5ms per classification) and cheap (no API call). TF-IDF captures enough signal for routing decisions. Embedding-based routing is a viable upgrade path but adds latency and cost per request.

**Why sample 5% instead of verifying everything?** The original plan verified 100% of requests, which can cost more than the savings. At 5%, verification overhead is bounded while still catching systematic routing failures.

**Why cross-family judging?** Using GPT-4o-mini to judge responses from Claude/Gemma reduces the bias that comes from a model family rating its own outputs higher.

**Why champion/challenger?** Retraining on accumulated data and promoting only if the new model beats the old one on a holdout set prevents regression. Training only on failures (the original plan) would bias the router toward over-escalation.

**Why report net savings?** The headline "saved X%" is meaningless if verification costs eat the savings. All stats account for verification and escalation overhead.

## Free tier limits

OpenRouter free models are rate-limited: 20 requests/minute, 50 requests/day (increases to 1,000/day after a one-time $10 credit purchase). Free models can rotate — if one goes offline, update the `model_id` in `config/models.yaml`. The `:free` suffix is what makes a model free on OpenRouter.
