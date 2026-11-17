"""Oracle table builder: run every benchmark prompt through every model once.

Produces the ground truth for router training:
- Quality score per (prompt, model) pair
- Cost and latency per (prompt, model) pair
- The "correct" tier label for each prompt (cheapest model that passes quality)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from src.models.registry import ModelRegistry
from src.models.schemas import ChatMessage, LLMResponse, Tier
from src.providers import get_provider
from src.quality.checks import (
    CheckResult,
    check_classification,
    check_extraction,
    check_json_validity,
    check_non_empty,
)

from benchmark.dataset import BenchmarkPrompt, load_dataset

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass
class OracleEntry:
    prompt_id: str
    model_name: str
    text: str
    quality_score: float
    quality_passed: bool
    quality_reason: str
    input_tokens: int
    output_tokens: int
    cost: float
    latency_ms: float


@dataclass
class OracleRow:
    prompt_id: str
    prompt_text: str
    task_type: str
    expected_tier: str
    oracle_tier: str  # cheapest model that passes
    entries: dict[str, OracleEntry]


def _run_quality_check(prompt: BenchmarkPrompt, response_text: str) -> CheckResult:
    """Run the task-specific quality check for a benchmark prompt."""
    check = prompt.quality_check
    check_type = check.get("type", "non_empty")

    match check_type:
        case "exact_fields":
            return check_extraction(response_text, check["fields"])
        case "valid_json":
            return check_json_validity(response_text)
        case "classification":
            return check_classification(response_text, check["valid_labels"])
        case "min_words":
            words = len(response_text.split())
            min_count = check["count"]
            if words >= min_count:
                return CheckResult(score=4.0, passed=True, reason=f"{words} words >= {min_count}")
            return CheckResult(score=2.0, passed=False, reason=f"{words} words < {min_count}")
        case _:
            return check_non_empty(response_text)


async def _eval_single(
    prompt: BenchmarkPrompt,
    model_name: str,
    registry: ModelRegistry,
) -> OracleEntry:
    """Evaluate a single prompt against a single model."""
    config = registry.get(model_name)
    provider = get_provider(config.provider)
    messages = [ChatMessage(role="user", content=prompt.text)]

    try:
        response = await provider.complete(messages, config, temperature=0.0, max_tokens=2048)
        check = _run_quality_check(prompt, response.text)
        return OracleEntry(
            prompt_id=prompt.id,
            model_name=model_name,
            text=response.text,
            quality_score=check.score,
            quality_passed=check.passed,
            quality_reason=check.reason,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost=response.cost,
            latency_ms=response.latency_ms,
        )
    except Exception as e:
        logger.error("Failed %s on %s: %s", prompt.id, model_name, e)
        return OracleEntry(
            prompt_id=prompt.id,
            model_name=model_name,
            text="",
            quality_score=0.0,
            quality_passed=False,
            quality_reason=f"Error: {e}",
            input_tokens=0,
            output_tokens=0,
            cost=0.0,
            latency_ms=0.0,
        )


def _determine_oracle_tier(entries: dict[str, OracleEntry], registry: ModelRegistry) -> str:
    """Find the cheapest model that passes quality checks."""
    tier_order = [Tier.LOW, Tier.MEDIUM, Tier.HIGH]
    for tier in tier_order:
        tier_models = registry.by_tier(tier)
        for model in sorted(tier_models, key=lambda m: m.cost_per_output_token):
            entry = entries.get(model.name)
            if entry and entry.quality_passed:
                return tier.value
    return "high"


async def build_oracle_table(
    dataset_path: Path | None = None,
    models: list[str] | None = None,
    concurrency: int = 5,
) -> list[OracleRow]:
    """Build the oracle table: evaluate every prompt on every model.

    Args:
        dataset_path: Path to benchmark prompts JSON (None = built-in)
        models: Specific model names to evaluate (None = all registered)
        concurrency: Max concurrent API calls
    """
    registry = ModelRegistry()
    prompts = load_dataset(dataset_path)
    model_names = models or [m.name for m in registry.list_models()]

    logger.info("Building oracle table: %d prompts x %d models", len(prompts), len(model_names))

    semaphore = asyncio.Semaphore(concurrency)
    oracle_rows: list[OracleRow] = []

    async def eval_with_limit(prompt: BenchmarkPrompt, model: str) -> OracleEntry:
        async with semaphore:
            return await _eval_single(prompt, model, registry)

    for i, prompt in enumerate(prompts):
        logger.info("[%d/%d] Evaluating: %s", i + 1, len(prompts), prompt.id)
        tasks = [eval_with_limit(prompt, m) for m in model_names]
        entries_list = await asyncio.gather(*tasks)
        entries = {e.model_name: e for e in entries_list}

        oracle_tier = _determine_oracle_tier(entries, registry)
        oracle_rows.append(OracleRow(
            prompt_id=prompt.id,
            prompt_text=prompt.text,
            task_type=prompt.task_type,
            expected_tier=prompt.expected_tier,
            oracle_tier=oracle_tier,
            entries=entries,
        ))

    return oracle_rows


def save_oracle_table(rows: list[OracleRow], path: Path | None = None) -> Path:
    path = path or DATA_DIR / "oracle_table.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    data = []
    for row in rows:
        data.append({
            "prompt_id": row.prompt_id,
            "prompt_text": row.prompt_text,
            "task_type": row.task_type,
            "expected_tier": row.expected_tier,
            "oracle_tier": row.oracle_tier,
            "entries": {
                name: {
                    "quality_score": e.quality_score,
                    "quality_passed": e.quality_passed,
                    "quality_reason": e.quality_reason,
                    "cost": e.cost,
                    "latency_ms": e.latency_ms,
                    "input_tokens": e.input_tokens,
                    "output_tokens": e.output_tokens,
                }
                for name, e in row.entries.items()
            },
        })

    path.write_text(json.dumps(data, indent=2))
    logger.info("Saved oracle table to %s", path)
    return path


def load_oracle_table(path: Path) -> list[dict]:
    return json.loads(path.read_text())
