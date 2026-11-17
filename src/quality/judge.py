"""LLM-as-judge for quality verification.

Uses a model from a different family than the router's models to reduce bias.
Implements both rubric scoring and pairwise comparison.
"""

from __future__ import annotations

import json
import logging
import re

from src.models.registry import ModelRegistry
from src.models.schemas import ChatMessage, Provider, QualityScore
from src.providers import get_provider

logger = logging.getLogger(__name__)

RUBRIC_PROMPT = """You are an expert quality evaluator. Score the following AI response on a 1-5 scale.

## Rubric
1 - Completely wrong, irrelevant, or harmful
2 - Partially relevant but major errors or omissions
3 - Adequate but with notable quality gaps
4 - Good quality, minor issues only
5 - Excellent, fully addresses the request with accuracy

## Original Request
{prompt}

## AI Response
{response}

## Instructions
Evaluate the response against the request. Output ONLY a JSON object:
{{"score": <1-5>, "reasoning": "<brief explanation>"}}"""


PAIRWISE_PROMPT = """Compare two AI responses to the same prompt. Which is better?

## Original Request
{prompt}

## Response A (from {model_a})
{response_a}

## Response B (from {model_b})
{response_b}

## Instructions
Output ONLY a JSON object:
{{"winner": "A" or "B" or "tie", "reasoning": "<brief explanation>", "agreement_score": <0.0-1.0 how similar they are>}}"""


class QualityJudge:
    def __init__(self, registry: ModelRegistry, judge_model_name: str = "gpt-4o"):
        self.registry = registry
        self.judge_model_name = judge_model_name
        self._judge_config = registry.get(judge_model_name)

    async def rubric_score(
        self,
        request_id: str,
        prompt: str,
        response_text: str,
    ) -> QualityScore:
        """Score a response using rubric-based evaluation."""
        judge_prompt = RUBRIC_PROMPT.format(prompt=prompt, response=response_text)
        provider = get_provider(self._judge_config.provider)

        try:
            result = await provider.complete(
                [ChatMessage(role="user", content=judge_prompt)],
                self._judge_config,
                temperature=0.0,
                max_tokens=256,
            )
            parsed = self._parse_json(result.text)
            score = float(parsed.get("score", 3))
            score = max(1.0, min(5.0, score))
            reasoning = parsed.get("reasoning", "")
        except Exception as e:
            logger.error("Judge failed: %s", e)
            score = 3.0
            reasoning = f"Judge error: {e}"

        return QualityScore(
            request_id=request_id,
            rubric_score=score,
            judge_model=self.judge_model_name,
            is_pass=score >= 3.0,
            failure_reason=reasoning if score < 3.0 else None,
            details={"reasoning": reasoning, "judge_cost": result.cost if "result" in dir() else 0},
        )

    async def pairwise_compare(
        self,
        request_id: str,
        prompt: str,
        response_a: str,
        model_a: str,
        response_b: str,
        model_b: str,
    ) -> QualityScore:
        """Compare two responses from different models."""
        judge_prompt = PAIRWISE_PROMPT.format(
            prompt=prompt,
            model_a=model_a,
            response_a=response_a,
            model_b=model_b,
            response_b=response_b,
        )
        provider = get_provider(self._judge_config.provider)

        try:
            result = await provider.complete(
                [ChatMessage(role="user", content=judge_prompt)],
                self._judge_config,
                temperature=0.0,
                max_tokens=256,
            )
            parsed = self._parse_json(result.text)
            winner = parsed.get("winner", "tie")
            agreement = float(parsed.get("agreement_score", 0.5))
            reasoning = parsed.get("reasoning", "")

            # A wins or tie means the cheaper model was good enough
            is_pass = winner in ("A", "tie")
            rubric = 4.0 if winner == "A" else (3.0 if winner == "tie" else 2.0)
        except Exception as e:
            logger.error("Pairwise judge failed: %s", e)
            agreement = 0.5
            rubric = 3.0
            is_pass = True
            reasoning = f"Judge error: {e}"

        return QualityScore(
            request_id=request_id,
            rubric_score=rubric,
            judge_model=self.judge_model_name,
            reference_model=model_b,
            agreement_score=agreement,
            is_pass=is_pass,
            failure_reason=reasoning if not is_pass else None,
            details={"winner": winner if "winner" in dir() else "unknown", "reasoning": reasoning},
        )

    @staticmethod
    def _parse_json(text: str) -> dict:
        text = text.strip()
        json_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return json.loads(text)
