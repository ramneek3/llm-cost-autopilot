"""Async quality verification loop.

Samples a fraction of traffic for quality verification. Uses:
1. Rubric scoring from a cross-family judge
2. Pairwise comparison against the reference (strong) model
3. Task-specific checks (JSON validity, extraction completeness, etc.)

Reports savings NET of verification overhead.
"""

from __future__ import annotations

import logging
import random

from src.logging.db import log_quality_score
from src.models.registry import ModelRegistry
from src.models.schemas import ChatMessage, LLMResponse, QualityScore
from src.providers import get_provider
from src.quality.checks import check_json_validity, check_non_empty
from src.quality.judge import QualityJudge

logger = logging.getLogger(__name__)


class QualityVerifier:
    def __init__(
        self,
        registry: ModelRegistry,
        judge: QualityJudge,
        sample_rate: float = 0.05,
        reference_model: str = "claude-sonnet-5-5",
    ):
        self.registry = registry
        self.judge = judge
        self.sample_rate = sample_rate
        self.reference_model = reference_model
        self._verification_cost: float = 0.0

    @property
    def total_verification_cost(self) -> float:
        return self._verification_cost

    def should_verify(self, response: LLMResponse) -> bool:
        """Decide whether to verify this request (sampling)."""
        if response.model_name == self.reference_model:
            return False
        return random.random() < self.sample_rate

    async def verify(
        self,
        response: LLMResponse,
        messages: list[ChatMessage],
    ) -> QualityScore | None:
        """Run async quality verification on a response."""
        if not self.should_verify(response):
            return None

        prompt_text = " ".join(m.content for m in messages if m.role == "user")

        # 1. Cheap checks first
        basic = check_non_empty(response.text)
        if not basic.passed:
            score = QualityScore(
                request_id=response.request_id,
                rubric_score=basic.score,
                judge_model="basic_check",
                is_pass=False,
                failure_reason=basic.reason,
            )
            log_quality_score(score)
            return score

        # 2. Rubric score from the judge
        rubric_score = await self.judge.rubric_score(
            response.request_id, prompt_text, response.text,
        )

        # 3. Pairwise comparison against reference model
        ref_config = self.registry.get(self.reference_model)
        ref_provider = get_provider(ref_config.provider)
        try:
            ref_response = await ref_provider.complete(messages, ref_config, temperature=0.0)
            self._verification_cost += ref_response.cost

            pairwise = await self.judge.pairwise_compare(
                response.request_id,
                prompt_text,
                response.text, response.model_name,
                ref_response.text, self.reference_model,
            )
            # Merge scores: use the lower of rubric and pairwise
            final_score = min(rubric_score.rubric_score, pairwise.rubric_score)
            combined = QualityScore(
                request_id=response.request_id,
                rubric_score=final_score,
                judge_model=self.judge.judge_model_name,
                reference_model=self.reference_model,
                agreement_score=pairwise.agreement_score,
                is_pass=rubric_score.is_pass and pairwise.is_pass,
                failure_reason=rubric_score.failure_reason or pairwise.failure_reason,
                details={
                    "rubric": rubric_score.details,
                    "pairwise": pairwise.details,
                },
            )
        except Exception as e:
            logger.error("Reference model comparison failed: %s", e)
            combined = rubric_score

        log_quality_score(combined)
        return combined
