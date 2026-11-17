"""Routing engine: decides which model handles a request.

Supports cascade mode: try cheap first, escalate on low-confidence signals.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import yaml

from src.models.registry import ModelRegistry
from src.models.schemas import (
    ChatMessage,
    LLMResponse,
    ModelConfig,
    Tier,
)
from src.providers import get_provider
from src.router.classifier import LearnedRouter

logger = logging.getLogger(__name__)

_ROUTING_CONFIG = Path(__file__).resolve().parent.parent.parent / "config" / "routing.yaml"


class RoutingEngine:
    def __init__(
        self,
        registry: ModelRegistry,
        router: LearnedRouter | None = None,
        config_path: Path = _ROUTING_CONFIG,
    ):
        self.registry = registry
        self.router = router
        self._config = yaml.safe_load(config_path.read_text())["routing"]
        self._tier_map: dict[str, str] = self._config["tier_map"]
        self._confidence_threshold: float = self._config["confidence_threshold"]
        self._cascade_enabled: bool = self._config.get("cascade", {}).get("enabled", False)
        self._escalation_signals: list[str] = self._config.get("cascade", {}).get("escalation_signals", [])

    def _model_for_tier(self, tier: Tier) -> ModelConfig:
        model_name = self._tier_map[tier.value]
        return self.registry.get(model_name)

    def _next_tier(self, tier: Tier) -> Tier | None:
        order = [Tier.LOW, Tier.MEDIUM, Tier.HIGH]
        idx = order.index(tier)
        return order[idx + 1] if idx < len(order) - 1 else None

    def classify(self, prompt: str) -> tuple[Tier, float]:
        """Classify a prompt into a tier with confidence."""
        if self.router and self.router.is_trained:
            return self.router.predict(prompt)
        # Fallback: keyword heuristic
        from src.router.baselines import KeywordHeuristicRouter
        return KeywordHeuristicRouter().route(prompt)

    async def route_request(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Route a request to the cheapest capable model."""
        prompt_text = " ".join(m.content for m in messages if m.role == "user")

        tier, confidence = self.classify(prompt_text)
        logger.info("Classified as tier=%s confidence=%.3f", tier.value, confidence)

        # If confidence is below threshold, bump up one tier
        if confidence < self._confidence_threshold:
            next_tier = self._next_tier(tier)
            if next_tier:
                logger.info("Low confidence (%.3f < %.3f), escalating %s -> %s",
                            confidence, self._confidence_threshold, tier.value, next_tier.value)
                tier = next_tier

        model_config = self._model_for_tier(tier)
        provider = get_provider(model_config.provider)

        response = await provider.complete(messages, model_config, temperature, max_tokens)
        response.routed_by = "learned_router" if (self.router and self.router.is_trained) else "heuristic"
        response.tier = tier

        # Cascade: check for escalation signals in the response
        if self._cascade_enabled:
            response = await self._maybe_escalate(response, messages, temperature, max_tokens)

        return response

    async def _maybe_escalate(
        self,
        response: LLMResponse,
        messages: list[ChatMessage],
        temperature: float,
        max_tokens: int | None,
    ) -> LLMResponse:
        """Check response for signals that require escalation."""
        reasons = []

        if "json_parse_failure" in self._escalation_signals:
            if self._looks_like_json_request(messages) and not self._is_valid_json(response.text):
                reasons.append("json_parse_failure")

        if "empty_response" in self._escalation_signals:
            if not response.text.strip():
                reasons.append("empty_response")

        if "refusal_detected" in self._escalation_signals:
            refusal_phrases = ["i cannot", "i'm sorry", "i can't", "as an ai"]
            if any(p in response.text.lower()[:200] for p in refusal_phrases):
                reasons.append("refusal_detected")

        if not reasons:
            return response

        next_tier = self._next_tier(response.tier)
        if not next_tier:
            return response

        logger.info("Escalating due to: %s", reasons)
        model_config = self._model_for_tier(next_tier)
        provider = get_provider(model_config.provider)

        escalated = await provider.complete(messages, model_config, temperature, max_tokens)
        escalated.routed_by = response.routed_by
        escalated.escalated = True
        escalated.escalation_reason = ", ".join(reasons)
        escalated.tier = next_tier
        return escalated

    @staticmethod
    def _looks_like_json_request(messages: list[ChatMessage]) -> bool:
        text = " ".join(m.content.lower() for m in messages)
        return "json" in text or "schema" in text or "structured" in text

    @staticmethod
    def _is_valid_json(text: str) -> bool:
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        try:
            json.loads(text.strip())
            return True
        except (json.JSONDecodeError, ValueError):
            return False
