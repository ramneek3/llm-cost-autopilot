"""Baseline routing strategies for comparison against the learned router."""

from __future__ import annotations

import abc
import random
import re

from src.models.schemas import Tier


class BaselineRouter(abc.ABC):
    @abc.abstractmethod
    def route(self, prompt: str) -> tuple[Tier, float]:
        """Returns (tier, confidence)."""
        ...


class AlwaysBigRouter(BaselineRouter):
    """Always routes to the highest tier. Maximum quality, maximum cost."""

    def route(self, prompt: str) -> tuple[Tier, float]:
        return Tier.HIGH, 1.0


class AlwaysSmallRouter(BaselineRouter):
    """Always routes to the lowest tier. Minimum cost, variable quality."""

    def route(self, prompt: str) -> tuple[Tier, float]:
        return Tier.LOW, 1.0


class RandomRouter(BaselineRouter):
    """Routes randomly at a given cost-equivalent distribution."""

    def __init__(self, low_pct: float = 0.5, med_pct: float = 0.3, high_pct: float = 0.2):
        self._weights = [low_pct, med_pct, high_pct]
        self._tiers = [Tier.LOW, Tier.MEDIUM, Tier.HIGH]

    def route(self, prompt: str) -> tuple[Tier, float]:
        tier = random.choices(self._tiers, weights=self._weights, k=1)[0]
        return tier, 0.5


class KeywordHeuristicRouter(BaselineRouter):
    """Simple keyword-based heuristic. The straw-man to beat."""

    _HIGH_SIGNALS = re.compile(
        r"\b(analyze|compare|contrast|evaluate|critique|debate|"
        r"synthesize|design|architect|create a plan|multi-step|"
        r"pros and cons|trade-?offs|nuanced|reasoning)\b",
        re.IGNORECASE,
    )
    _MEDIUM_SIGNALS = re.compile(
        r"\b(summarize|classify|categorize|explain|describe|"
        r"list the|outline|convert|translate|rewrite)\b",
        re.IGNORECASE,
    )

    def route(self, prompt: str) -> tuple[Tier, float]:
        high_hits = len(self._HIGH_SIGNALS.findall(prompt))
        med_hits = len(self._MEDIUM_SIGNALS.findall(prompt))
        token_count = len(prompt.split())

        if high_hits >= 2 or (high_hits >= 1 and token_count > 200):
            return Tier.HIGH, 0.6
        if med_hits >= 1 or token_count > 100:
            return Tier.MEDIUM, 0.6
        return Tier.LOW, 0.7
