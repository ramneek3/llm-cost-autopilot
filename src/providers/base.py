from __future__ import annotations

import abc
import time

from src.models.schemas import ChatMessage, LLMResponse, ModelConfig


class LLMProvider(abc.ABC):
    @abc.abstractmethod
    async def complete(
        self,
        messages: list[ChatMessage],
        model_config: ModelConfig,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        ...

    def _compute_cost(self, cfg: ModelConfig, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * cfg.cost_per_input_token) + (output_tokens * cfg.cost_per_output_token)
