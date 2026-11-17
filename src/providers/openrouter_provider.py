"""OpenRouter provider — unified access to 200+ models via OpenAI-compatible API.

Free models use the :free suffix. Paid models bill through your OpenRouter account.
One API key covers every model in the registry.
"""

from __future__ import annotations

import os
import time

import httpx

from src.models.schemas import ChatMessage, LLMResponse, ModelConfig, Provider
from src.providers.base import LLMProvider


class OpenRouterProvider(LLMProvider):
    def __init__(self):
        self._api_key = os.environ.get("OPEN_ROUTER_API_KEY", os.environ.get("OPENROUTER_API_KEY", ""))
        self._base_url = "https://openrouter.ai/api/v1/"
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/llm-cost-autopilot",
                    "X-Title": "LLM Cost Autopilot",
                },
                timeout=120.0,
            )
        return self._client

    async def complete(
        self,
        messages: list[ChatMessage],
        model_config: ModelConfig,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        client = await self._get_client()
        payload: dict = {
            "model": model_config.model_id,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        start = time.perf_counter()
        resp = await client.post("chat/completions", json=payload)
        latency_ms = (time.perf_counter() - start) * 1000
        resp.raise_for_status()
        data = resp.json()

        usage = data.get("usage", {})
        text = data["choices"][0]["message"]["content"] if data.get("choices") else ""
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        return LLMResponse(
            text=text,
            model_name=model_config.name,
            provider=Provider.OPENROUTER,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost=self._compute_cost(model_config, input_tokens, output_tokens),
            tier=model_config.tier,
        )
