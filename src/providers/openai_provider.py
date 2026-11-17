from __future__ import annotations

import os
import time

import httpx

from src.models.schemas import ChatMessage, LLMResponse, ModelConfig, Provider
from src.providers.base import LLMProvider


class OpenAIProvider(LLMProvider):
    def __init__(self):
        self._api_key = os.environ.get("OPENAI_API_KEY", "")
        self._base_url = "https://api.openai.com/v1"
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
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
        payload = {
            "model": model_config.model_id,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        start = time.perf_counter()
        resp = await client.post("/chat/completions", json=payload)
        latency_ms = (time.perf_counter() - start) * 1000
        resp.raise_for_status()
        data = resp.json()

        usage = data["usage"]
        text = data["choices"][0]["message"]["content"]
        input_tokens = usage["prompt_tokens"]
        output_tokens = usage["completion_tokens"]

        return LLMResponse(
            text=text,
            model_name=model_config.name,
            provider=Provider.OPENAI,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost=self._compute_cost(model_config, input_tokens, output_tokens),
            tier=model_config.tier,
        )
