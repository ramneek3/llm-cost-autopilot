from __future__ import annotations

import os
import time

import httpx

from src.models.schemas import ChatMessage, LLMResponse, ModelConfig, Provider
from src.providers.base import LLMProvider


class OllamaProvider(LLMProvider):
    def __init__(self):
        self._base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=180.0,
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
            "stream": False,
            "options": {"temperature": temperature},
        }
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        start = time.perf_counter()
        resp = await client.post("/api/chat", json=payload)
        latency_ms = (time.perf_counter() - start) * 1000
        resp.raise_for_status()
        data = resp.json()

        text = data.get("message", {}).get("content", "")
        input_tokens = data.get("prompt_eval_count", 0)
        output_tokens = data.get("eval_count", 0)

        return LLMResponse(
            text=text,
            model_name=model_config.name,
            provider=Provider.OLLAMA,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost=0.0,
            tier=model_config.tier,
        )
