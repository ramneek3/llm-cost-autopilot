from __future__ import annotations

import os
import time

import httpx

from src.models.schemas import ChatMessage, LLMResponse, ModelConfig, Provider
from src.providers.base import LLMProvider


class AnthropicProvider(LLMProvider):
    def __init__(self):
        self._api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        self._base_url = "https://api.anthropic.com/v1"
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers={
                    "x-api-key": self._api_key,
                    "anthropic-version": "2023-06-01",
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

        # Anthropic separates system from user/assistant messages
        system_text = ""
        api_messages = []
        for m in messages:
            if m.role == "system":
                system_text += m.content + "\n"
            else:
                api_messages.append({"role": m.role, "content": m.content})

        if not api_messages:
            api_messages = [{"role": "user", "content": "Hello"}]

        payload: dict = {
            "model": model_config.model_id,
            "messages": api_messages,
            "max_tokens": max_tokens or 4096,
            "temperature": temperature,
        }
        if system_text.strip():
            payload["system"] = system_text.strip()

        start = time.perf_counter()
        resp = await client.post("/messages", json=payload)
        latency_ms = (time.perf_counter() - start) * 1000
        resp.raise_for_status()
        data = resp.json()

        text = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                text += block["text"]

        usage = data.get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)

        return LLMResponse(
            text=text,
            model_name=model_config.name,
            provider=Provider.ANTHROPIC,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost=self._compute_cost(model_config, input_tokens, output_tokens),
            tier=model_config.tier,
        )
