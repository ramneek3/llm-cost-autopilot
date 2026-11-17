from __future__ import annotations

from src.models.schemas import Provider
from src.providers.anthropic_provider import AnthropicProvider
from src.providers.base import LLMProvider
from src.providers.ollama_provider import OllamaProvider
from src.providers.openai_provider import OpenAIProvider
from src.providers.openrouter_provider import OpenRouterProvider

_cache: dict[Provider, LLMProvider] = {}


def get_provider(provider: Provider) -> LLMProvider:
    if provider not in _cache:
        match provider:
            case Provider.OPENAI:
                _cache[provider] = OpenAIProvider()
            case Provider.ANTHROPIC:
                _cache[provider] = AnthropicProvider()
            case Provider.OLLAMA:
                _cache[provider] = OllamaProvider()
            case Provider.OPENROUTER:
                _cache[provider] = OpenRouterProvider()
    return _cache[provider]
