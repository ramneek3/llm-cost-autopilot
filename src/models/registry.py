from __future__ import annotations

from pathlib import Path

import yaml

from src.models.schemas import ModelConfig, Provider, Tier

_DEFAULT_CONFIG = Path(__file__).resolve().parent.parent.parent / "config" / "models.yaml"


class ModelRegistry:
    def __init__(self, config_path: Path | str = _DEFAULT_CONFIG):
        self._models: dict[str, ModelConfig] = {}
        self._load(Path(config_path))

    def _load(self, path: Path) -> None:
        raw = yaml.safe_load(path.read_text())
        for name, cfg in raw["models"].items():
            self._models[name] = ModelConfig(
                name=name,
                provider=Provider(cfg["provider"]),
                model_id=cfg["model_id"],
                tier=Tier(cfg["tier"]),
                cost_per_input_token=cfg["cost_per_input_token"],
                cost_per_output_token=cfg["cost_per_output_token"],
                cost_per_cached_token=cfg.get("cost_per_cached_token", 0.0),
                max_context=cfg["max_context"],
                max_output=cfg["max_output"],
                supports_streaming=cfg.get("supports_streaming", True),
            )

    def get(self, name: str) -> ModelConfig:
        return self._models[name]

    def list_models(self) -> list[ModelConfig]:
        return list(self._models.values())

    def by_tier(self, tier: Tier) -> list[ModelConfig]:
        return [m for m in self._models.values() if m.tier == tier]

    def cheapest_in_tier(self, tier: Tier) -> ModelConfig:
        candidates = self.by_tier(tier)
        if not candidates:
            raise ValueError(f"No models in tier {tier}")
        return min(candidates, key=lambda m: m.cost_per_output_token)

    def most_expensive(self) -> ModelConfig:
        return max(self._models.values(), key=lambda m: m.cost_per_output_token)

    def compute_cost(self, model_name: str, input_tokens: int, output_tokens: int) -> float:
        m = self.get(model_name)
        return (input_tokens * m.cost_per_input_token) + (output_tokens * m.cost_per_output_token)
