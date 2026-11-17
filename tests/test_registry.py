from src.models.registry import ModelRegistry
from src.models.schemas import Tier


def test_load_models():
    registry = ModelRegistry()
    models = registry.list_models()
    assert len(models) >= 4


def test_get_model():
    registry = ModelRegistry()
    m = registry.get("gpt-4o-mini")
    assert m.tier == Tier.MEDIUM
    assert m.provider.value == "openrouter"
    assert m.cost_per_output_token > 0


def test_get_free_model():
    registry = ModelRegistry()
    m = registry.get("gemma-free")
    assert m.tier == Tier.LOW
    assert m.cost_per_input_token == 0.0
    assert m.cost_per_output_token == 0.0
    assert ":free" in m.model_id


def test_cheapest_in_tier():
    registry = ModelRegistry()
    cheapest = registry.cheapest_in_tier(Tier.LOW)
    assert cheapest.tier == Tier.LOW
    assert cheapest.cost_per_output_token == 0.0


def test_compute_cost():
    registry = ModelRegistry()
    cost = registry.compute_cost("gpt-4o-mini", input_tokens=1000, output_tokens=500)
    expected = 1000 * 0.00000015 + 500 * 0.0000006
    assert abs(cost - expected) < 1e-10


def test_compute_cost_free():
    registry = ModelRegistry()
    cost = registry.compute_cost("gemma-free", input_tokens=1000, output_tokens=500)
    assert cost == 0.0
