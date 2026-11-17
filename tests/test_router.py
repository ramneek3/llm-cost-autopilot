from src.models.schemas import Tier
from src.router.baselines import (
    AlwaysBigRouter,
    AlwaysSmallRouter,
    KeywordHeuristicRouter,
    RandomRouter,
)
from src.router.classifier import LearnedRouter


def test_always_big():
    router = AlwaysBigRouter()
    tier, conf = router.route("anything")
    assert tier == Tier.HIGH
    assert conf == 1.0


def test_always_small():
    router = AlwaysSmallRouter()
    tier, conf = router.route("anything")
    assert tier == Tier.LOW


def test_keyword_heuristic_simple():
    router = KeywordHeuristicRouter()
    tier, _ = router.route("What is 2+2?")
    assert tier == Tier.LOW


def test_keyword_heuristic_complex():
    router = KeywordHeuristicRouter()
    tier, _ = router.route(
        "Analyze and compare the trade-offs between these two approaches. "
        "Evaluate the pros and cons of each design decision."
    )
    assert tier == Tier.HIGH


def test_learned_router_train():
    prompts = [
        "What is the capital of France?",
        "Extract the email from: john@example.com",
        "Convert 5km to miles",
        "Summarize this article about climate change",
        "Classify this review as positive or negative",
        "Explain how DNS works",
        "Analyze the trade-offs between SQL and NoSQL for this use case",
        "Design a system architecture for a real-time chat application",
        "Compare three sorting algorithms and evaluate their time complexity",
    ] * 5  # Repeat to get enough samples

    labels = [
        Tier.LOW, Tier.LOW, Tier.LOW,
        Tier.MEDIUM, Tier.MEDIUM, Tier.MEDIUM,
        Tier.HIGH, Tier.HIGH, Tier.HIGH,
    ] * 5

    router = LearnedRouter()
    metrics = router.train(prompts, labels, test_size=0.2)
    assert metrics["train_accuracy"] > 0.5
    assert router.is_trained

    tier, confidence = router.predict("What is 2+2?")
    assert isinstance(tier, Tier)
    assert 0 <= confidence <= 1
