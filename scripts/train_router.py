"""Train the router from oracle table data.

Usage:
    python -m scripts.train_router [--oracle-path data/oracle_table.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.schemas import Tier
from src.router.classifier import LearnedRouter
from src.router.baselines import AlwaysBigRouter, AlwaysSmallRouter, KeywordHeuristicRouter, RandomRouter


def main():
    parser = argparse.ArgumentParser(description="Train the router from oracle table data")
    parser.add_argument("--oracle-path", type=Path, default=Path("data/oracle_table.json"))
    args = parser.parse_args()

    if not args.oracle_path.exists():
        print(f"Oracle table not found at {args.oracle_path}")
        print("Run the oracle builder first: python -m benchmark.oracle")
        sys.exit(1)

    data = json.loads(args.oracle_path.read_text())
    prompts = [row["prompt_text"] for row in data]
    labels = [Tier(row["oracle_tier"]) for row in data]

    print(f"Training on {len(prompts)} prompts")
    print(f"Label distribution: {dict(zip(*map(list, zip(*[(t.value, labels.count(t)) for t in Tier]))))}")

    # ── Train learned router ──
    router = LearnedRouter()
    metrics = router.train(prompts, labels)
    model_path = router.save()

    print(f"\nLearned Router:")
    print(f"  Train accuracy: {metrics['train_accuracy']:.3f}")
    print(f"  Test accuracy:  {metrics['test_accuracy']:.3f}")
    print(f"  Saved to: {model_path}")

    # ── Compare baselines ──
    print(f"\nBaseline comparison:")
    tier_to_int = {"low": 0, "medium": 1, "high": 2}

    for name, baseline in [
        ("Always-Big", AlwaysBigRouter()),
        ("Always-Small", AlwaysSmallRouter()),
        ("Random (50/30/20)", RandomRouter()),
        ("Keyword Heuristic", KeywordHeuristicRouter()),
    ]:
        correct = sum(
            1 for p, l in zip(prompts, labels)
            if baseline.route(p)[0] == l
        )
        print(f"  {name}: {correct}/{len(prompts)} ({correct/len(prompts):.3f})")

    # ── Report ──
    report = metrics["classification_report"]
    print(f"\nPer-class metrics:")
    for cls in ["low", "medium", "high"]:
        if cls in report:
            r = report[cls]
            print(f"  {cls}: precision={r['precision']:.3f} recall={r['recall']:.3f} f1={r['f1-score']:.3f}")


if __name__ == "__main__":
    main()
