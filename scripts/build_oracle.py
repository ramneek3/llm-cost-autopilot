"""Build the oracle table: evaluate every benchmark prompt on every model.

Usage:
    python -m scripts.build_oracle [--concurrency 5] [--models claude-haiku-5-5,gpt-4o-mini,claude-sonnet-5-5]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.oracle import build_oracle_table, save_oracle_table


async def run(models: list[str] | None, concurrency: int):
    rows = await build_oracle_table(models=models, concurrency=concurrency)

    total = len(rows)
    tier_counts = {}
    for r in rows:
        tier_counts[r.oracle_tier] = tier_counts.get(r.oracle_tier, 0) + 1

    print(f"\nOracle table built: {total} prompts")
    print("Oracle tier distribution:")
    for tier, count in sorted(tier_counts.items()):
        print(f"  {tier}: {count} ({count/total*100:.1f}%)")

    total_cost = sum(
        e.cost
        for r in rows
        for e in r.entries.values()
    )
    print(f"Total evaluation cost: ${total_cost:.4f}")

    path = save_oracle_table(rows)
    print(f"Saved to: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model names (default: all)")
    args = parser.parse_args()
    models = args.models.split(",") if args.models else None
    asyncio.run(run(models, args.concurrency))


if __name__ == "__main__":
    main()
