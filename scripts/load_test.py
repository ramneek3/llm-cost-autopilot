"""Load test: send diverse prompts through the system and generate a cost report.

Usage:
    python -m scripts.load_test [--url http://localhost:8000] [--count 100] [--concurrency 10]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.dataset import load_dataset


async def send_request(
    client: httpx.AsyncClient,
    prompt: str,
    semaphore: asyncio.Semaphore,
) -> dict | None:
    async with semaphore:
        try:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                },
                timeout=120.0,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"  Error: {e}")
            return None


async def run_load_test(url: str, count: int, concurrency: int):
    prompts_data = load_dataset()
    # Cycle through prompts to reach desired count
    prompts = []
    while len(prompts) < count:
        for p in prompts_data:
            prompts.append(p.text)
            if len(prompts) >= count:
                break

    print(f"Sending {count} requests to {url} (concurrency={concurrency})")
    semaphore = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(base_url=url) as client:
        start = time.perf_counter()
        tasks = [send_request(client, p, semaphore) for p in prompts]
        results = await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - start

    successes = [r for r in results if r is not None]
    failures = count - len(successes)

    total_cost = sum(r["routing"]["cost"] for r in successes)
    total_tokens = sum(r["usage"]["total_tokens"] for r in successes)
    models_used = {}
    tiers_used = {}
    escalations = 0

    for r in successes:
        model = r["routing"]["routed_model"]
        tier = r["routing"]["tier"]
        models_used[model] = models_used.get(model, 0) + 1
        tiers_used[tier] = tiers_used.get(tier, 0) + 1
        if r["routing"]["escalated"]:
            escalations += 1

    # Baseline cost estimate
    baseline_rate = 0.000010
    baseline_cost = total_tokens * baseline_rate
    net_savings = baseline_cost - total_cost
    savings_pct = (net_savings / baseline_cost * 100) if baseline_cost > 0 else 0

    print(f"\n{'='*60}")
    print(f"LOAD TEST REPORT")
    print(f"{'='*60}")
    print(f"Requests:     {count} sent, {len(successes)} succeeded, {failures} failed")
    print(f"Duration:     {elapsed:.1f}s ({len(successes)/elapsed:.1f} req/s)")
    print(f"Total tokens: {total_tokens:,}")
    print(f"")
    print(f"COST ANALYSIS")
    print(f"  Actual cost:   ${total_cost:.4f}")
    print(f"  Baseline cost: ${baseline_cost:.4f} (all via claude-sonnet-5-5)")
    print(f"  Net savings:   ${net_savings:.4f} ({savings_pct:.1f}%)")
    print(f"")
    print(f"ROUTING")
    print(f"  Escalation rate: {escalations/len(successes)*100:.1f}%")
    print(f"  Models used:")
    for m, c in sorted(models_used.items(), key=lambda x: -x[1]):
        print(f"    {m}: {c} ({c/len(successes)*100:.1f}%)")
    print(f"  Tiers used:")
    for t, c in sorted(tiers_used.items()):
        print(f"    {t}: {c} ({c/len(successes)*100:.1f}%)")

    # Save report
    report = {
        "timestamp": time.time(),
        "config": {"url": url, "count": count, "concurrency": concurrency},
        "results": {
            "successes": len(successes),
            "failures": failures,
            "duration_s": elapsed,
            "total_cost": total_cost,
            "baseline_cost": baseline_cost,
            "net_savings": net_savings,
            "savings_pct": savings_pct,
            "total_tokens": total_tokens,
            "escalation_rate": escalations / len(successes) if successes else 0,
            "models_used": models_used,
            "tiers_used": tiers_used,
        },
    }
    report_path = Path("data/load_test_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nReport saved to {report_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=10)
    args = parser.parse_args()
    asyncio.run(run_load_test(args.url, args.count, args.concurrency))


if __name__ == "__main__":
    main()
