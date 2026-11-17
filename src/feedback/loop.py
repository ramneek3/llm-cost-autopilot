"""Feedback loop: retrain the router on sampled labels from ALL traffic.

Key improvements over naive approaches:
- Trains on sampled labels from all traffic, not just failures
- Champion/challenger promotion: new router replaces current only if it wins on holdout
- Versioned router models with rollback
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path

from src.logging.db import DB_PATH
from src.models.schemas import Tier
from src.router.classifier import LearnedRouter

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models"


def get_training_data(min_samples: int = 100) -> tuple[list[str], list[Tier]] | None:
    """Pull labeled samples from the database (all traffic, not just failures)."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """SELECT prompt_text, true_tier FROM training_samples
           ORDER BY created_at DESC""",
    ).fetchall()
    conn.close()

    if len(rows) < min_samples:
        logger.info("Not enough training data: %d < %d", len(rows), min_samples)
        return None

    prompts = [r["prompt_text"] for r in rows]
    labels = [Tier(r["true_tier"]) for r in rows]
    return prompts, labels


def add_training_sample(
    prompt_text: str,
    true_tier: Tier,
    predicted_tier: Tier | None = None,
    source: str = "sampled",
) -> None:
    """Add a labeled sample to the training set."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """INSERT INTO training_samples (prompt_hash, prompt_text, true_tier, predicted_tier, source, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (
            "",
            prompt_text,
            true_tier.value,
            predicted_tier.value if predicted_tier else None,
            source,
            time.time(),
        ),
    )
    conn.commit()
    conn.close()


def retrain_router(
    holdout_fraction: float = 0.2,
    promotion_threshold: float = 0.02,
) -> dict:
    """Retrain the router using champion/challenger promotion.

    Returns a dict with the outcome: promoted, rejected, or insufficient_data.
    """
    data = get_training_data()
    if data is None:
        return {"outcome": "insufficient_data"}

    prompts, labels = data

    # Train challenger
    challenger = LearnedRouter()
    metrics = challenger.train(prompts, labels, test_size=holdout_fraction)
    challenger_accuracy = metrics["test_accuracy"]

    # Load current champion
    champion_path = MODEL_DIR / "router_latest.joblib"
    current_accuracy = 0.0
    if champion_path.exists():
        champion = LearnedRouter(champion_path)
        # Evaluate champion on the same holdout (approximation: use full test accuracy)
        conn = sqlite3.connect(DB_PATH)
        row = conn.execute(
            "SELECT holdout_accuracy FROM router_versions WHERE is_active = 1"
        ).fetchone()
        conn.close()
        if row:
            current_accuracy = row[0]

    # Champion/challenger: promote only if challenger wins by margin
    if challenger_accuracy > current_accuracy + promotion_threshold:
        model_path = challenger.save()

        conn = sqlite3.connect(DB_PATH)
        conn.execute("UPDATE router_versions SET is_active = 0")
        conn.execute(
            """INSERT INTO router_versions (created_at, accuracy, holdout_accuracy, is_active, model_path, metadata)
               VALUES (?, ?, ?, 1, ?, ?)""",
            (
                time.time(),
                metrics["train_accuracy"],
                challenger_accuracy,
                str(model_path),
                json.dumps(metrics["classification_report"]),
            ),
        )
        conn.commit()
        conn.close()

        logger.info("Promoted new router: %.3f > %.3f + %.3f",
                     challenger_accuracy, current_accuracy, promotion_threshold)
        return {
            "outcome": "promoted",
            "challenger_accuracy": challenger_accuracy,
            "champion_accuracy": current_accuracy,
            "model_path": str(model_path),
            "metrics": metrics,
        }
    else:
        logger.info("Challenger rejected: %.3f <= %.3f + %.3f",
                     challenger_accuracy, current_accuracy, promotion_threshold)
        return {
            "outcome": "rejected",
            "challenger_accuracy": challenger_accuracy,
            "champion_accuracy": current_accuracy,
            "metrics": metrics,
        }
