from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from pathlib import Path

from src.models.schemas import LLMResponse, QualityScore

_default_path = str(Path(__file__).resolve().parent.parent.parent / "autopilot.db")
DB_PATH = os.environ.get("DATABASE_PATH", _default_path)
if DB_PATH != ":memory:" and not os.access(str(Path(DB_PATH).parent), os.W_OK):
    DB_PATH = "/tmp/autopilot.db"


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS requests (
            request_id TEXT PRIMARY KEY,
            timestamp REAL NOT NULL,
            prompt_hash TEXT NOT NULL,
            prompt_preview TEXT,
            tier TEXT NOT NULL,
            model_name TEXT NOT NULL,
            provider TEXT NOT NULL,
            input_tokens INTEGER NOT NULL,
            output_tokens INTEGER NOT NULL,
            latency_ms REAL NOT NULL,
            cost REAL NOT NULL,
            routed_by TEXT NOT NULL,
            escalated INTEGER NOT NULL DEFAULT 0,
            escalation_reason TEXT
        );

        CREATE TABLE IF NOT EXISTS quality_scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id TEXT NOT NULL,
            rubric_score REAL NOT NULL,
            judge_model TEXT NOT NULL,
            reference_model TEXT,
            agreement_score REAL,
            is_pass INTEGER NOT NULL,
            failure_reason TEXT,
            details TEXT,
            FOREIGN KEY (request_id) REFERENCES requests(request_id)
        );

        CREATE TABLE IF NOT EXISTS router_versions (
            version_id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at REAL NOT NULL,
            accuracy REAL NOT NULL,
            holdout_accuracy REAL NOT NULL,
            is_active INTEGER NOT NULL DEFAULT 0,
            model_path TEXT NOT NULL,
            metadata TEXT
        );

        CREATE TABLE IF NOT EXISTS training_samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt_hash TEXT NOT NULL,
            prompt_text TEXT NOT NULL,
            true_tier TEXT NOT NULL,
            predicted_tier TEXT,
            source TEXT NOT NULL,
            features TEXT,
            created_at REAL NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_requests_timestamp ON requests(timestamp);
        CREATE INDEX IF NOT EXISTS idx_requests_model ON requests(model_name);
        CREATE INDEX IF NOT EXISTS idx_quality_request ON quality_scores(request_id);
    """)
    conn.close()


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def log_request(response: LLMResponse, prompt_text: str) -> None:
    conn = _get_conn()
    conn.execute(
        """INSERT OR REPLACE INTO requests
           (request_id, timestamp, prompt_hash, prompt_preview, tier, model_name, provider,
            input_tokens, output_tokens, latency_ms, cost, routed_by, escalated, escalation_reason)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            response.request_id,
            response.timestamp,
            prompt_hash(prompt_text),
            prompt_text[:200],
            response.tier.value,
            response.model_name,
            response.provider.value,
            response.input_tokens,
            response.output_tokens,
            response.latency_ms,
            response.cost,
            response.routed_by,
            int(response.escalated),
            response.escalation_reason,
        ),
    )
    conn.commit()
    conn.close()


def log_quality_score(score: QualityScore) -> None:
    conn = _get_conn()
    conn.execute(
        """INSERT INTO quality_scores
           (request_id, rubric_score, judge_model, reference_model, agreement_score,
            is_pass, failure_reason, details)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            score.request_id,
            score.rubric_score,
            score.judge_model,
            score.reference_model,
            score.agreement_score,
            int(score.is_pass),
            score.failure_reason,
            json.dumps(score.details),
        ),
    )
    conn.commit()
    conn.close()


def get_stats(since_timestamp: float | None = None) -> dict:
    conn = _get_conn()
    where = ""
    params: tuple = ()
    if since_timestamp:
        where = "WHERE timestamp >= ?"
        params = (since_timestamp,)

    row = conn.execute(
        f"""SELECT
            COUNT(*) as total_requests,
            COALESCE(SUM(cost), 0) as total_cost,
            COALESCE(SUM(input_tokens + output_tokens), 0) as total_tokens,
            COALESCE(SUM(CASE WHEN escalated = 1 THEN 1 ELSE 0 END), 0) as escalated_count,
            COALESCE(AVG(latency_ms), 0) as avg_latency_ms
        FROM requests {where}""",
        params,
    ).fetchone()

    quality_row = conn.execute(
        f"""SELECT
            COUNT(*) as scored_count,
            COALESCE(AVG(qs.rubric_score), 0) as avg_quality,
            COALESCE(SUM(CASE WHEN qs.is_pass = 1 THEN 1 ELSE 0 END), 0) as pass_count
        FROM quality_scores qs
        JOIN requests r ON qs.request_id = r.request_id
        {where.replace('timestamp', 'r.timestamp')}""",
        params,
    ).fetchone()

    model_dist = conn.execute(
        f"""SELECT model_name, COUNT(*) as count, SUM(cost) as model_cost
        FROM requests {where} GROUP BY model_name""",
        params,
    ).fetchall()

    conn.close()

    total_requests = row["total_requests"]
    escalated = row["escalated_count"]
    scored = quality_row["scored_count"]

    return {
        "total_requests": total_requests,
        "total_cost": row["total_cost"],
        "total_tokens": row["total_tokens"],
        "avg_latency_ms": row["avg_latency_ms"],
        "escalation_rate": escalated / total_requests if total_requests else 0,
        "quality": {
            "scored_count": scored,
            "avg_score": quality_row["avg_quality"],
            "pass_rate": quality_row["pass_count"] / scored if scored else 0,
        },
        "model_distribution": {r["model_name"]: {"count": r["count"], "cost": r["model_cost"]} for r in model_dist},
    }
