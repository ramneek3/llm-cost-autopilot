"""Learned router: TF-IDF + logistic regression predicting complexity tier.

ML imports (sklearn, numpy, joblib) are lazy — loaded only when training or
loading a model — so the serverless bundle stays under Vercel's size limit.
"""

from __future__ import annotations

import re
from pathlib import Path

from src.models.schemas import Tier

MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models"


def extract_features(prompt: str) -> dict[str, float]:
    words = prompt.split()
    sentences = [s.strip() for s in re.split(r"[.!?]+", prompt) if s.strip()]
    avg_word_len = sum(len(w) for w in words) / len(words) if words else 0
    return {
        "token_count": len(words),
        "char_count": len(prompt),
        "sentence_count": len(sentences),
        "avg_word_length": avg_word_len,
        "question_marks": prompt.count("?"),
        "has_code_block": float("```" in prompt or "def " in prompt or "function " in prompt),
        "has_json_request": float(any(k in prompt.lower() for k in ["json", "schema", "format"])),
        "has_list_request": float(any(k in prompt.lower() for k in ["list ", "enumerate", "bullet"])),
        "constraint_count": len(re.findall(r"\b(must|should|ensure|require|constraint|at least|at most)\b", prompt, re.I)),
        "reasoning_signals": len(re.findall(r"\b(analyze|reason|think|compare|evaluate|debate|argue)\b", prompt, re.I)),
        "extraction_signals": len(re.findall(r"\b(extract|parse|find|get|retrieve|pull out)\b", prompt, re.I)),
        "creative_signals": len(re.findall(r"\b(write|create|generate|compose|draft|invent|imagine)\b", prompt, re.I)),
    }


class LearnedRouter:
    def __init__(self, model_path: Path | None = None):
        self._pipeline = None
        self._feature_names: list[str] = []
        if model_path and model_path.exists():
            self.load(model_path)

    @property
    def is_trained(self) -> bool:
        return self._pipeline is not None

    def train(
        self,
        prompts: list[str],
        labels: list[Tier],
        test_size: float = 0.2,
    ) -> dict:
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, classification_report
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline

        tier_to_int = {Tier.LOW: 0, Tier.MEDIUM: 1, Tier.HIGH: 2}
        y = np.array([tier_to_int[t] for t in labels])

        X_train_text, X_test_text, y_train, y_test = train_test_split(
            prompts, y, test_size=test_size, random_state=42, stratify=y,
        )

        self._pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=3000, ngram_range=(1, 2), sublinear_tf=True)),
            ("clf", LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")),
        ])
        self._pipeline.fit(X_train_text, y_train)

        y_pred_train = self._pipeline.predict(X_train_text)
        y_pred_test = self._pipeline.predict(X_test_text)

        return {
            "train_accuracy": float(accuracy_score(y_train, y_pred_train)),
            "test_accuracy": float(accuracy_score(y_test, y_pred_test)),
            "classification_report": classification_report(
                y_test, y_pred_test, target_names=["low", "medium", "high"], output_dict=True,
            ),
            "train_size": len(X_train_text),
            "test_size": len(X_test_text),
        }

    def predict(self, prompt: str) -> tuple[Tier, float]:
        if not self._pipeline:
            raise RuntimeError("Router not trained. Call train() or load() first.")

        import numpy as np

        int_to_tier = {0: Tier.LOW, 1: Tier.MEDIUM, 2: Tier.HIGH}
        probs = self._pipeline.predict_proba([prompt])[0]
        predicted_class = int(np.argmax(probs))
        confidence = float(probs[predicted_class])
        return int_to_tier[predicted_class], confidence

    def predict_batch(self, prompts: list[str]) -> list[tuple[Tier, float]]:
        if not self._pipeline:
            raise RuntimeError("Router not trained.")

        import numpy as np

        int_to_tier = {0: Tier.LOW, 1: Tier.MEDIUM, 2: Tier.HIGH}
        probs = self._pipeline.predict_proba(prompts)
        results = []
        for p in probs:
            cls = int(np.argmax(p))
            results.append((int_to_tier[cls], float(p[cls])))
        return results

    def save(self, path: Path | None = None) -> Path:
        import joblib

        path = path or MODEL_DIR / "router_latest.joblib"
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._pipeline, path)
        return path

    def load(self, path: Path) -> None:
        import joblib

        self._pipeline = joblib.load(path)
