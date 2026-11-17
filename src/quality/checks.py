"""Task-specific quality checks.

Goes beyond simple agreement: uses exact match, JSON schema validation,
unit tests, and rubric-based judging depending on the task type.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass


@dataclass
class CheckResult:
    score: float  # 0.0 to 5.0
    passed: bool
    reason: str


def check_extraction(response_text: str, expected_fields: list[str]) -> CheckResult:
    """For extraction tasks: did it get all the key fields?"""
    text_lower = response_text.lower()
    found = sum(1 for f in expected_fields if f.lower() in text_lower)
    ratio = found / len(expected_fields) if expected_fields else 1.0
    score = ratio * 5.0
    return CheckResult(
        score=score,
        passed=ratio >= 0.8,
        reason=f"Found {found}/{len(expected_fields)} expected fields",
    )


def check_json_validity(response_text: str) -> CheckResult:
    """For structured output tasks: is the JSON valid?"""
    text = response_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:]).strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, (dict, list)):
            return CheckResult(score=5.0, passed=True, reason="Valid JSON")
        return CheckResult(score=3.0, passed=True, reason="Valid JSON but not object/array")
    except (json.JSONDecodeError, ValueError) as e:
        return CheckResult(score=0.0, passed=False, reason=f"Invalid JSON: {e}")


def check_classification(response_text: str, valid_labels: list[str]) -> CheckResult:
    """For classification tasks: does the output match a valid label?"""
    text_lower = response_text.strip().lower()
    for label in valid_labels:
        if label.lower() in text_lower:
            return CheckResult(score=5.0, passed=True, reason=f"Matched label: {label}")
    return CheckResult(score=0.0, passed=False, reason=f"No valid label found in response")


def check_length_ratio(response_text: str, reference_text: str, tolerance: float = 0.5) -> CheckResult:
    """Basic length sanity check: is the response roughly the right length?"""
    resp_len = len(response_text.split())
    ref_len = len(reference_text.split())
    if ref_len == 0:
        return CheckResult(score=3.0, passed=True, reason="No reference length")
    ratio = resp_len / ref_len
    if 1 - tolerance <= ratio <= 1 + tolerance:
        return CheckResult(score=4.0, passed=True, reason=f"Length ratio {ratio:.2f} within tolerance")
    return CheckResult(
        score=max(0, 3.0 - abs(1 - ratio) * 2),
        passed=False,
        reason=f"Length ratio {ratio:.2f} outside tolerance {tolerance}",
    )


def check_non_empty(response_text: str) -> CheckResult:
    """Basic check: is the response non-empty and substantial?"""
    text = response_text.strip()
    if not text:
        return CheckResult(score=0.0, passed=False, reason="Empty response")
    word_count = len(text.split())
    if word_count < 2:
        return CheckResult(score=1.0, passed=False, reason=f"Too short: {word_count} words")
    return CheckResult(score=4.0, passed=True, reason=f"Non-empty response: {word_count} words")
