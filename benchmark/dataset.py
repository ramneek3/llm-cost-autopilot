"""Benchmark dataset: diverse prompts across task types with automatic quality checks.

Provides a mix of:
- Extraction tasks (pull structured data from text)
- Classification tasks (categorize inputs)
- Summarization tasks (condense long text)
- Reasoning tasks (multi-step logic, analysis)
- Code tasks (generate, debug, explain code)

Each prompt includes metadata for automatic quality evaluation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BenchmarkPrompt:
    id: str
    text: str
    task_type: str  # extraction, classification, summarization, reasoning, code
    expected_tier: str  # low, medium, high
    quality_check: dict = field(default_factory=dict)
    # quality_check examples:
    #   {"type": "exact_fields", "fields": ["name", "email", "phone"]}
    #   {"type": "valid_json"}
    #   {"type": "classification", "valid_labels": ["positive", "negative", "neutral"]}
    #   {"type": "min_words", "count": 50}


def load_dataset(path: Path | None = None) -> list[BenchmarkPrompt]:
    """Load benchmark prompts from JSON file, or return built-in defaults."""
    if path and path.exists():
        raw = json.loads(path.read_text())
        return [BenchmarkPrompt(**p) for p in raw]
    return _builtin_dataset()


def save_dataset(prompts: list[BenchmarkPrompt], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [
        {"id": p.id, "text": p.text, "task_type": p.task_type,
         "expected_tier": p.expected_tier, "quality_check": p.quality_check}
        for p in prompts
    ]
    path.write_text(json.dumps(data, indent=2))


def _builtin_dataset() -> list[BenchmarkPrompt]:
    """Built-in diverse prompt set covering all tiers and task types."""
    prompts = []

    # ── Tier 1 (Low): Extraction, reformatting, basic Q&A ──

    prompts.extend([
        BenchmarkPrompt(
            id="ext-001", task_type="extraction", expected_tier="low",
            text="Extract the person's name, email, and phone number from this text: "
                 "'Hi, I'm Jane Smith. You can reach me at jane.smith@example.com or call 555-0123.'",
            quality_check={"type": "exact_fields", "fields": ["Jane Smith", "jane.smith@example.com", "555-0123"]},
        ),
        BenchmarkPrompt(
            id="ext-002", task_type="extraction", expected_tier="low",
            text="What is the capital of France?",
            quality_check={"type": "exact_fields", "fields": ["Paris"]},
        ),
        BenchmarkPrompt(
            id="ext-003", task_type="extraction", expected_tier="low",
            text="Convert this date to ISO 8601 format: March 15, 2024",
            quality_check={"type": "exact_fields", "fields": ["2024-03-15"]},
        ),
        BenchmarkPrompt(
            id="ext-004", task_type="extraction", expected_tier="low",
            text="Extract all URLs from this text: 'Visit https://example.com or http://test.org for more info. "
                 "Also check out https://docs.python.org/3/'",
            quality_check={"type": "exact_fields", "fields": ["https://example.com", "http://test.org", "https://docs.python.org/3/"]},
        ),
        BenchmarkPrompt(
            id="ext-005", task_type="extraction", expected_tier="low",
            text="Reformat this JSON into a markdown table: "
                 '[{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]',
            quality_check={"type": "exact_fields", "fields": ["Alice", "Bob", "30", "25"]},
        ),
        BenchmarkPrompt(
            id="fmt-001", task_type="extraction", expected_tier="low",
            text="Convert the following list to a numbered list: apples, bananas, oranges, grapes",
            quality_check={"type": "exact_fields", "fields": ["1", "apples", "bananas", "oranges", "grapes"]},
        ),
        BenchmarkPrompt(
            id="fmt-002", task_type="extraction", expected_tier="low",
            text="Translate 'Hello, how are you?' to Spanish.",
            quality_check={"type": "exact_fields", "fields": ["Hola"]},
        ),
        BenchmarkPrompt(
            id="qa-001", task_type="extraction", expected_tier="low",
            text="Given this context: 'The Python programming language was created by Guido van Rossum and first released in 1991.' "
                 "Who created Python?",
            quality_check={"type": "exact_fields", "fields": ["Guido van Rossum"]},
        ),
    ])

    # ── Tier 2 (Medium): Summarization, classification, structured analysis ──

    prompts.extend([
        BenchmarkPrompt(
            id="cls-001", task_type="classification", expected_tier="medium",
            text="Classify the sentiment of this product review as positive, negative, or neutral: "
                 "'The laptop arrived on time and works great for basic tasks, but the battery life is disappointing.'",
            quality_check={"type": "classification", "valid_labels": ["positive", "negative", "neutral", "mixed"]},
        ),
        BenchmarkPrompt(
            id="cls-002", task_type="classification", expected_tier="medium",
            text="Categorize this support ticket: 'I've been charged twice for my subscription this month. "
                 "Order #12345. Please refund the duplicate charge.'",
            quality_check={"type": "classification", "valid_labels": ["billing", "refund", "duplicate charge", "payment"]},
        ),
        BenchmarkPrompt(
            id="sum-001", task_type="summarization", expected_tier="medium",
            text="Summarize in 2-3 sentences: Machine learning is a subset of artificial intelligence that "
                 "focuses on building systems that learn from data. Unlike traditional programming where rules "
                 "are explicitly coded, ML systems improve their performance through experience. Common approaches "
                 "include supervised learning (learning from labeled examples), unsupervised learning (finding "
                 "patterns in unlabeled data), and reinforcement learning (learning through trial and error with rewards). "
                 "Applications span from image recognition and natural language processing to recommendation systems "
                 "and autonomous vehicles.",
            quality_check={"type": "min_words", "count": 20},
        ),
        BenchmarkPrompt(
            id="sum-002", task_type="summarization", expected_tier="medium",
            text="Explain what a REST API is to someone who knows programming but hasn't worked with APIs before.",
            quality_check={"type": "min_words", "count": 50},
        ),
        BenchmarkPrompt(
            id="struct-001", task_type="classification", expected_tier="medium",
            text='Parse this job posting and return structured JSON with fields: title, company, location, salary_range, requirements. '
                 '"We are hiring a Senior Backend Engineer at TechCorp in San Francisco. Salary: $150k-$200k. '
                 'Requirements: 5+ years Python, distributed systems experience, CS degree preferred."',
            quality_check={"type": "valid_json"},
        ),
        BenchmarkPrompt(
            id="struct-002", task_type="classification", expected_tier="medium",
            text="Convert this unstructured meeting notes into an action items list with owner and deadline: "
                 "'John will finish the API docs by Friday. Sarah needs to review the PR today. "
                 "Team agreed to postpone the database migration to next sprint.'",
            quality_check={"type": "exact_fields", "fields": ["John", "Sarah", "API docs", "Friday", "PR", "database migration"]},
        ),
    ])

    # ── Tier 3 (High): Multi-step reasoning, creative, nuanced judgment ──

    prompts.extend([
        BenchmarkPrompt(
            id="reason-001", task_type="reasoning", expected_tier="high",
            text="A company has three products: A ($10, 40% margin), B ($25, 60% margin), C ($50, 30% margin). "
                 "They sell 1000 units of A, 500 of B, and 200 of C per month. They can invest $50,000 to either: "
                 "(1) increase A's sales by 30%, (2) increase B's margin to 70%, or (3) launch product D at $35 with "
                 "45% margin, projected 300 units/month. Which option maximizes monthly profit? Show your work.",
            quality_check={"type": "min_words", "count": 100},
        ),
        BenchmarkPrompt(
            id="reason-002", task_type="reasoning", expected_tier="high",
            text="Analyze the trade-offs between microservices and monolithic architecture for a startup "
                 "with 5 engineers building a B2B SaaS product. Consider: team size, deployment complexity, "
                 "debugging, performance, and scaling. Provide a recommendation with justification.",
            quality_check={"type": "min_words", "count": 150},
        ),
        BenchmarkPrompt(
            id="code-001", task_type="code", expected_tier="high",
            text="Write a Python function that implements a least-recently-used (LRU) cache with O(1) get and put "
                 "operations. Include type hints and handle edge cases. Explain your data structure choices.",
            quality_check={"type": "exact_fields", "fields": ["def ", "class ", "dict", "OrderedDict"]},
        ),
        BenchmarkPrompt(
            id="code-002", task_type="code", expected_tier="high",
            text="Debug this Python code and explain what's wrong:\n"
                 "```python\n"
                 "def merge_sorted(a, b):\n"
                 "    result = []\n"
                 "    i = j = 0\n"
                 "    while i < len(a) and j < len(b):\n"
                 "        if a[i] <= b[j]:\n"
                 "            result.append(a[i])\n"
                 "            i += 1\n"
                 "        else:\n"
                 "            result.append(b[j])\n"
                 "            j += 1\n"
                 "    return result\n"
                 "```",
            quality_check={"type": "min_words", "count": 30},
        ),
        BenchmarkPrompt(
            id="creative-001", task_type="reasoning", expected_tier="high",
            text="Design a database schema for a multi-tenant project management tool that supports: "
                 "projects, tasks with subtasks, team members with roles, time tracking, and audit logging. "
                 "Consider performance at scale (100K+ tasks per tenant). Provide the schema with indexes "
                 "and explain your normalization decisions.",
            quality_check={"type": "min_words", "count": 200},
        ),
        BenchmarkPrompt(
            id="reason-003", task_type="reasoning", expected_tier="high",
            text="Compare three approaches to rate limiting in a distributed system: "
                 "token bucket, sliding window log, and fixed window counter. "
                 "For each, discuss: accuracy, memory usage, and behavior at window boundaries. "
                 "Which would you recommend for a payment processing API handling 10,000 req/s?",
            quality_check={"type": "min_words", "count": 150},
        ),
    ])

    return prompts
