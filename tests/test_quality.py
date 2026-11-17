from src.quality.checks import (
    check_classification,
    check_extraction,
    check_json_validity,
    check_length_ratio,
    check_non_empty,
)


def test_extraction_all_found():
    result = check_extraction("Name: Alice, Email: alice@test.com", ["Alice", "alice@test.com"])
    assert result.passed
    assert result.score == 5.0


def test_extraction_partial():
    result = check_extraction("Name: Alice", ["Alice", "alice@test.com", "555-1234"])
    assert not result.passed


def test_json_valid():
    result = check_json_validity('{"key": "value"}')
    assert result.passed


def test_json_invalid():
    result = check_json_validity("not json at all")
    assert not result.passed


def test_json_in_code_block():
    result = check_json_validity('```json\n{"key": "value"}\n```')
    assert result.passed


def test_classification_match():
    result = check_classification("The sentiment is positive.", ["positive", "negative", "neutral"])
    assert result.passed


def test_classification_no_match():
    result = check_classification("The weather is sunny.", ["positive", "negative", "neutral"])
    assert not result.passed


def test_non_empty():
    assert check_non_empty("Hello world").passed
    assert not check_non_empty("").passed
    assert not check_non_empty("  ").passed


def test_length_ratio():
    result = check_length_ratio("one two three four", "one two three four five")
    assert result.passed
