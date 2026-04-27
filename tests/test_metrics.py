"""Metrics: known-input/known-output coverage."""

from __future__ import annotations

import pytest

from eval.metrics import (
    accuracy_delta,
    bootstrap_ci,
    exact_match,
    f1_token_overlap,
    faithfulness_score,
    rouge_l,
    semantic_similarity,
)


def test_em_basic_match() -> None:
    assert exact_match("Paris", "paris") == 1.0

def test_em_squad_normalization_strips_articles() -> None:
    assert exact_match("the dog", "dog") == 1.0

def test_em_strips_punctuation() -> None:
    assert exact_match("Paris.", "Paris") == 1.0

def test_em_mismatch() -> None:
    assert exact_match("Paris", "London") == 0.0

def test_f1_perfect_match() -> None:
    assert f1_token_overlap("the quick brown fox", "the quick brown fox") == pytest.approx(1.0)

def test_f1_partial_match() -> None:
    assert f1_token_overlap("the quick brown fox", "the lazy brown fox") == pytest.approx(2 / 3)

def test_f1_no_overlap() -> None:
    assert f1_token_overlap("apple banana", "cherry date") == 0.0

def test_f1_both_empty() -> None:
    assert f1_token_overlap("", "") == 1.0

def test_f1_one_empty() -> None:
    assert f1_token_overlap("", "hello world") == 0.0

def test_rouge_l_perfect_match() -> None:
    assert rouge_l("the cat sat on the mat", "the cat sat on the mat") == pytest.approx(1.0)

def test_rouge_l_partial_match() -> None:
    score = rouge_l("the cat sat on mat", "a cat on mat")
    assert 0.0 < score < 1.0

def test_rouge_l_no_match() -> None:
    assert rouge_l("apple banana", "cherry date") == 0.0

def test_tfidf_cosine_identical_strings_high() -> None:
    score = semantic_similarity("the quick brown fox", "the quick brown fox")
    assert score == pytest.approx(1.0, abs=1e-6)

def test_tfidf_cosine_unrelated_low() -> None:
    score = semantic_similarity("financial earnings report", "garden vegetable soup")
    assert score < 0.2

def test_semantic_similarity_handles_empty() -> None:
    assert semantic_similarity("", "anything") == 0.0

def test_accuracy_delta_positive_when_compression_helps() -> None:
    assert accuracy_delta(0.7, 0.75) == pytest.approx(0.05)

def test_accuracy_delta_negative_when_compression_hurts() -> None:
    assert accuracy_delta(0.8, 0.7) == pytest.approx(-0.1)

def test_faithfulness_lexical_full_overlap_high() -> None:
    src = "Apple reported revenue of forty billion dollars in the third quarter."
    summary = "Apple revenue was forty billion dollars in Q3."
    assert faithfulness_score(summary, src) > 0.4

def test_faithfulness_lexical_invented_facts_low() -> None:
    src = "Apple reported revenue of forty billion dollars."
    summary = "Microsoft acquired three rocket companies and built a moonbase."
    assert faithfulness_score(summary, src) < 0.2

def test_faithfulness_handles_empty() -> None:
    assert faithfulness_score("", "any source") == 0.0
    assert faithfulness_score("any summary", "") == 0.0

def test_bootstrap_ci_brackets_mean() -> None:
    values = [0.1, 0.3, 0.5, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.4]
    mean = sum(values) / len(values)
    lo, hi = bootstrap_ci(values, n_resamples=500, seed=1)
    assert lo <= mean <= hi

def test_bootstrap_ci_constant_input_zero_width() -> None:
    values = [0.5] * 20
    lo, hi = bootstrap_ci(values, n_resamples=200, seed=1)
    assert lo == pytest.approx(0.5)
    assert hi == pytest.approx(0.5)

def test_bootstrap_ci_empty_returns_zeros() -> None:
    assert bootstrap_ci([]) == (0.0, 0.0)

def test_bootstrap_ci_single_value() -> None:
    assert bootstrap_ci([0.7]) == (0.7, 0.7)
