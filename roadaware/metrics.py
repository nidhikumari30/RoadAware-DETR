"""Small metric helpers for project experiment reports."""

from __future__ import annotations


def precision(true_positive: int, false_positive: int) -> float:
    denominator = true_positive + false_positive
    return true_positive / denominator if denominator else 0.0


def recall(true_positive: int, false_negative: int) -> float:
    denominator = true_positive + false_negative
    return true_positive / denominator if denominator else 0.0


def false_discovery_rate(true_positive: int, false_positive: int) -> float:
    denominator = true_positive + false_positive
    return false_positive / denominator if denominator else 0.0


def f1_score(true_positive: int, false_positive: int, false_negative: int) -> float:
    p = precision(true_positive, false_positive)
    r = recall(true_positive, false_negative)
    return 2 * p * r / (p + r) if p + r else 0.0
