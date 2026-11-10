"""Tests for customer-pattern discovery utilities."""

from __future__ import annotations

import pandas as pd

from src.patterns import mine_associations, reduce_dimensions, segment_customers


def sample_customers() -> pd.DataFrame:
    """Create a small deterministic customer matrix."""
    return pd.DataFrame(
        {
            "tenure": [3, 5, 8, 40, 45, 50],
            "monthly_charge": [35, 40, 42, 90, 94, 100],
            "bandwidth": [100, 120, 140, 800, 850, 900],
        }
    )


def test_segment_customers_returns_requested_clusters() -> None:
    """K-means should return one label per row and two centroids."""
    result = segment_customers(
        sample_customers(),
        ["tenure", "monthly_charge", "bandwidth"],
        n_clusters=2,
    )
    assert result.labels.nunique() == 2
    assert result.centroids.shape == (2, 3)


def test_reduce_dimensions_meets_variance_threshold() -> None:
    """Retained components should meet the requested variance target."""
    result = reduce_dimensions(
        sample_customers(),
        ["tenure", "monthly_charge", "bandwidth"],
        variance_threshold=0.90,
    )
    assert result.explained_variance_ratio.sum() >= 0.90
    assert len(result.scores) == 6


def test_mine_associations_discovers_repeated_bundle() -> None:
    """Repeated product pairs should produce an association rule."""
    transactions = [
        ["Streaming TV", "Internet"],
        ["Streaming TV", "Internet"],
        ["Internet", "Security"],
        ["Streaming TV", "Internet", "Security"],
    ]
    rules = mine_associations(transactions, min_support=0.25, min_confidence=0.50)
    assert not rules.empty
    assert "lift" in rules.columns
