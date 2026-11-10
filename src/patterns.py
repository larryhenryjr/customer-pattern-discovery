"""Customer segmentation, dimensionality reduction, and basket-analysis tools."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class SegmentResult:
    """Customer segment assignments and interpretable centroids."""

    labels: pd.Series
    centroids: pd.DataFrame
    inertia: float


@dataclass(frozen=True)
class ReductionResult:
    """Principal-component scores and explained-variance metadata."""

    scores: pd.DataFrame
    loadings: pd.DataFrame
    explained_variance_ratio: np.ndarray


def validate_numeric_features(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return a complete numeric feature matrix for modeling."""
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    matrix = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    if matrix.isna().any().any():
        raise ValueError("Selected features must be numeric and nonmissing.")
    if len(matrix) < 2:
        raise ValueError("At least two customer records are required.")
    return matrix


def segment_customers(
    frame: pd.DataFrame,
    feature_columns: list[str],
    n_clusters: int = 4,
    random_state: int = 42,
) -> SegmentResult:
    """Standardize customer features and fit K-means segments."""
    matrix = validate_numeric_features(frame, feature_columns)
    if not 2 <= n_clusters <= len(matrix):
        raise ValueError("n_clusters must be between 2 and the number of records.")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
    labels = model.fit_predict(scaled)
    centroids = pd.DataFrame(
        scaler.inverse_transform(model.cluster_centers_),
        columns=feature_columns,
    )
    centroids.index.name = "segment"
    return SegmentResult(
        labels=pd.Series(labels, index=frame.index, name="segment"),
        centroids=centroids,
        inertia=float(model.inertia_),
    )


def reduce_dimensions(
    frame: pd.DataFrame,
    feature_columns: list[str],
    variance_threshold: float = 0.90,
) -> ReductionResult:
    """Fit PCA and retain enough components to meet a variance threshold."""
    if not 0 < variance_threshold <= 1:
        raise ValueError("variance_threshold must be in the interval (0, 1].")
    matrix = validate_numeric_features(frame, feature_columns)
    scaled = StandardScaler().fit_transform(matrix)
    full_model = PCA().fit(scaled)
    cumulative = np.cumsum(full_model.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumulative, variance_threshold) + 1)
    model = PCA(n_components=n_components).fit(scaled)
    columns = [f"PC{index + 1}" for index in range(n_components)]
    scores = pd.DataFrame(model.transform(scaled), columns=columns, index=frame.index)
    loadings = pd.DataFrame(model.components_.T, index=feature_columns, columns=columns)
    return ReductionResult(scores, loadings, model.explained_variance_ratio_)


def mine_associations(
    transactions: list[list[str]],
    min_support: float = 0.02,
    min_confidence: float = 0.30,
) -> pd.DataFrame:
    """Discover product associations from transaction-level item lists."""
    cleaned = [sorted({item.strip() for item in basket if item.strip()}) for basket in transactions]
    cleaned = [basket for basket in cleaned if basket]
    if not cleaned:
        raise ValueError("At least one nonempty transaction is required.")
    encoder = TransactionEncoder()
    encoded = pd.DataFrame(encoder.fit(cleaned).transform(cleaned), columns=encoder.columns_)
    itemsets = apriori(encoded, min_support=min_support, use_colnames=True)
    if itemsets.empty:
        return pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])
    rules = association_rules(itemsets, metric="confidence", min_threshold=min_confidence)
    if rules.empty:
        return pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])
    output = rules.loc[:, ["antecedents", "consequents", "support", "confidence", "lift"]].copy()
    output["antecedents"] = output["antecedents"].map(lambda values: ", ".join(sorted(values)))
    output["consequents"] = output["consequents"].map(lambda values: ", ".join(sorted(values)))
    return output.sort_values(["lift", "confidence", "support"], ascending=False).reset_index(drop=True)
