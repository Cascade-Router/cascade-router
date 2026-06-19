"""
Train the Tiny Routing Classifier for cascade-router.

Merges prompt features (X) with evaluation pass scores (y), fits a logistic
regression model, persists it to disk, and prints inference probabilities.

Usage:
    python -m src.train_router
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.config import (
    EVALUATION_RESULTS_PATH,
    PROMPT_FEATURES_PATH,
    RANDOM_SEED,
    TINY_ROUTER_MODEL_PATH,
    ensure_models_dir,
)

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 384
SCALAR_FEATURE_COUNT = 2
TOTAL_FEATURE_COUNT = SCALAR_FEATURE_COUNT + EMBEDDING_DIM
MIN_ROWS_FOR_SPLIT = 20


def load_training_data(
    features_path: Path | None = None,
    labels_path: Path | None = None,
) -> pd.DataFrame:
    """Load and inner-join feature rows with evaluation labels on prompt_id."""
    features_file = features_path or PROMPT_FEATURES_PATH
    labels_file = labels_path or EVALUATION_RESULTS_PATH

    if not features_file.exists():
        raise FileNotFoundError(f"Features not found at {features_file}")
    if not labels_file.exists():
        raise FileNotFoundError(f"Evaluation results not found at {labels_file}")

    features = pd.read_parquet(features_file, engine="pyarrow")
    labels = pd.read_parquet(labels_file, engine="pyarrow")

    merged = features.merge(
        labels[["prompt_id", "pass_score"]],
        on="prompt_id",
        how="inner",
    )
    if merged.empty:
        raise ValueError("No overlapping prompt_id rows between features and labels.")

    logger.info("Merged %d labeled feature rows for training", len(merged))
    return merged


def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Construct X with shape (n_samples, 386).

    Columns: token_count, structural_complexity, embedding[0..383]
    """
    scalars = df[["token_count", "structural_complexity"]].to_numpy(dtype=np.float64)
    embeddings = np.vstack(
        [np.asarray(row, dtype=np.float64).ravel() for row in df["embedding"]]
    )

    if embeddings.shape[1] != EMBEDDING_DIM:
        raise ValueError(
            f"Expected embedding dim {EMBEDDING_DIM}, got {embeddings.shape[1]}"
        )

    X = np.hstack([scalars, embeddings])
    if X.shape[1] != TOTAL_FEATURE_COUNT:
        raise ValueError(
            f"Expected {TOTAL_FEATURE_COUNT} features, got {X.shape[1]}"
        )
    return X


def build_target(df: pd.DataFrame) -> np.ndarray:
    """Return y as integer pass/fail labels."""
    return df["pass_score"].astype(int).to_numpy()


def train_router(
    df: pd.DataFrame,
    *,
    random_state: int = RANDOM_SEED,
) -> LogisticRegression:
    """Fit logistic regression on X and y; skip holdout split when n < 20."""
    X = build_feature_matrix(df)
    y = build_target(df)

    if len(df) < MIN_ROWS_FOR_SPLIT:
        logger.info(
            "Dataset has %d rows (< %d); training on full dataset without split",
            len(df),
            MIN_ROWS_FOR_SPLIT,
        )
        X_train, y_train = X, y
    else:
        X_train, _, y_train, _ = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=random_state,
            stratify=y,
        )
        logger.info("Train/holdout split: %d train rows", len(y_train))

    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(X_train, y_train)
    logger.info("Model trained on %d samples", len(y_train))
    return model


def save_model(model: LogisticRegression, path: Path | None = None) -> Path:
    """Persist trained model with joblib."""
    target = path or TINY_ROUTER_MODEL_PATH
    ensure_models_dir()
    joblib.dump(model, target)
    logger.info("Saved model to %s", target)
    return target


def pass_probability(model: LogisticRegression, X: np.ndarray) -> np.ndarray:
    """Return P(pass_score=1) for each row."""
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError(f"Model classes {classes} do not include pass label 1")
    pass_idx = classes.index(1)
    return model.predict_proba(X)[:, pass_idx]


def print_inference_report(
    df: pd.DataFrame,
    model: LogisticRegression,
) -> None:
    """Print prompt_id, actual pass score, and predicted pass probability."""
    X = build_feature_matrix(df)
    probs = pass_probability(model, X)

    print("\n" + "=" * 72)
    print("TINY ROUTER — INFERENCE ON TRAINING DATA")
    print("=" * 72)
    print(f"{'prompt_id':<38} {'actual':>8} {'p(pass)':>10}")
    print("-" * 72)

    for prompt_id, actual, prob in zip(
        df["prompt_id"].astype(str),
        df["pass_score"].astype(int),
        probs,
        strict=True,
    ):
        print(f"{prompt_id:<38} {actual:>8} {prob:>10.4f}")

    print("=" * 72)
    print(f"Features per row: {TOTAL_FEATURE_COUNT} (2 scalar + {EMBEDDING_DIM} embedding)")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    df = load_training_data()
    model = train_router(df)
    model_path = save_model(model)
    print_inference_report(df, model)
    print(f"\nModel saved to: {model_path}")


if __name__ == "__main__":
    main()
