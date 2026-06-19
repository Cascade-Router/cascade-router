"""
Export trained logistic regression weights for C++ inference.

Loads models/tiny_router.pkl and writes models/router_weights.json.

Usage:
    python -m src.export_weights
"""

from __future__ import annotations

import json

import joblib

from src.config import MODELS_DIR, TINY_ROUTER_MODEL_PATH

OUTPUT_PATH = MODELS_DIR / "router_weights.json"


def main() -> None:
    if not TINY_ROUTER_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {TINY_ROUTER_MODEL_PATH}. Run `python -m src.train_router` first."
        )

    clf = joblib.load(TINY_ROUTER_MODEL_PATH)
    weights = clf.coef_[0].tolist()
    intercept = float(clf.intercept_[0])

    payload = {"weights": weights, "intercept": intercept}
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Exported {len(weights)} weights + intercept to {OUTPUT_PATH}")
    print(f"  intercept: {intercept:.6f}")


if __name__ == "__main__":
    main()
