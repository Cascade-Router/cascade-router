"""
Generate latency benchmark charts for Cascade Router documentation.

Usage:
    python -m src.generate_charts
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "docs" / "latency_chart.png"

ARCHITECTURES = [
    "Cascade Router\n(C++ / ONNX)",
    "Python Proxy\n(e.g., LiteLLM)",
    "SaaS Router\n(Network Hop)",
    "LLM-as-a-Judge\n(Zero-Shot)",
]
LATENCIES_MS = [4.6, 65.0, 180.0, 850.0]

BRAND_BLUE = "#38bdf8"
COMPETITOR_GRAY = "#4b5563"


def generate_latency_chart(output_path: Path = OUTPUT_PATH) -> Path:
    """Create log-scale latency comparison bar chart."""
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    colors = [BRAND_BLUE] + [COMPETITOR_GRAY] * (len(ARCHITECTURES) - 1)
    bars = ax.bar(ARCHITECTURES, LATENCIES_MS, color=colors, edgecolor="#1f2937", linewidth=0.8)

    ax.set_yscale("log")
    ax.set_ylabel("Routing Overhead (ms, log scale)", fontsize=12, color="#e5e7eb")
    ax.set_title(
        "LLM Routing Latency: Cascade Router vs Industry Standards",
        fontsize=14,
        fontweight="bold",
        color="#f9fafb",
        pad=16,
    )
    ax.tick_params(axis="x", colors="#d1d5db", labelsize=10)
    ax.tick_params(axis="y", colors="#d1d5db", labelsize=10)
    ax.grid(axis="y", which="both", linestyle="--", alpha=0.25, color="#6b7280")

    for bar, latency in zip(bars, LATENCIES_MS, strict=True):
        label = f"{latency:g} ms"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.15,
            label,
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#f9fafb",
        )

    ax.set_ylim(min(LATENCIES_MS) * 0.5, max(LATENCIES_MS) * 2.5)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, transparent=True, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    path = generate_latency_chart()
    print(f"Saved latency chart to {path}")


if __name__ == "__main__":
    main()
