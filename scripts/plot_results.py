from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "results_summary.csv"
FIGURES_DIR = ROOT / "figures"

RESULT_FILES = {
    "DPO": {
        42: ROOT / "outputs/minimal_dpo_pythia410m/seed_42/metrics/dpo_test_prefs.json",
        13: ROOT / "outputs/minimal_dpo_pythia410m/seed_13/metrics/dpo_test_prefs.json",
        3407: ROOT / "outputs/minimal_dpo_pythia410m/seed_3407/metrics/dpo_test_prefs.json",
    },
    "DDO-RM": {
        42: ROOT / "outputs/minimal_ddorm_pythia410m/seed_42/metrics/ddorm_test_prefs.json",
        13: ROOT / "outputs/minimal_ddorm_pythia410m/seed_13/metrics/ddorm_test_prefs.json",
        3407: ROOT / "outputs/minimal_ddorm_pythia410m/seed_3407/metrics/ddorm_test_prefs.json",
    },
}

FALLBACK_RESULTS = {
    "DPO": {
        42: {"pair_accuracy": 0.5285, "auc": 0.5335, "mean_margin": 0.1308, "num_examples": 2000, "split": "test_prefs"},
        13: {"pair_accuracy": 0.5205, "auc": 0.5301, "mean_margin": 0.1384, "num_examples": 2000, "split": "test_prefs"},
        3407: {"pair_accuracy": 0.5225, "auc": 0.5308, "mean_margin": 0.1439, "num_examples": 2000, "split": "test_prefs"},
    },
    "DDO-RM": {
        42: {"pair_accuracy": 0.5410, "auc": 0.5335, "mean_margin": 0.2995, "num_examples": 2000, "split": "test_prefs"},
        13: {"pair_accuracy": 0.5630, "auc": 0.5388, "mean_margin": 0.5196, "num_examples": 2000, "split": "test_prefs"},
        3407: {"pair_accuracy": 0.5765, "auc": 0.5423, "mean_margin": 0.7867, "num_examples": 2000, "split": "test_prefs"},
    },
}

METRICS = [
    ("pair_accuracy", "Pair Accuracy", FIGURES_DIR / "pair_accuracy_bar.png"),
    ("auc", "AUC", FIGURES_DIR / "auc_bar.png"),
    ("mean_margin", "Mean Margin", FIGURES_DIR / "mean_margin_bar.png"),
]
PAIR_LINES_PATH = FIGURES_DIR / "pair_accuracy_seed_lines.png"

METHOD_ORDER = ["DPO", "DDO-RM"]
SEED_ORDER = ["42", "13", "3407", "mean"]
COLORS = {"DPO": "#4C78A8", "DDO-RM": "#F58518"}


def load_metric(path: Path, fallback: dict[str, float | int | str]) -> dict[str, float | int | str]:
    if path.exists():
        return json.loads(path.read_text())
    return fallback


def build_results_table() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method in METHOD_ORDER:
        for seed in [42, 13, 3407]:
            payload = load_metric(RESULT_FILES[method][seed], FALLBACK_RESULTS[method][seed])
            rows.append(
                {
                    "seed": str(seed),
                    "method": method,
                    "split": payload["split"],
                    "num_examples": int(payload["num_examples"]),
                    "pair_accuracy": float(payload["pair_accuracy"]),
                    "auc": float(payload["auc"]),
                    "mean_margin": float(payload["mean_margin"]),
                }
            )

    df = pd.DataFrame(rows)
    mean_rows = (
        df.groupby("method", as_index=False)[["num_examples", "pair_accuracy", "auc", "mean_margin"]]
        .mean(numeric_only=True)
        .assign(seed="mean", split="test_prefs")
    )
    mean_rows["num_examples"] = mean_rows["num_examples"].round().astype(int)

    combined = pd.concat([df, mean_rows], ignore_index=True)
    combined["seed"] = pd.Categorical(combined["seed"], categories=SEED_ORDER, ordered=True)
    combined["method"] = pd.Categorical(combined["method"], categories=METHOD_ORDER, ordered=True)
    combined = combined.sort_values(["seed", "method"]).reset_index(drop=True)
    return combined


def write_summary_csv(df: pd.DataFrame) -> None:
    export_df = df.copy()
    export_df["seed"] = export_df["seed"].astype(str)
    export_df.to_csv(SUMMARY_PATH, index=False, float_format="%.4f")


def metric_ylim(values: np.ndarray, metric_name: str) -> tuple[float, float]:
    if metric_name == "pair_accuracy":
        return 0.50, 0.60
    if metric_name == "auc":
        return 0.50, 0.55
    if metric_name == "mean_margin":
        return 0.0, 0.90
    vmin = float(values.min())
    vmax = float(values.max())
    return max(0.0, vmin), vmax


def annotate_bars(ax: plt.Axes, bars) -> None:
    ymin, ymax = ax.get_ylim()
    offset = (ymax - ymin) * 0.02
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + offset,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#1F2937",
        )


def plot_metric(df: pd.DataFrame, metric_name: str, title: str, output_path: Path) -> None:
    pivot = (
        df.pivot(index="seed", columns="method", values=metric_name)
        .reindex(SEED_ORDER)
        .loc[:, METHOD_ORDER]
    )
    values = pivot.to_numpy(dtype=float)

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )

    fig, ax = plt.subplots(figsize=(7.4, 4.6), constrained_layout=True)
    x = np.arange(len(SEED_ORDER))
    width = 0.34

    dpo_bars = ax.bar(x - width / 2, pivot["DPO"], width=width, color=COLORS["DPO"], label="DPO", zorder=3)
    ddorm_bars = ax.bar(x + width / 2, pivot["DDO-RM"], width=width, color=COLORS["DDO-RM"], label="DDO-RM", zorder=3)

    ax.set_title(title)
    ax.set_ylabel(title)
    ax.set_xticks(x, ["Seed 42", "Seed 13", "Seed 3407", "Mean"])
    ax.set_ylim(*metric_ylim(values, metric_name))
    ax.grid(axis="y", color="#D1D5DB", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9CA3AF")
    ax.spines["bottom"].set_color("#9CA3AF")
    ax.tick_params(colors="#374151")
    ax.legend(frameon=False, loc="upper left")

    annotate_bars(ax, dpo_bars)
    annotate_bars(ax, ddorm_bars)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_pair_accuracy_seed_lines(df: pd.DataFrame, output_path: Path) -> None:
    seed_df = df[df["seed"].astype(str) != "mean"].copy()
    x = np.arange(3)
    labels = ["Seed 42", "Seed 13", "Seed 3407"]

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    for method in METHOD_ORDER:
        y = seed_df[seed_df["method"] == method]["pair_accuracy"].to_numpy(dtype=float)
        ax.plot(
            x,
            y,
            marker="o",
            markersize=7,
            linewidth=2.2,
            color=COLORS[method],
            label=method,
        )
        for xi, yi in zip(x, y):
            ax.text(xi, yi + 0.002, f"{yi:.4f}", ha="center", va="bottom", fontsize=9, color="#1F2937")

    ax.set_title("Pair Accuracy Across Seeds")
    ax.set_ylabel("Pair Accuracy")
    ax.set_xticks(x, labels)
    ax.set_ylim(0.50, 0.60)
    ax.grid(axis="y", color="#D1D5DB", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9CA3AF")
    ax.spines["bottom"].set_color("#9CA3AF")
    ax.tick_params(colors="#374151")
    ax.legend(frameon=False, loc="upper left")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = build_results_table()
    write_summary_csv(df)
    for metric_name, title, output_path in METRICS:
        plot_metric(df, metric_name, title, output_path)
    plot_pair_accuracy_seed_lines(df, PAIR_LINES_PATH)
    print(f"Wrote {SUMMARY_PATH}")
    for _, _, output_path in METRICS:
        print(f"Wrote {output_path}")
    print(f"Wrote {PAIR_LINES_PATH}")


if __name__ == "__main__":
    main()
