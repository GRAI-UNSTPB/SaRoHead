import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CATEGORIES = ["social", "politic", "sport"]
MODEL_LABEL = {
    "bert": "BERT",
    "distilled-bert": "Distilled-BERT",
    "xlm-roberta": "XLM-RoBERTa",
}
BERT_VARIANTS = [
    ("None", "No intermediate task"),
    ("SaRoCo", "TTL: SaRoCo"),
    ("RoCliCo", "TTL: RoCliCo"),
    ("SciTechBaitRo", "TTL: SciTechBaitRo"),
    ("reptile", "Reptile"),
]


def reptile_vs_ttl(runs, out_dir):
    df = runs[
        (runs["split_family"] == "random")
        & (runs["eval_split"] == "test")
        & runs["category"].isin(CATEGORIES)
    ]
    df = df[df["method"].isin(["bert", "reptile"])]
    per_seed = (
        df.groupby(["model", "variant", "seed"])["satire_f1"].mean().reset_index()
    )
    stats = (
        per_seed.groupby(["model", "variant"])["satire_f1"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    models = [m for m in MODEL_LABEL if m in set(stats["model"])]
    width = 0.16
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    x = np.arange(len(models))
    for i, (variant, label) in enumerate(BERT_VARIANTS):
        means, stds = [], []
        for m in models:
            row = stats[(stats["model"] == m) & (stats["variant"] == variant)]
            means.append(100 * row["mean"].iloc[0] if len(row) else np.nan)
            stds.append(100 * row["std"].iloc[0] if len(row) else 0.0)
        bars = ax.bar(
            x + (i - 2) * width,
            means,
            width,
            yerr=stds,
            capsize=2.5,
            label=label,
            error_kw={"lw": 0.8},
        )
        for b, mval in zip(bars, means):
            if not np.isnan(mval):
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    60.5,
                    f"{mval:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                    rotation=90,
                    color="white",
                )
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABEL[m] for m in models])
    ax.set_ylabel("Satire F1 (%), test set")
    ax.set_ylim(60, 95)
    ax.legend(fontsize=8, ncol=3, loc="upper right", frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(
            out_dir / f"test_set_Reptile_vs_UTTL.{ext}", bbox_inches="tight", dpi=200
        )
    plt.close(fig)
    stats["variant"] = pd.Categorical(stats["variant"], [v for v, _ in BERT_VARIANTS])
    return stats.sort_values(["model", "variant"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    args = parser.parse_args()
    results = Path(args.results)
    plots = results / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    runs = pd.read_csv(results / "metrics_per_run.csv", keep_default_na=False)
    stats = reptile_vs_ttl(runs, plots)
    stats.to_csv(results / "figure_reptile_vs_ttl.csv", index=False)
    print(stats.to_string(index=False))
    print(f"wrote {plots / 'test_set_Reptile_vs_UTTL.pdf'}")


if __name__ == "__main__":
    main()
