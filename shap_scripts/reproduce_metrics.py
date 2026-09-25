import os

import pandas as pd
import torch
from common import (
    CATEGORIES,
    SEED,
    SPLIT_FAMILY,
    load_model,
    load_split,
    make_predict_proba,
)
from sklearn.metrics import precision_recall_fscore_support

RUNS_CSV = os.environ.get("SAROHEAD_RUNS_CSV", "../results/metrics_per_run.csv")


def expected_values():
    if not os.path.exists(RUNS_CSV):
        raise SystemExit(f"{RUNS_CSV} not found, run aggregate_results.py first")
    runs = pd.read_csv(RUNS_CSV, keep_default_na=False)
    runs = runs[
        (runs["method"] == "reptile")
        & (runs["model"] == "bert")
        & (runs["variant"] == "reptile")
        & (runs["split_family"] == SPLIT_FAMILY)
        & (runs["eval_split"] == "test")
        & (runs["seed"] == SEED)
        & runs["category"].isin(CATEGORIES)
    ]
    if len(runs) != len(CATEGORIES):
        raise SystemExit(f"no reptile/bert test run for seed {SEED} in {RUNS_CSV}")
    return {
        r["category"]: {
            "recall": round(100 * r["satire_recall"], 2),
            "precision": round(100 * r["satire_precision"], 2),
            "f1": round(100 * r["satire_f1"], 2),
        }
        for _, r in runs.iterrows()
    }


def main():
    reference = expected_values()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    rows = []
    for category in CATEGORIES:
        model, tokenizer = load_model(category, device)
        predict_proba = make_predict_proba(model, tokenizer, device)
        titles, labels = load_split(category, "test")
        probs = predict_proba(titles)
        preds = (probs >= 0.5).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary", pos_label=1
        )
        rows.append(
            {
                "category": category,
                "n_test": len(labels),
                "recall": round(100 * recall, 2),
                "precision": round(100 * precision, 2),
                "f1": round(100 * f1, 2),
                "reference_recall": reference[category]["recall"],
                "reference_precision": reference[category]["precision"],
                "reference_f1": reference[category]["f1"],
            }
        )
        print(rows[-1])
        del model
        torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/reproduced_metrics.csv", index=False)
    max_diff = (df["f1"] - df["reference_f1"]).abs().max()
    print(f"\nmax |F1 - reference F1| = {max_diff:.2f}")
    if max_diff > 1.0:
        raise SystemExit("metrics do not match the reference run")


if __name__ == "__main__":
    main()
