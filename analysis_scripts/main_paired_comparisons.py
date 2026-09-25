import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis_scripts.aggregate_results import CATEGORIES, binary_metrics, satire_f1_vec
from analysis_scripts.prediction_io import COLUMNS, predictions_root

DISPLAY = {
    "lr_anova_F1500": "Logistic Regression (ANOVA, 1{,}500 feat.)",
    "bert_None": "BERT (no TTL)",
    "bert_best_TTL": "BERT + best TTL",
    "reptile_bert": "Reptile-BERT",
    "lora_RoGemma": "LoRA RoGemma",
    "lora_RoLlama3": "LoRA RoLlama3",
    "few_shot_RoLlama2_n6": "RoLlama2, 6-shot",
    "few_shot_RoGemma_n6": "RoGemma, 6-shot",
    "all_satirical": "All-satirical baseline",
}


def load_matching(root, method, model, variant):
    files = []
    for category in CATEGORIES:
        for path in sorted(
            (root / method).glob(
                f"{method}__{model}__{variant}__{category}__seed*__random__test.csv"
            )
        ):
            parts = path.name.split("__")
            if len(parts) < 7:
                continue
            if parts[2] != variant or parts[3] != category:
                continue
            files.append(path)
    if not files:
        raise FileNotFoundError(f"no random/test files for {method}/{model}/{variant}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{method}/{model}/{variant}: missing columns {missing}")
    return df


def runs_by_seed(df, category):
    part = df if category is None else df[df["category"] == category]
    if category is None:
        part = part[part["category"].isin(CATEGORIES)].copy()
        part = part.sort_values(["seed", "category", "title"], kind="stable")
    else:
        part = part.sort_values(["seed", "title"], kind="stable")
    return {int(s): g.reset_index(drop=True) for s, g in part.groupby("seed")}


def mean_f1(runs):
    return float(
        np.mean(
            [
                binary_metrics(g["ground_truth"], g["prediction"])["satire_f1"]
                for g in runs.values()
            ]
        )
    )


def paired_boot(a_runs, b_runs, n_boot, rng):
    common = sorted(set(a_runs) & set(b_runs))
    if not common:
        raise ValueError("no shared seeds")
    n = min(*(len(a_runs[s]) for s in common), *(len(b_runs[s]) for s in common))
    idx = rng.integers(0, n, size=(n_boot, n))
    a_f1 = np.mean(
        [
            satire_f1_vec(
                a_runs[s]["ground_truth"].to_numpy(int)[:n],
                a_runs[s]["prediction"].to_numpy(int)[:n],
                idx,
            )
            for s in common
        ],
        axis=0,
    )
    b_f1 = np.mean(
        [
            satire_f1_vec(
                b_runs[s]["ground_truth"].to_numpy(int)[:n],
                b_runs[s]["prediction"].to_numpy(int)[:n],
                idx,
            )
            for s in common
        ],
        axis=0,
    )
    diff = a_f1 - b_f1
    point = mean_f1({s: a_runs[s].iloc[:n] for s in common}) - mean_f1(
        {s: b_runs[s].iloc[:n] for s in common}
    )
    return (
        point,
        float(np.percentile(diff, 2.5)),
        float(np.percentile(diff, 97.5)),
        float(np.mean(diff <= 0)),
    )


def all_satirical_runs(template):
    out = {}
    for s, g in template.items():
        h = g.copy()
        h["prediction"] = 1
        out[s] = h
    return out


def best_bert_ttl_variant(root):
    scores = {}
    for v in ["SciTechBaitRo", "SaRoCo", "RoCliCo"]:
        df = load_matching(root, "bert", "bert", v)
        scores[v] = mean_f1(runs_by_seed(df, None))
    return max(scores, key=scores.get)


def compute_comparisons(root, n_boot, seed):
    rng = np.random.default_rng(seed)
    best_ttl = best_bert_ttl_variant(root)

    configs = {
        "lr_anova_F1500": load_matching(root, "classic_ml", "lr", "anova_F1500"),
        "bert_None": load_matching(root, "bert", "bert", "None"),
        "bert_best_TTL": load_matching(root, "bert", "bert", best_ttl),
        "reptile_bert": load_matching(root, "reptile", "bert", "reptile"),
        "lora_RoGemma": load_matching(root, "lora", "RoGemma", "lora"),
        "lora_RoLlama3": load_matching(root, "lora", "RoLlama3", "lora"),
        "few_shot_RoLlama2_n6": load_matching(root, "few_shot", "RoLlama2", "n6"),
        "few_shot_RoGemma_n6": load_matching(root, "few_shot", "RoGemma", "n6"),
    }

    pairs = [
        ("bert_None", "lr_anova_F1500", "BERT vs classic ML"),
        ("bert_best_TTL", "bert_None", "TTL vs no TTL"),
        ("reptile_bert", "bert_best_TTL", "Reptile vs best TTL"),
        ("lora_RoGemma", "reptile_bert", "LoRA vs Reptile"),
        ("lora_RoLlama3", "reptile_bert", "LoRA vs Reptile"),
        ("lora_RoGemma", "lora_RoLlama3", "LoRA ranking"),
        ("few_shot_RoLlama2_n6", "all_satirical", "Few-shot vs trivial"),
        ("few_shot_RoGemma_n6", "all_satirical", "Few-shot vs trivial"),
    ]

    scopes = [("all_pooled", None)] + [(c, c) for c in CATEGORIES]
    rows = []
    for comp_key, base_key, family in pairs:
        for scope_name, cat in scopes:
            a = runs_by_seed(configs[comp_key], cat)
            if base_key == "all_satirical":
                b = all_satirical_runs(a)
            else:
                b = runs_by_seed(configs[base_key], cat)
            point, lo, hi, p_le0 = paired_boot(a, b, n_boot, rng)
            rows.append(
                {
                    "comparison_family": family,
                    "competitor": comp_key,
                    "baseline": base_key,
                    "baseline_variant_note": (
                        best_ttl if "TTL" in base_key or "TTL" in comp_key else ""
                    ),
                    "best_ttl_variant": best_ttl,
                    "scope": scope_name,
                    "diff_satire_f1": point,
                    "ci_low": lo,
                    "ci_high": hi,
                    "p_diff_le_0": p_le0,
                    "significant": lo > 0 or hi < 0,
                }
            )
    return pd.DataFrame(rows)


def latex_table(df):
    pooled = df[df["scope"] == "all_pooled"].copy()
    lines = [
        r"\begin{tabularx}{\textwidth}{llX}",
        r"\toprule",
        r"\textbf{Competitor} & \textbf{Baseline} & \textbf{$\Delta$ F1 [95\% CI]} \\",
        r"\midrule",
    ]
    for _, r in pooled.iterrows():
        d = 100 * r["diff_satire_f1"]
        lo = 100 * r["ci_low"]
        hi = 100 * r["ci_high"]
        star = "$^{*}$" if r["significant"] else ""
        sign = "+" if d >= 0 else ""
        cell = f"{sign}{d:.2f} [{lo:.2f}, {hi:.2f}]{star}"
        lines.append(
            f"{DISPLAY[r['competitor']]} & {DISPLAY[r['baseline']]} & {cell} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabularx}"]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--out_csv", default="results/main_paired_comparisons.csv")
    parser.add_argument("--tables_dir", default="results/tables")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    root = Path(args.predictions) if args.predictions else predictions_root()
    df = compute_comparisons(root, args.bootstrap, args.seed)
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {out} ({len(df)} rows)")
    print(
        df[df["scope"] == "all_pooled"][
            [
                "competitor",
                "baseline",
                "diff_satire_f1",
                "ci_low",
                "ci_high",
                "significant",
            ]
        ].to_string(index=False)
    )

    tables = Path(args.tables_dir)
    tables.mkdir(parents=True, exist_ok=True)
    tex_path = tables / "main_paired_comparisons.tex"
    tex_path.write_text(latex_table(df) + "\n", encoding="utf-8")
    print(f"wrote {tex_path}")
    print(f"best TTL variant: {df['best_ttl_variant'].iloc[0]}")


if __name__ == "__main__":
    main()
