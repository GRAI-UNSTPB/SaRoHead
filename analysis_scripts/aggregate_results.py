import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis_scripts.prediction_io import load_all_predictions

CONFIG_KEYS = ["method", "model", "variant", "category", "split_family", "eval_split"]
RUN_KEYS = CONFIG_KEYS + ["seed"]
CATEGORIES = ["social", "politic", "sport"]
METRICS = [
    "satire_precision",
    "satire_recall",
    "satire_f1",
    "mainstream_precision",
    "mainstream_recall",
    "mainstream_f1",
    "macro_f1",
    "accuracy",
    "fpr",
]


def binary_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    def prf(tp_, fp_, fn_):
        p = tp_ / (tp_ + fp_) if tp_ + fp_ else 0.0
        r = tp_ / (tp_ + fn_) if tp_ + fn_ else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        return p, r, f

    sp, sr, sf = prf(tp, fp, fn)
    mp, mr, mf = prf(tn, fn, fp)
    n = len(y_true)
    return {
        "satire_precision": sp,
        "satire_recall": sr,
        "satire_f1": sf,
        "mainstream_precision": mp,
        "mainstream_recall": mr,
        "mainstream_f1": mf,
        "macro_f1": (sf + mf) / 2,
        "accuracy": (tp + tn) / n if n else 0.0,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "n": n,
        "n_satire": int(np.sum(y_true == 1)),
        "n_mainstream": int(np.sum(y_true == 0)),
    }


def satire_f1_vec(y_true, y_pred, idx):
    t = y_true[idx]
    p = y_pred[idx]
    tp = np.sum((t == 1) & (p == 1), axis=1)
    fp = np.sum((t == 0) & (p == 1), axis=1)
    fn = np.sum((t == 1) & (p == 0), axis=1)
    denom = 2 * tp + fp + fn
    return np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)


def macro_f1_vec(y_true, y_pred, idx):
    sat = satire_f1_vec(y_true, y_pred, idx)
    main = satire_f1_vec(1 - y_true, 1 - y_pred, idx)
    return (sat + main) / 2


def per_run_metrics(pred):
    rows = []
    for keys, grp in pred.groupby(RUN_KEYS, sort=False):
        m = binary_metrics(grp["ground_truth"].to_numpy(), grp["prediction"].to_numpy())
        rows.append({**dict(zip(RUN_KEYS, keys)), **m})
    return pd.DataFrame(rows)


def add_pooled_runs(pred):
    keys = [k for k in RUN_KEYS if k != "category"]
    pooled = []
    for k, grp in pred.groupby(keys, sort=False):
        if grp["category"].nunique() < len(CATEGORIES):
            continue
        g = grp.copy()
        g["category"] = "all_pooled"
        pooled.append(g)
    return pd.concat([pred, *pooled], ignore_index=True) if pooled else pred


def bootstrap_ci(pred, n_boot, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for keys, grp in pred.groupby(CONFIG_KEYS, sort=False):
        seeds = sorted(grp["seed"].unique())
        per_seed = []
        for s in seeds:
            g = grp[grp["seed"] == s].sort_values("title", kind="stable")
            per_seed.append(
                (g["ground_truth"].to_numpy(int), g["prediction"].to_numpy(int))
            )
        n = min(len(t) for t, _ in per_seed)
        if n == 0:
            continue
        idx = rng.integers(0, n, size=(n_boot, n))
        sat = np.mean([satire_f1_vec(t[:n], p[:n], idx) for t, p in per_seed], axis=0)
        mac = np.mean([macro_f1_vec(t[:n], p[:n], idx) for t, p in per_seed], axis=0)
        rows.append(
            {
                **dict(zip(CONFIG_KEYS, keys)),
                "satire_f1_ci_low": float(np.percentile(sat, 2.5)),
                "satire_f1_ci_high": float(np.percentile(sat, 97.5)),
                "macro_f1_ci_low": float(np.percentile(mac, 2.5)),
                "macro_f1_ci_high": float(np.percentile(mac, 97.5)),
            }
        )
    return pd.DataFrame(rows)


def summarize(runs):
    agg = runs.groupby(CONFIG_KEYS, sort=False)
    out = agg[METRICS].agg(["mean", "std"])
    out.columns = [f"{m}_{s}" for m, s in out.columns]
    out["n_seeds"] = agg["seed"].nunique()
    out["n_items"] = agg["n"].first()
    out = out.reset_index()
    cat_runs = runs[runs["category"].isin(CATEGORIES)]
    keys = [k for k in RUN_KEYS if k != "category"]
    complete = cat_runs.groupby(keys)["category"].transform("nunique") == len(
        CATEGORIES
    )
    per_seed = (
        cat_runs[complete].groupby(keys, sort=False)[METRICS].mean().reset_index()
    )
    if len(per_seed):
        g = per_seed.groupby([k for k in CONFIG_KEYS if k != "category"], sort=False)
        allu = g[METRICS].agg(["mean", "std"])
        allu.columns = [f"{m}_{s}" for m, s in allu.columns]
        allu["n_seeds"] = g["seed"].nunique()
        allu["n_items"] = np.nan
        allu = allu.reset_index()
        allu["category"] = "all_unweighted"
        out = pd.concat([out, allu[out.columns]], ignore_index=True)
    return out


def paired_differences(pred, n_boot, seed):
    rng = np.random.default_rng(seed)
    rows = []
    test = pred[(pred["eval_split"] == "test") & pred["category"].isin(CATEGORIES)]
    reptile = test[(test["method"] == "reptile")]
    bert = test[(test["method"] == "bert")]
    if reptile.empty or bert.empty:
        return pd.DataFrame()
    for (model, category, family, variant_r), rg in reptile.groupby(
        ["model", "category", "split_family", "variant"]
    ):
        cand = bert[
            (bert["model"] == model)
            & (bert["category"] == category)
            & (bert["split_family"] == family)
        ]
        if cand.empty:
            continue
        f1_by_variant = {
            v: np.mean(
                [
                    binary_metrics(g["ground_truth"], g["prediction"])["satire_f1"]
                    for _, g in cg.groupby("seed")
                ]
            )
            for v, cg in cand.groupby("variant")
        }
        targets = {
            "None": [v for v in f1_by_variant if v.startswith("None")],
            "best_TTL": [max(f1_by_variant, key=f1_by_variant.get)],
        }
        for label, variants in targets.items():
            if not variants:
                continue
            bg = cand[cand["variant"] == variants[0]]
            r_runs = {
                s: g.sort_values("title", kind="stable") for s, g in rg.groupby("seed")
            }
            b_runs = {
                s: g.sort_values("title", kind="stable") for s, g in bg.groupby("seed")
            }
            n = min(
                min(len(g) for g in r_runs.values()),
                min(len(g) for g in b_runs.values()),
            )
            idx = rng.integers(0, n, size=(n_boot, n))
            r_f1 = np.mean(
                [
                    satire_f1_vec(
                        g["ground_truth"].to_numpy(int)[:n],
                        g["prediction"].to_numpy(int)[:n],
                        idx,
                    )
                    for g in r_runs.values()
                ],
                axis=0,
            )
            b_f1 = np.mean(
                [
                    satire_f1_vec(
                        g["ground_truth"].to_numpy(int)[:n],
                        g["prediction"].to_numpy(int)[:n],
                        idx,
                    )
                    for g in b_runs.values()
                ],
                axis=0,
            )
            diff = r_f1 - b_f1
            point = np.mean(
                [
                    binary_metrics(g["ground_truth"], g["prediction"])["satire_f1"]
                    for g in r_runs.values()
                ]
            ) - np.mean(
                [
                    binary_metrics(g["ground_truth"], g["prediction"])["satire_f1"]
                    for g in b_runs.values()
                ]
            )
            rows.append(
                {
                    "model": model,
                    "category": category,
                    "split_family": family,
                    "reptile_variant": variant_r,
                    "baseline": label,
                    "baseline_variant": variants[0],
                    "diff_satire_f1": point,
                    "ci_low": float(np.percentile(diff, 2.5)),
                    "ci_high": float(np.percentile(diff, 97.5)),
                    "p_diff_le_0": float(np.mean(diff <= 0)),
                }
            )
    return pd.DataFrame(rows)


def fmt(mean, std, pct=True):
    if pd.isna(mean):
        return "--"
    scale = 100 if pct else 1
    if pd.isna(std) or std == 0:
        return f"{mean * scale:.2f}"
    return f"{mean * scale:.2f} $\\pm$ {std * scale:.2f}"


def latex_rows(
    summary, row_keys, metrics=("satire_recall", "satire_precision", "satire_f1")
):
    lines = []
    for _, r in summary.iterrows():
        cells = [str(r[k]) for k in row_keys] + [
            fmt(r[f"{m}_mean"], r[f"{m}_std"]) for m in metrics
        ]
        lines.append(" & ".join(cells) + " \\\\")
    return "\n".join(lines)


def reaverage(runs, keys):
    per_seed = runs.groupby(keys + ["seed"], sort=False)[METRICS].mean().reset_index()
    g = per_seed.groupby(keys, sort=False)
    out = g[METRICS].agg(["mean", "std"])
    out.columns = [f"{m}_{s}" for m, s in out.columns]
    out["n_seeds"] = g["seed"].nunique()
    return out.reset_index()


def export_tables(runs, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    plain = runs[
        (runs["eval_split"] == "test")
        & runs["category"].isin(CATEGORIES)
        & (runs["split_family"] == "random")
    ]

    def save(name, body):
        (out_dir / f"{name}.tex").write_text(body + "\n", encoding="utf-8")
        print(f"wrote {out_dir / name}.tex")

    ml = plain[plain["method"] == "classic_ml"].copy()
    if not ml.empty:
        ml["selector"] = (
            ml["variant"]
            .str.extract(r"^(anova|chi2|mi)_")[0]
            .str.upper()
            .replace({"ANOVA": "ANOVA", "CHI2": "CHI2", "MI": "MI"})
        )
        ml["num_features"] = ml["variant"].str.extract(r"_F(\d+)")[0].astype(int)
        ml["model"] = ml["model"].str.upper()
        t = reaverage(ml, ["model", "num_features"]).sort_values(
            ["model", "num_features"]
        )
        save("ml_algorithm_features", latex_rows(t, ["model", "num_features"]))
        t = reaverage(ml, ["model", "selector"]).sort_values(["model", "selector"])
        save("ml_algorithm_selector", latex_rows(t, ["model", "selector"]))
        t = reaverage(ml, ["category", "model"])
        t["category"] = pd.Categorical(t["category"], CATEGORIES)
        save(
            "ml_category_algorithm",
            latex_rows(t.sort_values(["category", "model"]), ["category", "model"]),
        )

    bert = plain[plain["method"] == "bert"]
    if not bert.empty:
        t = reaverage(bert, ["model", "variant"]).sort_values(["model", "variant"])
        save("bert_model_intermediate", latex_rows(t, ["model", "variant"]))
        t = reaverage(bert, ["category", "model"])
        t["category"] = pd.Categorical(t["category"], CATEGORIES)
        save(
            "bert_category_model",
            latex_rows(t.sort_values(["category", "model"]), ["category", "model"]),
        )

    rep = plain[plain["method"] == "reptile"]
    if not rep.empty:
        save("reptile_model", latex_rows(reaverage(rep, ["model"]), ["model"]))
        t = reaverage(rep, ["category", "model"])
        t["category"] = pd.Categorical(t["category"], CATEGORIES)
        save(
            "reptile_category_model",
            latex_rows(t.sort_values(["category", "model"]), ["category", "model"]),
        )

    fs = plain[plain["method"] == "few_shot"].copy()
    if not fs.empty:
        fs["n_shot"] = fs["variant"].str.extract(r"^n(\d+)")[0].astype(int)
        t = reaverage(fs, ["n_shot", "model"]).sort_values(["n_shot", "model"])
        save("fewshot_nshot_model", latex_rows(t, ["n_shot", "model"]))
        t = reaverage(fs[fs["n_shot"] > 0], ["category", "model"])
        t["category"] = pd.Categorical(t["category"], CATEGORIES)
        save(
            "fewshot_category_model",
            latex_rows(t.sort_values(["category", "model"]), ["category", "model"]),
        )
    lora = plain[plain["method"] == "lora"]
    if not lora.empty:
        save("lora_model", latex_rows(reaverage(lora, ["model"]), ["model"]))
        t = reaverage(lora, ["category", "model"])
        t["category"] = pd.Categorical(t["category"], CATEGORIES)
        save(
            "lora_category_model",
            latex_rows(t.sort_values(["category", "model"]), ["category", "model"]),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--out_dir", default="results")
    parser.add_argument("--tables_dir", default="results/tables")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    pred = load_all_predictions(Path(args.predictions) if args.predictions else None)
    pred = add_pooled_runs(pred)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    runs = per_run_metrics(pred)
    runs.to_csv(out / "metrics_per_run.csv", index=False)
    print(
        f"{len(runs)} runs from {pred[RUN_KEYS].drop_duplicates().shape[0]} prediction files"
    )

    summary = summarize(runs)
    ci = bootstrap_ci(pred, args.bootstrap, args.seed)
    if not ci.empty:
        summary = summary.merge(ci, on=CONFIG_KEYS, how="left")
    summary.to_csv(out / "summary.csv", index=False)
    print(f"wrote {out / 'summary.csv'} ({len(summary)} configurations)")

    diffs = paired_differences(pred, args.bootstrap, args.seed)
    if not diffs.empty:
        diffs.to_csv(out / "paired_differences.csv", index=False)
        print(f"wrote {out / 'paired_differences.csv'}")

    export_tables(runs, Path(args.tables_dir))


if __name__ == "__main__":
    main()
