import os
from pathlib import Path

import pandas as pd

COLUMNS = [
    "method",
    "model",
    "variant",
    "category",
    "seed",
    "split_family",
    "eval_split",
    "title",
    "source",
    "ground_truth",
    "prediction",
    "score",
]

SPLIT_FILES = {"train": "train.csv", "val": "validation.csv", "test": "test.csv"}
TEXT_COLUMN = "new_title"
RELEASED_TEXT_COLUMN = "proc_title"


def split_path(data_dir, split):
    return Path(data_dir) / SPLIT_FILES[split]


def read_split(data_dir, split, category=None):
    df = pd.read_csv(split_path(data_dir, split))
    if TEXT_COLUMN not in df.columns and RELEASED_TEXT_COLUMN in df.columns:
        df = df.rename(columns={RELEASED_TEXT_COLUMN: TEXT_COLUMN})
    if category is not None and category != "all":
        df = df[df["category"] == category]
    return df.reset_index(drop=True)


def predictions_root(start=None):
    env = os.environ.get("SAROHEAD_RESULTS")
    if env:
        root = Path(env)
    else:
        here = Path(start or os.getcwd()).resolve()
        root = None
        for candidate in [here, *here.parents]:
            if (candidate / "data").is_dir() and (
                candidate / "analysis_scripts"
            ).is_dir():
                root = candidate / "results" / "predictions"
                break
        if root is None:
            root = here / "results" / "predictions"
    root.mkdir(parents=True, exist_ok=True)
    return root


def prediction_filename(
    method, model, variant, category, seed, split_family, eval_split
):
    safe = lambda s: str(s).replace("/", "-").replace(" ", "_")
    return f"{safe(method)}__{safe(model)}__{safe(variant)}__{safe(category)}__seed{int(seed)}__{safe(split_family)}__{safe(eval_split)}.csv"


def write_predictions(
    *,
    method,
    model,
    variant,
    category,
    seed,
    split_family,
    eval_split,
    titles,
    ground_truth,
    prediction,
    score=None,
    source=None,
    root=None,
):
    titles = list(titles)
    n = len(titles)
    df = pd.DataFrame(
        {
            "method": method,
            "model": model,
            "variant": variant,
            "category": category,
            "seed": int(seed),
            "split_family": split_family,
            "eval_split": eval_split,
            "title": titles,
            "source": list(source) if source is not None else ["unknown"] * n,
            "ground_truth": [int(x) for x in ground_truth],
            "prediction": [int(x) for x in prediction],
            "score": list(score) if score is not None else [float("nan")] * n,
        }
    )[COLUMNS]
    root = root or predictions_root()
    sub = root / method
    sub.mkdir(parents=True, exist_ok=True)
    path = sub / prediction_filename(
        method, model, variant, category, seed, split_family, eval_split
    )
    df.to_csv(path, index=False)
    return path


def load_all_predictions(root=None):
    root = root or predictions_root()
    files = sorted(root.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No prediction files under {root}")
    frames = [pd.read_csv(f, keep_default_na=False, na_values=[""]) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    return df


def source_column(df):
    if "source" in df.columns:
        return df["source"].fillna("unknown")
    return pd.Series(["unknown"] * len(df), index=df.index)
