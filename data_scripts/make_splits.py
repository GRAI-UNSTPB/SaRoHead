import argparse
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

CATEGORIES = ["social", "politic", "sport"]
OUTPUT_FILES = {"train": "train.csv", "val": "validation.csv", "test": "test.csv"}
PUBLIC_COLUMNS = ["title", "proc_title", "category", "satiric"]


def normalize(text):
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    text = text.lower()
    text = re.sub(r"\[[a-z_]+\]", " ", text)
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def near_duplicate_groups(texts, threshold, chunk=1500):
    vec = CountVectorizer(
        analyzer="char", ngram_range=(5, 5), binary=True, dtype=np.float32
    )
    X = vec.fit_transform(texts.tolist()).tocsr()
    sizes = np.asarray(X.sum(axis=1)).ravel()
    n = X.shape[0]
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    Xt = X.T.tocsc()
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        inter = (X[start:stop] @ Xt).toarray()
        union_sizes = sizes[start:stop, None] + sizes[None, :] - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            jacc = np.where(union_sizes > 0, inter / union_sizes, 0.0)
        rows, cols = np.nonzero(jacc >= threshold)
        for r, c in zip(rows, cols):
            i, j = start + r, c
            if i < j:
                union(i, j)
    return np.array([find(i) for i in range(n)])


def stratified_three_way(df, seed, val_frac, test_frac):
    strata = df["category"] + "_" + df["satiric"].astype(str)
    idx_rest, idx_test = train_test_split(
        df.index, test_size=test_frac, stratify=strata, random_state=seed
    )
    rel_val = val_frac / (1.0 - test_frac)
    idx_train, idx_val = train_test_split(
        idx_rest, test_size=rel_val, stratify=strata.loc[idx_rest], random_state=seed
    )
    split = pd.Series(index=df.index, dtype=object)
    split.loc[idx_train] = "train"
    split.loc[idx_val] = "val"
    split.loc[idx_test] = "test"
    return split


def count_table(df, split):
    tab = pd.crosstab([df["category"], df["satiric"]], split)
    tab = tab.reindex(columns=["train", "val", "test"], fill_value=0)
    tab["total"] = tab.sum(axis=1)
    return tab


def md_table(tab):
    tab = tab.reset_index()
    header = "| " + " | ".join(str(c) for c in tab.columns) + " |"
    sep = "|" + "---|" * len(tab.columns)
    rows = [
        "| " + " | ".join(str(v) for v in r) + " |" for r in tab.itertuples(index=False)
    ]
    return "\n".join([header, sep, *rows])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out_dir", default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--test_frac", type=float, default=0.15)
    parser.add_argument("--near_dup_threshold", type=float, default=0.8)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    frames = []
    for path in args.inputs:
        part = pd.read_csv(path)
        if "proc_title" in part.columns and "new_title" not in part.columns:
            part = part.rename(columns={"proc_title": "new_title"})
        frames.append(part[["title", "new_title", "category", "satiric"]])
    df = pd.concat(frames, ignore_index=True)
    n_raw = len(df)
    report = [
        f"# SaRoHead split report\n",
        f"Seed: {args.seed}\n",
        f"Rows loaded: {n_raw}\n",
    ]

    df["_norm_title"] = df["title"].map(normalize)
    exact_dup = df.duplicated("_norm_title", keep="first")
    n_exact = int(exact_dup.sum())
    df = df[~exact_dup].copy()

    df["_norm_new_title"] = df["new_title"].map(normalize)
    df["_dup_group"] = near_duplicate_groups(
        df["_norm_new_title"], args.near_dup_threshold
    )
    grp_sizes = df["_dup_group"].map(df["_dup_group"].value_counts())
    in_group = grp_sizes > 1
    n_near_groups = int(df.loc[in_group, "_dup_group"].nunique())
    n_near_rows = int(in_group.sum())
    label_conflict = df.groupby("_dup_group")["satiric"].transform("nunique") > 1
    n_conflict_groups = int(df.loc[label_conflict, "_dup_group"].nunique())
    n_conflict_rows = int(label_conflict.sum())

    keep_first = ~df.duplicated("_dup_group", keep="first")
    df = df[keep_first & ~label_conflict].copy()
    n_clean = len(df)

    report += [
        "## Duplicate removal\n",
        f"- Exact duplicates (normalized `title`): {n_exact} rows removed",
        f"- Near-duplicate groups (masked headline, char 5-gram Jaccard >= {args.near_dup_threshold}): {n_near_groups} groups covering {n_near_rows} rows",
        f"  - groups with conflicting labels (all members removed): {n_conflict_groups} groups, {n_conflict_rows} rows",
        f"- Rows kept after de-duplication: {n_clean} (removed {n_raw - n_clean})\n",
    ]

    df = df.drop(columns=["_norm_title", "_norm_new_title", "_dup_group"]).reset_index(
        drop=True
    )

    split = stratified_three_way(df, args.seed, args.val_frac, args.test_frac)
    public = df.rename(columns={"new_title": "proc_title"})[PUBLIC_COLUMNS]
    for name, filename in OUTPUT_FILES.items():
        public[split == name].to_csv(out / filename, index=False)
    report += [
        "## Stratified split (category x label, 70/15/15)\n",
        md_table(count_table(df, split)),
        "",
    ]

    (out / "split_report.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
