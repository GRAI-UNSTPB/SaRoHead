import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis_scripts.prediction_io import read_split, source_column, write_predictions

CATEGORIES = ["social", "politic", "sport"]
ALGORITHMS = ["svm", "rf", "nb", "lr"]
SELECTORS = ["anova", "chi2", "mi"]
NUM_FEATURES = [500, 1000, 1500, 3000, 5000, 7000]


def build_classifier(name, seed):
    if name == "svm":
        return SVC(C=1.0, kernel="rbf", random_state=seed)
    if name == "rf":
        return RandomForestClassifier(criterion="gini", random_state=seed, n_jobs=-1)
    if name == "nb":
        return MultinomialNB(alpha=1.0)
    if name == "lr":
        return LogisticRegression(C=1.0, max_iter=2000, random_state=seed)
    raise ValueError(name)


def feature_scores(name, X, y, seed):
    if name == "anova":
        scores, _ = f_classif(X, y)
    elif name == "chi2":
        scores, _ = chi2(X, y)
    elif name == "mi":
        scores = mutual_info_classif(X, y, discrete_features=True, random_state=seed)
    else:
        raise ValueError(name)
    return np.nan_to_num(np.asarray(scores), nan=-np.inf)


def satire_scores(clf, X):
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[:, 1]
    return clf.decision_function(X)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--split_family", default="random")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--categories", nargs="+", default=CATEGORIES)
    parser.add_argument("--algorithms", nargs="+", default=ALGORITHMS)
    parser.add_argument("--selectors", nargs="+", default=SELECTORS)
    parser.add_argument("--num_features", nargs="+", type=int, default=NUM_FEATURES)
    parser.add_argument("--text_column", default="new_title")
    args = parser.parse_args()

    splits = {
        name: read_split(args.data_dir, name) for name in ["train", "val", "test"]
    }

    for category in args.categories:
        parts = {
            k: v[v["category"] == category].reset_index(drop=True)
            for k, v in splits.items()
        }
        vectorizer = CountVectorizer(ngram_range=(1, 3), min_df=2)
        X_train = vectorizer.fit_transform(parts["train"][args.text_column].fillna(""))
        y_train = parts["train"]["satiric"].to_numpy()
        X_eval = {
            k: vectorizer.transform(parts[k][args.text_column].fillna(""))
            for k in ["val", "test"]
        }
        print(f"{category} vocabulary size: {X_train.shape[1]}", flush=True)

        for selector in args.selectors:
            t0 = time.time()
            scores = feature_scores(selector, X_train, y_train, args.seed)
            ranking = np.argsort(-scores, kind="stable")
            print(
                f"{category} {selector} scores in {time.time() - t0:.1f}s", flush=True
            )
            for num_features in args.num_features:
                keep = ranking[: min(num_features, len(ranking))]
                Xtr = X_train[:, keep]
                for algorithm in args.algorithms:
                    clf = build_classifier(algorithm, args.seed)
                    clf.fit(Xtr, y_train)
                    for eval_split in ["val", "test"]:
                        Xev = X_eval[eval_split][:, keep]
                        pred = clf.predict(Xev)
                        write_predictions(
                            method="classic_ml",
                            model=algorithm,
                            variant=f"{selector}_F{num_features}",
                            category=category,
                            seed=args.seed,
                            split_family=args.split_family,
                            eval_split=eval_split,
                            titles=parts[eval_split][args.text_column].fillna(""),
                            ground_truth=parts[eval_split]["satiric"],
                            prediction=pred,
                            score=satire_scores(clf, Xev),
                            source=source_column(parts[eval_split]),
                        )
                print(f"{category} {selector} F={num_features} done", flush=True)


if __name__ == "__main__":
    main()
