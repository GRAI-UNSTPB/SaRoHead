import os
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from topic_common import CATEGORIES, load_full_dataset, romanian_stopwords

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis_scripts"),
)
from unmasked_entities import is_unmasked_entity

TOP_K = 15
TABLE_K = 8


def main():
    os.makedirs("results", exist_ok=True)
    df = load_full_dataset()
    stopwords = romanian_stopwords()

    rows = []
    for category in CATEGORIES:
        subset = df[df["category"] == category]
        vectorizer = TfidfVectorizer(stop_words=stopwords, min_df=5, sublinear_tf=True)
        matrix = vectorizer.fit_transform(subset["new_title"])
        terms = np.array(vectorizer.get_feature_names_out())
        for label, class_name in [(1, "satirical"), (0, "mainstream")]:
            mask = (subset["satiric"] == label).to_numpy()
            mean_tfidf = np.asarray(matrix[mask].mean(axis=0)).ravel()
            order = np.argsort(mean_tfidf)[::-1][:TOP_K]
            for rank, term_idx in enumerate(order, start=1):
                rows.append(
                    {
                        "category": category,
                        "class": class_name,
                        "rank": rank,
                        "term": terms[term_idx],
                        "mean_tfidf": round(float(mean_tfidf[term_idx]), 4),
                        "unmasked_entity": is_unmasked_entity(terms[term_idx]),
                    }
                )

    out = pd.DataFrame(rows)
    out.to_csv("results/tfidf_top_terms.csv", index=False)
    for category in CATEGORIES:
        for class_name in ["satirical", "mainstream"]:
            rows_ = out[(out.category == category) & (out["class"] == class_name)]
            top = rows_[~rows_["unmasked_entity"]].head(TABLE_K)
            excluded = rows_[rows_["unmasked_entity"]]["term"].tolist()
            print(
                f"{category} / {class_name}: {', '.join(top['term'])}"
                + (f"   [excluded: {', '.join(excluded)}]" if excluded else "")
            )


if __name__ == "__main__":
    main()
