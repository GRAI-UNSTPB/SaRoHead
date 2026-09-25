import os

import numpy as np
import pandas as pd
import torch
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from topic_common import (
    PLOTS_DIR,
    load_full_dataset,
    plot_satire_share,
    romanian_stopwords,
)
from umap import UMAP

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MIN_TOPIC_SIZE = 100
SEED = 42


def fit_bertopic(docs):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st_model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    embeddings = st_model.encode(docs, batch_size=128, show_progress_bar=True)

    umap_model = UMAP(
        n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=SEED
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=MIN_TOPIC_SIZE,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    vectorizer_model = CountVectorizer(stop_words=romanian_stopwords(), min_df=5)
    topic_model = BERTopic(
        embedding_model=st_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        top_n_words=10,
        verbose=True,
    )
    topics, _ = topic_model.fit_transform(docs, embeddings)
    return topic_model, np.array(topics)


def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    df = load_full_dataset()
    docs = df["new_title"].tolist()

    topic_model, topics = fit_bertopic(docs)
    df["topic"] = topics

    info = topic_model.get_topic_info().copy()
    info["top_words"] = info["Topic"].map(
        lambda t: "|".join(w for w, _ in topic_model.get_topic(t)) if t != -1 else ""
    )
    info = info.rename(columns={"Topic": "topic", "Count": "count", "Name": "name"})
    info[["topic", "count", "name", "top_words"]].to_csv(
        "results/topic_info.csv", index=False
    )

    df.to_csv("results/document_topics.csv", index=False)

    crosstab = pd.crosstab(df["topic"], df["category"])
    crosstab.to_csv("results/topic_category_crosstab.csv")

    share = (
        df.groupby("topic")
        .agg(count=("satiric", "size"), satire_share=("satiric", "mean"))
        .reset_index()
        .merge(info[["topic", "top_words"]], on="topic", how="left")
    )
    share.to_csv("results/topic_satire_share.csv", index=False)

    plot_satire_share(share)

    n_topics = (info["topic"] != -1).sum()
    outliers = (
        int(info.loc[info["topic"] == -1, "count"].sum())
        if (info["topic"] == -1).any()
        else 0
    )
    print(
        f"Found {n_topics} topics; {outliers} outlier headlines ({100 * outliers / len(df):.1f}%)."
    )


if __name__ == "__main__":
    main()
