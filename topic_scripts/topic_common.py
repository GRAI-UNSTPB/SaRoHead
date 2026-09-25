import unicodedata

import matplotlib

matplotlib.use("Agg")
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from spacy.lang.ro.stop_words import STOP_WORDS

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from analysis_scripts.ro_en_glossary import gloss_pair

PLOTS_DIR = os.environ.get("SAROHEAD_PLOTS_DIR", "plots")
DATA_DIR = os.environ.get("SAROHEAD_DATA_DIR", "../data")
CATEGORIES = ["social", "politic", "sport"]

PLACEHOLDER_WORDS = [
    "person",
    "organization",
    "gpe",
    "loc",
    "facility",
    "event",
    "money",
    "period",
    "product",
    "language",
    "work_of_art",
    "nat_rel_pol",
    "nat_geo_rel",
    "datetime",
    "numeric_value",
    "ordinal",
    "quantity",
]


def strip_diacritics(text):
    return "".join(
        ch
        for ch in unicodedata.normalize("NFD", text)
        if unicodedata.category(ch) != "Mn"
    )


def romanian_stopwords():
    stops = set()
    for word in STOP_WORDS:
        stops.add(word.lower())
        stops.add(strip_diacritics(word.lower()))
    stops.update(PLACEHOLDER_WORDS)
    return sorted(stops)


def load_full_dataset():
    full = pd.read_csv(os.path.join(DATA_DIR, "all_headlines.csv"))
    full = full.rename(columns={"proc_title": "new_title"})[
        ["new_title", "category", "satiric"]
    ]
    full["new_title"] = full["new_title"].astype(str)
    return full


def plot_satire_share(share_df, n_topics=15):
    top = share_df[share_df["topic"] != -1].nlargest(n_topics, "count")
    labels = [
        f"{row.topic}: {gloss_pair(row.top_words.split('|')[:2])}"
        for row in top.itertuples()
    ]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    colors = ["#c0392b" if s > 0.5 else "#2471a3" for s in top["satire_share"]]
    ax.barh(np.arange(len(top))[::-1], 100 * top["satire_share"], color=colors)
    ax.set_yticks(np.arange(len(top))[::-1])
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(50, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Satirical headlines in topic (%)")
    fig.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig.savefig(f"{PLOTS_DIR}/topic_satire_share.pdf", bbox_inches="tight")
    plt.close(fig)
