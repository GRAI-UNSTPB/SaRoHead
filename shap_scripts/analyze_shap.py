import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from common import CATEGORIES, load_model, make_predict_proba
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis_scripts"),
)
from ro_en_glossary import gloss, translate_headline
from unmasked_entities import contains_unmasked_entity, is_unmasked_entity

PLOTS_DIR = os.environ.get("SAROHEAD_PLOTS_DIR", "plots")
MIN_COUNT = {"social": 10, "politic": 10, "sport": 10}
TOP_K_PLOT = 10
FAITHFULNESS_KS = [1, 2, 3, 4, 5]
N_RANDOM_DRAWS = 5
MAX_EXAMPLE_WORDS = 14

PUNCT_ONLY = re.compile(r"^[^\w\[\]]+$")


def merge_words(raw_tokens, values):
    words, word_values = [], []
    current, current_value = "", 0.0
    for raw, value in zip(raw_tokens, values):
        raw = str(raw)
        stripped = raw.strip()
        if stripped == "":
            if current:
                words.append(current)
                word_values.append(current_value)
                current, current_value = "", 0.0
            continue
        if current and PUNCT_ONLY.match(stripped):
            words.append(current)
            word_values.append(current_value)
            current, current_value = "", 0.0
        current += stripped
        current_value += float(value)
        if raw != raw.rstrip():
            words.append(current)
            word_values.append(current_value)
            current, current_value = "", 0.0
    if current:
        words.append(current)
        word_values.append(current_value)
    return words, np.array(word_values)


def load_word_level(category):
    df = pd.read_csv(f"results/shap_tokens_{category}.csv", keep_default_na=False)
    rows = []
    for idx, group in df.groupby("idx"):
        group = group.sort_values("position")
        words, values = merge_words(
            group["token"].tolist(), group["shap_value"].to_numpy()
        )
        meta = group.iloc[0]
        for word, value in zip(words, values):
            rows.append(
                {
                    "idx": idx,
                    "label": meta["label"],
                    "prob_satire": meta["prob_satire"],
                    "prediction": meta["prediction"],
                    "word": word.lower(),
                    "shap_value": value,
                }
            )
    return pd.DataFrame(rows)


def global_importance(category):
    df = load_word_level(category)
    grouped = (
        df.groupby("word")["shap_value"]
        .agg(count="count", mean_shap="mean", mean_abs_shap=lambda s: s.abs().mean())
        .reset_index()
        .sort_values("mean_shap", ascending=False)
    )
    grouped["unmasked_entity"] = grouped["word"].map(is_unmasked_entity)
    grouped.to_csv(f"results/global_importance_{category}.csv", index=False)
    return grouped


def plot_global(importances):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    for ax, category in zip(axes, CATEGORIES):
        grouped = importances[category]
        frequent = grouped[
            (grouped["count"] >= MIN_COUNT[category]) & ~grouped["unmasked_entity"]
        ]
        top_satire = frequent.nlargest(TOP_K_PLOT, "mean_shap")
        top_mainstream = frequent.nsmallest(TOP_K_PLOT, "mean_shap").iloc[::-1]
        plot_df = pd.concat([top_satire, top_mainstream])
        colors = ["#c0392b" if v > 0 else "#2471a3" for v in plot_df["mean_shap"]]
        ax.barh(np.arange(len(plot_df))[::-1], plot_df["mean_shap"], color=colors)
        ax.set_yticks(np.arange(len(plot_df))[::-1])
        ax.set_yticklabels([gloss(w) for w in plot_df["word"]], fontsize=8)
        ax.axvline(0, color="black", linewidth=0.6)
        ax.set_title(category.capitalize())
        ax.set_xlabel("Mean SHAP value")
    fig.suptitle("Words pushing towards satire (red) and mainstream (blue)", y=1.0)
    fig.tight_layout()
    fig.savefig(f"{PLOTS_DIR}/shap_global_tokens.pdf", bbox_inches="tight")
    plt.close(fig)


def get_example_words(category):
    df = load_word_level(category)
    return {idx: g.reset_index(drop=True) for idx, g in df.groupby("idx")}


def pick_examples(preds, word_map):
    lengths = {idx: len(g) for idx, g in word_map.items()}
    clean = {
        idx: not contains_unmasked_entity(g["word"]) for idx, g in word_map.items()
    }
    short = preds[
        (preds["idx"].map(lengths) <= MAX_EXAMPLE_WORDS) & preds["idx"].map(clean)
    ]
    examples = []
    correct_sat = short[(short.label == 1) & (short.prediction == 1)]
    if len(correct_sat):
        examples.append(
            (
                int(correct_sat.nlargest(1, "prob_satire").iloc[0]["idx"]),
                "satirical, correctly classified",
            )
        )
    correct_main = short[(short.label == 0) & (short.prediction == 0)]
    if len(correct_main):
        examples.append(
            (
                int(correct_main.nsmallest(1, "prob_satire").iloc[0]["idx"]),
                "mainstream, correctly classified",
            )
        )
    wrong = short[short.label != short.prediction].copy()
    if len(wrong):
        wrong["confidence"] = (wrong["prob_satire"] - 0.5).abs()
        examples.append(
            (int(wrong.nlargest(1, "confidence").iloc[0]["idx"]), "misclassified")
        )
    return examples


def plot_local_examples(category):
    preds = pd.read_csv(f"results/predictions_{category}.csv")
    word_map = get_example_words(category)
    examples = pick_examples(preds, word_map)

    fig, axes = plt.subplots(len(examples), 1, figsize=(10, 1.6 * len(examples)))
    if len(examples) == 1:
        axes = [axes]
    all_vals = np.concatenate(
        [word_map[idx]["shap_value"].to_numpy() for idx, _ in examples]
    )
    vmax = np.abs(all_vals).max()
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.get_cmap("RdBu_r")

    for ax, (idx, kind) in zip(axes, examples):
        g = word_map[idx]
        x = 0.0
        for word, value in zip(g["word"], g["shap_value"]):
            width = 0.014 * (len(word) + 1) + 0.012
            ax.add_patch(
                plt.Rectangle(
                    (x, 0.12), width, 0.76, color=cmap(norm(value)), ec="white", lw=0.8
                )
            )
            ax.text(x + width / 2, 0.5, word, ha="center", va="center", fontsize=8.5)
            x += width
        row = preds[preds["idx"] == idx].iloc[0]
        ax.set_xlim(0, max(1.0, x))
        ax.set_ylim(0, 1)
        ax.axis("off")
        title = f"{kind} - P(satire)={row['prob_satire']:.2f}, label={'satirical' if row['label'] == 1 else 'mainstream'}"
        english = translate_headline(g["word"].tolist())
        if english:
            title += f'\nEnglish: "{english}"'
        ax.set_title(title, fontsize=9, loc="left")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(
        sm, ax=axes, orientation="vertical", fraction=0.02, pad=0.01, label="SHAP value"
    )
    fig.savefig(f"{PLOTS_DIR}/shap_local_examples_{category}.pdf", bbox_inches="tight")
    plt.close(fig)


def faithfulness(device):
    rng = np.random.default_rng(42)
    rows = []
    for category in CATEGORIES:
        model, tokenizer = load_model(category, device)
        predict_proba = make_predict_proba(model, tokenizer, device, batch_size=128)
        preds = pd.read_csv(f"results/predictions_{category}.csv")
        word_map = get_example_words(category)

        target_idx = preds[(preds.label == 1) & (preds.prediction == 1)]["idx"].tolist()
        base_probs = preds.set_index("idx").loc[target_idx, "prob_satire"].to_numpy()
        rows.append(
            {
                "category": category,
                "k": 0,
                "strategy": "shap",
                "mean_prob": base_probs.mean(),
            }
        )
        rows.append(
            {
                "category": category,
                "k": 0,
                "strategy": "random",
                "mean_prob": base_probs.mean(),
            }
        )

        word_lists = [word_map[idx]["word"].tolist() for idx in target_idx]
        value_lists = [word_map[idx]["shap_value"].to_numpy() for idx in target_idx]

        for k in FAITHFULNESS_KS:
            shap_texts = []
            for words, values in zip(word_lists, value_lists):
                order = np.argsort(values)[::-1]
                drop = set(order[: min(k, len(words))])
                shap_texts.append(
                    " ".join(w for i, w in enumerate(words) if i not in drop)
                )
            shap_probs = predict_proba(shap_texts)
            rows.append(
                {
                    "category": category,
                    "k": k,
                    "strategy": "shap",
                    "mean_prob": shap_probs.mean(),
                }
            )

            draw_means = []
            for _ in range(N_RANDOM_DRAWS):
                rand_texts = []
                for words in word_lists:
                    drop = set(
                        rng.choice(len(words), size=min(k, len(words)), replace=False)
                    )
                    rand_texts.append(
                        " ".join(w for i, w in enumerate(words) if i not in drop)
                    )
                draw_means.append(predict_proba(rand_texts).mean())
            rows.append(
                {
                    "category": category,
                    "k": k,
                    "strategy": "random",
                    "mean_prob": float(np.mean(draw_means)),
                }
            )
            print(
                f"{category} k={k}: shap={shap_probs.mean():.3f} random={np.mean(draw_means):.3f}"
            )

        del model
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv("results/faithfulness_deletion.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
    for ax, category in zip(axes, CATEGORIES):
        sub = df[df.category == category]
        for strategy, style, label in [
            ("shap", "o-", "SHAP-guided deletion"),
            ("random", "s--", "Random deletion"),
        ]:
            line = sub[sub.strategy == strategy].sort_values("k")
            ax.plot(line["k"], line["mean_prob"], style, label=label)
        ax.set_title(category.capitalize())
        ax.set_xlabel("Deleted words (k)")
        ax.set_xticks([0] + FAITHFULNESS_KS)
    axes[0].set_ylabel("Mean P(satire)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{PLOTS_DIR}/shap_faithfulness.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    importances = {c: global_importance(c) for c in CATEGORIES}
    plot_global(importances)
    for category in CATEGORIES:
        plot_local_examples(category)
    faithfulness(device)
    print("Analysis complete.")


if __name__ == "__main__":
    main()
