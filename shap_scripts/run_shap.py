import argparse
import os
import pickle
import time

import numpy as np
import pandas as pd
import shap
import torch
from common import CATEGORIES, load_model, load_split, make_predict_proba


def explain_category(category, device, limit=None):
    model, tokenizer = load_model(category, device)
    predict_proba = make_predict_proba(model, tokenizer, device, batch_size=128)
    titles, labels = load_split(category, "test")
    if limit is not None:
        titles, labels = titles[:limit], labels[:limit]

    probs = predict_proba(titles)
    preds = (probs >= 0.5).astype(int)
    pred_df = pd.DataFrame(
        {
            "idx": np.arange(len(titles)),
            "title": titles,
            "label": labels,
            "prob_satire": probs,
            "prediction": preds,
        }
    )
    pred_df.to_csv(f"results/predictions_{category}.csv", index=False)

    masker = shap.maskers.Text(tokenizer)
    explainer = shap.Explainer(predict_proba, masker, seed=42)

    start = time.time()
    explanation = explainer(titles)
    print(
        f"{category}: explained {len(titles)} headlines in {time.time() - start:.0f}s"
    )

    with open(f"results/shap_values_{category}.pkl", "wb") as f:
        pickle.dump(explanation, f)

    rows = []
    for i in range(len(titles)):
        values = explanation.values[i]
        tokens = explanation.data[i]
        for pos, (token, value) in enumerate(zip(tokens, values)):
            rows.append(
                {
                    "idx": i,
                    "label": labels[i],
                    "prob_satire": probs[i],
                    "prediction": preds[i],
                    "position": pos,
                    "token": token,
                    "shap_value": value,
                }
            )
    pd.DataFrame(rows).to_csv(f"results/shap_tokens_{category}.csv", index=False)

    del model
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="all", choices=CATEGORIES + ["all"])
    parser.add_argument(
        "--limit", type=int, default=None, help="only explain the first N headlines"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)
    os.makedirs("results", exist_ok=True)

    categories = CATEGORIES if args.category == "all" else [args.category]
    for category in categories:
        explain_category(category, device, limit=args.limit)


if __name__ == "__main__":
    main()
