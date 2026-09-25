import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from bert_models import NeuralClassifier
from train_reptile import ADDED_TOKENS, STANDARD_NAMES, device, evaluate, reset_seed
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis_scripts.prediction_io import read_split, source_column, write_predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default="bert", choices=list(STANDARD_NAMES))
    parser.add_argument("--category", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", default="../data")
    parser.add_argument("--split_family", default="random")
    parser.add_argument("--eval_splits", nargs="+", default=["val", "test"])
    parser.add_argument("--variant_suffix", default="")
    args = parser.parse_args()

    reset_seed(args.seed)
    standard_name = STANDARD_NAMES[args.model_type]
    tokenizer = AutoTokenizer.from_pretrained(standard_name, do_lower_case=True)
    tokenizer.add_tokens(ADDED_TOKENS)
    model = NeuralClassifier(
        standard_name, len(tokenizer), num_labels=2, full_finetune=True
    )
    model.load_state_dict(
        torch.load(args.checkpoint, weights_only=True, map_location=device)
    )
    model = model.to(device).eval()
    loss_fn = nn.CrossEntropyLoss()
    variant = "reptile" + (f"__{args.variant_suffix}" if args.variant_suffix else "")
    for eval_split in args.eval_splits:
        df = read_split(args.data_dir, eval_split, args.category)
        preds, scores, _ = evaluate(model, tokenizer, df, loss_fn)
        path = write_predictions(
            method="reptile",
            model=args.model_type,
            variant=variant,
            category=args.category,
            seed=args.seed,
            split_family=args.split_family,
            eval_split=eval_split,
            titles=df["new_title"].fillna(""),
            ground_truth=df["satiric"].astype(int),
            prediction=preds,
            score=scores,
            source=source_column(df),
        )
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
