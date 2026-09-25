import argparse
import sys
from pathlib import Path

import lightning as pl
from bert_models import BERTmodule
from train_BERT_satire import INTER_TASK_LABEL, STANDARD_NAMES, HeadlineDataset, predict
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis_scripts.prediction_io import source_column, write_predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model_type", default="bert")
    parser.add_argument(
        "--inter_task", default="standard", choices=list(INTER_TASK_LABEL)
    )
    parser.add_argument("--category", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", default="../data")
    parser.add_argument("--split_family", default="random")
    parser.add_argument("--eval_splits", nargs="+", default=["val", "test"])
    parser.add_argument("--variant_suffix", default="")
    args = parser.parse_args()

    pl.seed_everything(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(STANDARD_NAMES[args.model_type])
    module = BERTmodule.load_from_checkpoint(
        args.checkpoint, tokenizer=tokenizer, weights_only=False
    )
    variant = INTER_TASK_LABEL[args.inter_task] + (
        f"__{args.variant_suffix}" if args.variant_suffix else ""
    )
    for eval_split in args.eval_splits:
        dataset = HeadlineDataset(args.category, args.data_dir, eval_split)
        preds, scores = predict(module, dataset)
        path = write_predictions(
            method="bert",
            model=args.model_type,
            variant=variant,
            category=args.category,
            seed=args.seed,
            split_family=args.split_family,
            eval_split=eval_split,
            titles=dataset.titles,
            ground_truth=dataset.labels,
            prediction=preds,
            score=scores,
            source=source_column(dataset.df),
        )
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
