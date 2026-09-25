import argparse
import sys
from pathlib import Path

import lightning as pl
import pandas as pd
import torch
from bert_models import BERTmodule
from lightning.pytorch.loggers.csv_logs import CSVLogger
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis_scripts.prediction_io import read_split, source_column, write_predictions

try:
    from aim.pytorch_lightning import AimLogger
except Exception:
    AimLogger = None

STANDARD_NAMES = {
    "bert": "dumitrescustefan/bert-base-romanian-uncased-v1",
    "xlm-roberta": "xlm-roberta-base",
    "distilled-bert": "racai/distilbert-base-romanian-uncased",
}
FOLDER_SUFFIX = {"bert": "BERT", "xlm-roberta": "XLM", "distilled-bert": "Distil"}
INTER_TASK_LABEL = {
    "standard": "None",
    "saroco": "SaRoCo",
    "click": "RoCliCo",
    "scitechbait": "SciTechBaitRo",
}


class HeadlineDataset(Dataset):
    def __init__(self, category, data_dir, split):
        self.df = read_split(data_dir, split, category)
        self.titles = self.df["new_title"].fillna("")
        self.labels = self.df["satiric"].to_numpy()

    def __getitem__(self, idx):
        return self.titles.iloc[idx], self.labels[idx]

    def __len__(self):
        return self.labels.shape[0]


def collate_fn(batch):
    titles, labels = [], []
    for title, label in batch:
        titles.append(title)
        labels.append(label)
    return titles, torch.tensor(labels).to(dtype=torch.int64)


@torch.no_grad()
def predict(module, dataset, batch_size=32):
    module.eval()
    model, tokenizer = module.neural_classifier, module.tokenizer
    loader = DataLoader(
        dataset, shuffle=False, batch_size=batch_size, collate_fn=collate_fn
    )
    preds, scores = [], []
    for titles, _ in loader:
        tokenized = tokenizer(
            titles,
            return_tensors="pt",
            padding="longest",
            max_length=128,
            truncation=True,
        )
        logits = model(tokenized).detach().float().cpu()
        probs = torch.softmax(logits, dim=1)
        preds.extend(torch.argmax(logits, dim=1).tolist())
        scores.extend(probs[:, 1].tolist())
    return preds, scores


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default="bert", choices=list(STANDARD_NAMES))
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--sch_type", default="linear")
    parser.add_argument(
        "--inter_task", default="standard", choices=list(INTER_TASK_LABEL)
    )
    parser.add_argument("--num_epochs", type=int, default=35)
    parser.add_argument("--full_finetune", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", default="../data")
    parser.add_argument("--split_family", default="random")
    parser.add_argument("--variant_suffix", default="")
    parser.add_argument(
        "--categories", nargs="+", default=["social", "politic", "sport"]
    )
    parser.add_argument("--checkpoint_root", default="checkpoints")
    parser.add_argument("--use_aim", action="store_true")
    return parser.parse_args()


def train_category(args, category_name):
    pl.seed_everything(args.seed, workers=True)
    train_data = HeadlineDataset(category_name, args.data_dir, "train")
    val_data = HeadlineDataset(category_name, args.data_dir, "val")
    test_data = HeadlineDataset(category_name, args.data_dir, "test")

    standard_name = STANDARD_NAMES[args.model_type]
    model_path = (
        standard_name
        if args.inter_task == "standard"
        else f"../unsupervised_TTL/{args.inter_task}/backup_{args.inter_task}_{FOLDER_SUFFIX[args.model_type]}"
    )
    batch_size = 32
    train_loader = DataLoader(
        train_data, shuffle=True, batch_size=batch_size, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_data, shuffle=False, batch_size=batch_size, collate_fn=collate_fn
    )
    total_steps = args.num_epochs * len(train_loader)
    tokenizer = AutoTokenizer.from_pretrained(standard_name)
    module = BERTmodule(
        model_path,
        tokenizer,
        num_labels=2,
        sch_name=args.sch_type,
        warmup_steps=0,
        total_steps=total_steps,
        lr=args.lr,
        full_finetune=args.full_finetune,
    )

    run_name = f"{category_name}_{args.model_type}_{args.inter_task}_ft{args.full_finetune}_{args.split_family}_seed{args.seed}"
    ckpt_dir = Path(args.checkpoint_root) / run_name
    checkpoint = pl.pytorch.callbacks.ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename=f"{args.model_type}_CKPT",
        monitor="Validation_Satire_F1",
        mode="max",
        save_top_k=1,
        enable_version_counter=False,
    )
    loggers = [CSVLogger(str(Path("logs") / run_name), f"{args.model_type}_logs")]
    if args.use_aim and AimLogger is not None:
        aim_logger = AimLogger(experiment=f"train_{run_name}")
        aim_logger.log_hyperparams(
            {
                "lr": args.lr,
                "batch_size": batch_size,
                "lr_sched": args.sch_type,
                "num_epochs": args.num_epochs,
                "model_type": args.model_type,
                "seed": args.seed,
            }
        )
        loggers.append(aim_logger)

    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        precision="bf16",
        logger=loggers,
        callbacks=[checkpoint],
    )
    trainer.fit(module, train_loader, val_loader)

    best_path = checkpoint.best_model_path
    if not best_path:
        raise RuntimeError("no checkpoint saved")
    best = BERTmodule.load_from_checkpoint(
        best_path, tokenizer=tokenizer, weights_only=False
    )
    variant = INTER_TASK_LABEL[args.inter_task] + (
        f"__{args.variant_suffix}" if args.variant_suffix else ""
    )
    for eval_split, dataset in [("val", val_data), ("test", test_data)]:
        preds, scores = predict(best, dataset, batch_size)
        path = write_predictions(
            method="bert",
            model=args.model_type,
            variant=variant,
            category=category_name,
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
    args = parse_args()
    for category in args.categories:
        print("#" * 100)
        print(f"TRAINING {category} (seed {args.seed}, split {args.split_family})")
        train_category(args, category)
