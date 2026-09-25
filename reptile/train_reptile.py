import argparse
import random
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from bert_models import NeuralClassifier
from sklearn.metrics import precision_recall_fscore_support
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis_scripts.prediction_io import read_split, source_column, write_predictions

try:
    from aim import Run
except Exception:
    Run = None

print("LOADED Modules")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

STANDARD_NAMES = {
    "bert": "dumitrescustefan/bert-base-romanian-uncased-v1",
    "xlm-roberta": "xlm-roberta-base",
    "distilled-bert": "racai/distilbert-base-romanian-uncased",
}
ADDED_TOKENS = [
    "[EVENT]",
    "[FACILITY]",
    "[GPE]",
    "[LANGUAGE]",
    "[LOC]",
    "[MONEY]",
    "[NAT_REL_POL]",
    "[ORGANIZATION]",
    "[PERIOD]",
    "[PERSON]",
    "[PRODUCT]",
    "[WORK_OF_ART]",
]

META_EPOCHS = 150
INNER_BATCH_SIZE = 32
BETA_LR = 1e-4
ALPHA_LR = 0.8
END_ALPHA = 1e-5
TASKS_PER_META_EPOCH = 3


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type", type=str, default="bert", choices=list(STANDARD_NAMES)
    )
    parser.add_argument(
        "--train_style",
        type=str,
        default="separate",
        choices=["separate", "sequential"],
    )
    parser.add_argument("--K", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", default="../data")
    parser.add_argument("--split_family", default="random")
    parser.add_argument("--variant_suffix", default="")
    parser.add_argument(
        "--categories", nargs="+", default=["social", "politic", "sport"]
    )
    parser.add_argument("--meta_epochs", type=int, default=META_EPOCHS)
    parser.add_argument("--checkpoint_root", default="backup")
    parser.add_argument("--use_aim", action="store_true")
    return parser.parse_args()


def reset_seed(seed_value):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    random.seed(seed_value)


def load_tasks(data_dir):
    def prep(df, text_col, label_col):
        df = df.rename(columns={text_col: "text", label_col: "label"})
        df = df[["text", "label"] + (["category"] if "category" in df.columns else [])]
        df = df.dropna(subset=["text", "label"])
        df = df[df["text"].astype(str).str.strip().str.len() > 0]
        df["label"] = df["label"].astype(int)
        return df.reset_index(drop=True)

    aux_root = Path(__file__).resolve().parents[1] / "unsupervised_TTL"

    train_satire = prep(read_split(data_dir, "train"), "new_title", "satiric")
    train_roclico = prep(
        pd.read_csv(aux_root / "click" / "train.csv"), "title", "label"
    )  # title,label
    train_scitech = prep(
        pd.read_csv(aux_root / "scitechbait" / "train.csv"), "headline", "clickbait"
    )  # headline,clickbait
    train_saroco = prep(
        pd.read_csv(aux_root / "saroco" / "train.csv"), "title", "label"
    )  # title,label
    return train_satire, {
        "RoCliCo": train_roclico,
        "SaRoCo": train_saroco,
        "SciTechBaitRo": train_scitech,
    }


@torch.no_grad()
def evaluate(model, tokenizer, df, loss_fn):
    model.eval()
    titles, labels = (
        df["new_title"].fillna("").tolist(),
        df["satiric"].astype(int).tolist(),
    )
    preds, scores, loss_total, steps = [], [], 0.0, 0
    for j in range(0, len(titles), INNER_BATCH_SIZE):
        batch_titles = titles[j : j + INNER_BATCH_SIZE]
        batch_labels = labels[j : j + INNER_BATCH_SIZE]
        real_input = tokenizer(
            batch_titles,
            padding="longest",
            max_length=128,
            truncation=True,
            return_tensors="pt",
        )
        logits = model(real_input)
        loss_total += loss_fn(
            logits, torch.tensor(batch_labels, dtype=torch.int64).to(device)
        ).item()
        steps += 1
        probs = torch.softmax(logits.detach().float().cpu(), dim=1)
        preds += torch.argmax(probs, dim=1).tolist()
        scores += probs[:, 1].tolist()
    return preds, scores, loss_total / max(steps, 1)


def train_reptile(
    args,
    news_category,
    tokenizer,
    train_satire,
    aux_tasks,
    val_df,
    neural_classifier=None,
):
    standard_name = STANDARD_NAMES[args.model_type]
    if args.train_style == "separate":
        reset_seed(args.seed)
        neural_classifier = NeuralClassifier(
            standard_name, len(tokenizer), num_labels=2, full_finetune=True
        ).to(device)

    aim_run = None
    if args.use_aim and Run is not None:
        aim_run = Run(
            experiment=f"{args.model_type}_{news_category}_REPTILE_{args.train_style}_{args.split_family}_seed{args.seed}"
        )
        aim_run["hparams"] = {
            "META_EPOCHS": args.meta_epochs,
            "BATCH_SIZE": INNER_BATCH_SIZE,
            "BETA_LR": BETA_LR,
            "ALPHA_LR": ALPHA_LR,
            "END_ALPHA": END_ALPHA,
            "model_type": args.model_type,
            "K": args.K,
            "seed": args.seed,
        }

    tasks = {
        "SaRoHead": train_satire[train_satire["category"] == news_category].reset_index(
            drop=True
        ),
        **aux_tasks,
    }
    task_names = list(tasks)
    loss_fn = nn.CrossEntropyLoss()
    alpha = ALPHA_LR
    alpha_step = (END_ALPHA - ALPHA_LR) / args.meta_epochs
    best_val_loss = None
    ckpt_dir = Path(args.checkpoint_root)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = (
        ckpt_dir
        / f"{args.train_style}_meta_learning_{args.model_type}_{news_category}_{args.split_family}_seed{args.seed}.pt"
    )

    task_shell = NeuralClassifier(
        standard_name, len(tokenizer), num_labels=2, full_finetune=True
    ).to(device)

    for epoch_no in range(args.meta_epochs):
        deltas = []
        train_loss = 0.0
        neural_classifier.train()
        sampled = random.sample(task_names, k=TASKS_PER_META_EPOCH)
        for task_name in sampled:
            task_df = tasks[task_name]
            task_shell.load_state_dict(neural_classifier.state_dict())
            task_shell.train()
            pairs = list(zip(task_df["text"].tolist(), task_df["label"].tolist()))
            chosen = random.sample(pairs, k=min(INNER_BATCH_SIZE * args.K, len(pairs)))
            opt_inner = torch.optim.Adam(task_shell.parameters(), lr=BETA_LR)
            task_loss = 0.0
            for j in range(0, len(chosen), INNER_BATCH_SIZE):
                opt_inner.zero_grad()
                batch = chosen[j : j + INNER_BATCH_SIZE]
                titles = [b[0] for b in batch]
                labels = [int(b[1]) for b in batch]
                real_input = tokenizer(
                    titles,
                    padding="longest",
                    max_length=128,
                    truncation=True,
                    return_tensors="pt",
                )
                inner_loss = loss_fn(
                    task_shell(real_input),
                    torch.tensor(labels, dtype=torch.int64).to(device),
                )
                task_loss += inner_loss.item()
                inner_loss.backward()
                opt_inner.step()
            train_loss += task_loss / args.K
            base_state = neural_classifier.state_dict()
            deltas.append(
                {
                    name: param.detach() - base_state[name]
                    for name, param in task_shell.named_parameters()
                }
            )
        n_tasks = len(deltas)
        with torch.no_grad():
            for name, param in neural_classifier.named_parameters():
                delta = torch.zeros_like(param)
                for d in deltas:
                    delta += alpha * d[name]
                param.add_(delta / n_tasks)
        alpha += alpha_step

        preds, _, val_loss = evaluate(neural_classifier, tokenizer, val_df, loss_fn)
        true = val_df["satiric"].astype(int).tolist()
        _, _, macro_f1, _ = precision_recall_fscore_support(
            true, preds, average="macro", zero_division=0
        )
        _, _, per_class_f1, _ = precision_recall_fscore_support(
            true, preds, average=None, labels=[0, 1], zero_division=0
        )
        print(
            f"{news_category} epoch {epoch_no + 1} {sampled} train_loss={train_loss / n_tasks:.4f} "
            f"val_loss={val_loss:.4f} macro_f1={macro_f1:.4f} satire_f1={per_class_f1[1]:.4f}",
            file=sys.stderr,
            flush=True,
        )
        if aim_run is not None:
            aim_run.track(
                train_loss / n_tasks, epoch=epoch_no, name="Train_loss_meta_learning"
            )
            aim_run.track(
                val_loss, epoch=epoch_no, name="Validation_loss_meta_learning"
            )
            aim_run.track(macro_f1, epoch=epoch_no, name="Validation_macro_F1")
            aim_run.track(per_class_f1[1], epoch=epoch_no, name="Validation_satire_F1")
        if best_val_loss is None or val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(neural_classifier.state_dict(), ckpt_path)

    neural_classifier.load_state_dict(
        torch.load(ckpt_path, map_location=device, weights_only=True)
    )
    variant = "reptile" + (f"__{args.variant_suffix}" if args.variant_suffix else "")
    for eval_split in ["val", "test"]:
        df = read_split(args.data_dir, eval_split, news_category)
        preds, scores, _ = evaluate(neural_classifier, tokenizer, df, loss_fn)
        path = write_predictions(
            method="reptile",
            model=args.model_type,
            variant=variant,
            category=news_category,
            seed=args.seed,
            split_family=args.split_family,
            eval_split=eval_split,
            titles=df["new_title"].fillna(""),
            ground_truth=df["satiric"].astype(int),
            prediction=preds,
            score=scores,
            source=source_column(df),
        )
        print(f"wrote {path}", file=sys.stderr)
    del task_shell
    return neural_classifier


if __name__ == "__main__":
    args = parse_args()
    data_dir = Path(args.data_dir)
    standard_name = STANDARD_NAMES[args.model_type]
    tokenizer = AutoTokenizer.from_pretrained(standard_name, do_lower_case=True)
    tokenizer.add_tokens(ADDED_TOKENS)
    train_satire, aux_tasks = load_tasks(data_dir)
    val_satire = read_split(data_dir, "val")

    model = None
    if args.train_style == "sequential":
        reset_seed(args.seed)
        model = NeuralClassifier(
            standard_name, len(tokenizer), num_labels=2, full_finetune=True
        ).to(device)
    for category in args.categories:
        val_df = val_satire[val_satire["category"] == category].reset_index(drop=True)
        model = train_reptile(
            args, category, tokenizer, train_satire, aux_tasks, val_df, model
        )
