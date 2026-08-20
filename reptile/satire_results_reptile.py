import argparse

import lightning as pl
import pandas as pd
import spacy
import torch
import torch.nn as nn
from aim.pytorch_lightning import AimLogger
from bert_models import *
from bert_models import NeuralClassifier
from torch.utils.data import DataLoader, Dataset
from torchmetrics import functional
from transformers import AutoTokenizer


class HeadlineDataset(Dataset):
    def __init__(self, category, df_name):
        self.df = pd.read_csv(df_name)
        self.df = self.df[self.df["category"] == category]
        title_col = (
            "proc_title"
            if "proc_title" in self.df.columns
            else ("new_title" if "new_title" in self.df.columns else "title")
        )
        self.titles = self.df[title_col]
        self.labels = self.df["satiric"].to_numpy()

    def __getitem__(self, idx):
        return self.titles.iloc[idx], self.labels[idx]

    def __len__(self):
        return self.labels.shape[0]


def collate_fn(batch):
    titles = []
    labels = []
    for title, label in batch:
        titles.append(title)
        labels.append(label)
    labels = torch.tensor(labels).to(dtype=torch.int64)
    return titles, labels


ro_nlp_spacy = spacy.load("ro_core_news_lg")
total_labels = list(ro_nlp_spacy.get_pipe("ner").labels)
skip_labels = ["NUMERIC_VALUE", "DATETIME", "ORDINAL", "QUANTITY"]
added_tokens = [label for label in total_labels if label not in skip_labels]
added_tokens = list(map(lambda tok: "[" + tok + "]", added_tokens))
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--model_type", default="bert")
parsed_args = arg_parser.parse_args()
model_type = parsed_args.model_type

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
df_dict = {
    "category": [],
    "model": [],
    "satire_recall": [],
    "satire_precision": [],
    "satire_F1": [],
    "mainstream_recall": [],
    "mainstream_precision": [],
    "mainstream_F1": [],
    "test_type": [],
}

if model_type == "bert":
    standard_name = "dumitrescustefan/bert-base-romanian-uncased-v1"
elif model_type == "xlm-roberta":
    standard_name = "xlm-roberta-base"
elif model_type == "distilled-bert":
    standard_name = "racai/distilbert-base-romanian-uncased"


def test_category(category_name):

    pl.seed_everything(42)
    testData = HeadlineDataset(category_name, "../data/test.csv")
    testLoader = DataLoader(
        testData, shuffle=False, batch_size=32, collate_fn=collate_fn
    )
    valData = HeadlineDataset(category_name, "../data/validation.csv")
    valLoader = DataLoader(valData, shuffle=False, batch_size=32, collate_fn=collate_fn)
    tokenizer = AutoTokenizer.from_pretrained(standard_name, do_lower_case=True)
    tokenizer.add_tokens(added_tokens)
    full_model = NeuralClassifier(
        standard_name, len(tokenizer), num_labels=2, full_finetune=True
    )
    full_model.load_state_dict(
        torch.load(
            f"backup/V2_separate_meta_learning_{model_type}_{category_name}.pt",
            weights_only=True,
        )
    )
    full_model = full_model.to(device)
    full_model.eval()
    total_preds = torch.tensor([])
    total_GT = torch.tensor([])
    for batch in testLoader:
        titles = batch[0]
        tokenized_titles = tokenizer(
            titles,
            return_tensors="pt",
            padding="longest",
            max_length=128,
            truncation=True,
        )
        labels = batch[1]
        total_GT = torch.concatenate((total_GT, labels))
        out_logits = full_model(tokenized_titles)
        y_pred = torch.argmax(out_logits.detach().cpu(), dim=1)
        total_preds = torch.concatenate((total_preds, y_pred))
    df_dict["category"].append(category_name)
    df_dict["model"].append(model_type)
    df_dict["test_type"].append("test")
    df_dict["satire_recall"].append(
        functional.classification.binary_recall(total_preds, total_GT)
        .round(decimals=4)
        .item()
    )
    df_dict["satire_precision"].append(
        functional.classification.binary_precision(total_preds, total_GT)
        .round(decimals=4)
        .item()
    )
    df_dict["mainstream_recall"].append(
        functional.classification.multiclass_recall(
            total_preds, total_GT, num_classes=2, average=None
        )[0]
        .round(decimals=4)
        .item()
    )
    df_dict["mainstream_precision"].append(
        functional.classification.multiclass_precision(
            total_preds, total_GT, num_classes=2, average=None
        )[0]
        .round(decimals=4)
        .item()
    )
    df_dict["mainstream_F1"].append(
        functional.classification.multiclass_f1_score(
            total_preds, total_GT, num_classes=2, average=None
        )[0]
        .round(decimals=4)
        .item()
    )
    df_dict["satire_F1"].append(
        functional.classification.binary_f1_score(total_preds, total_GT)
        .round(decimals=4)
        .item()
    )

    total_preds = torch.tensor([])
    total_GT = torch.tensor([])
    for batch in valLoader:
        titles = batch[0]
        tokenized_titles = tokenizer(
            titles,
            return_tensors="pt",
            padding="longest",
            max_length=128,
            truncation=True,
        )
        labels = batch[1]
        total_GT = torch.concatenate((total_GT, labels))
        out_logits = full_model(tokenized_titles)
        y_pred = torch.argmax(out_logits.detach().cpu(), dim=1)
        total_preds = torch.concatenate((total_preds, y_pred))
    df_dict["category"].append(category_name)
    df_dict["model"].append(model_type)
    df_dict["test_type"].append("validation")
    df_dict["satire_recall"].append(
        functional.classification.binary_recall(total_preds, total_GT)
        .round(decimals=4)
        .item()
    )
    df_dict["satire_precision"].append(
        functional.classification.binary_precision(total_preds, total_GT)
        .round(decimals=4)
        .item()
    )
    df_dict["mainstream_recall"].append(
        functional.classification.multiclass_recall(
            total_preds, total_GT, num_classes=2, average=None
        )[0]
        .round(decimals=4)
        .item()
    )
    df_dict["mainstream_precision"].append(
        functional.classification.multiclass_precision(
            total_preds, total_GT, num_classes=2, average=None
        )[0]
        .round(decimals=4)
        .item()
    )
    df_dict["mainstream_F1"].append(
        functional.classification.multiclass_f1_score(
            total_preds, total_GT, num_classes=2, average=None
        )[0]
        .round(decimals=4)
        .item()
    )
    df_dict["satire_F1"].append(
        functional.classification.binary_f1_score(total_preds, total_GT)
        .round(decimals=4)
        .item()
    )


if __name__ == "__main__":
    for category_name in ["social", "politic", "sport"]:
        print("#" * 100)
        print(f"TESTING {category_name}")
        print("#" * 100)
        test_category(category_name)

    df = pd.DataFrame.from_dict(df_dict)
    df.to_csv(
        f"../results_classif/meta_learning_{model_type}_results.csv",
        index=False,
        index_label=False,
    )
