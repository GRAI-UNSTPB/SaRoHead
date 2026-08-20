import argparse

import lightning as pl
import pandas as pd
import spacy
import torch
import torch.nn as nn
from aim.pytorch_lightning import AimLogger
from bert_models import *
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset
from torchmetrics import functional


class HeadlineDataset(Dataset):
    def __init__(self, category, df_name):
        self.df_name = df_name
        self.df = pd.read_csv(df_name)
        if category != "all":
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

unsupervised_intermediate = ["standard", "click", "saroco", "scitechbait"]
folder_suffix = {"bert": "BERT", "xlm-roberta": "XLM", "distilled-bert": "Distil"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_category(category_name, dataset, intermediate, model_type):
    df_dict = {
        "category": [],
        "model": [],
        "intermediate_task": [],
        "satire_recall": [],
        "satire_precision": [],
        "satire_F1": [],
    }
    full_module = BERTmodule.load_from_checkpoint(
        f"{category_name}_satire_from_{model_type}_{intermediate}_finetune_False/{model_type}_CKPT.ckpt"
    )
    full_model = full_module.neural_classifier
    tokenizer = full_module.tokenizer
    full_model.eval()
    pl.seed_everything(42)
    if dataset == "regular":
        testData = HeadlineDataset(category_name, "../data/test.csv")
        testLoader = DataLoader(
            testData, shuffle=False, batch_size=32, collate_fn=collate_fn
        )
        loaders = {"test": testLoader}
    else:
        cataData = HeadlineDataset(category_name, "../data/catavencii.csv")
        cataLoader = DataLoader(
            cataData, shuffle=False, batch_size=32, collate_fn=collate_fn
        )
        loaders = {"none": cataLoader}

    total_preds = torch.tensor([])
    total_GT = torch.tensor([])
    pred_df = {
        "title": [],
        "prediction": [],
        "ground_truth": [],
        "satire_prob": [],
    }
    loader = loaders["test"]
    for batch in loader:
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
        probs = torch.softmax(out_logits.detach().cpu(), dim=1)
        satire_prob = probs[:, 1].tolist()
        y_pred = torch.argmax(out_logits.detach().cpu(), dim=1)
        total_preds = torch.concatenate((total_preds, y_pred))
        pred_df["title"].extend(titles)
        pred_df["ground_truth"].extend(labels.tolist())
        pred_df["prediction"].extend(y_pred.tolist())
        pred_df["satire_prob"].extend(satire_prob)
    prec, recall, f1, _ = precision_recall_fscore_support(
        pred_df["ground_truth"], pred_df["prediction"], average="binary"
    )
    df_dict["category"].append(category_name)
    df_dict["intermediate_task"].append(intermediate)
    df_dict["model"].append(model_type)
    df_dict["satire_F1"].append(f1)
    df_dict["satire_precision"].append(prec)
    df_dict["satire_recall"].append(recall)
    pd.DataFrame.from_dict(pred_df).to_csv(
        f"{category_name}_{model_type}_{dataset}.csv"
    )
    return pd.DataFrame.from_dict(df_dict)


if __name__ == "__main__":
    for model_type in ["bert", "distilled-bert", "xlm-roberta"]:
        model_df = pd.DataFrame()
        for category_name in ["social", "politic", "sport"]:
            for dataset_type in unsupervised_intermediate:
                print("#" * 100)
                print(f"TESTING {category_name}")
                print("#" * 100)
                df = test_category(category_name, "regular", dataset_type, model_type)
                model_df = pd.concat([model_df, df])
        model_df.to_csv(f"../results_classif/{model_type}_results.csv")
