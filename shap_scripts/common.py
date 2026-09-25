import os
import sys

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from analysis_scripts.prediction_io import read_split

MODEL_NAME = "dumitrescustefan/bert-base-romanian-uncased-v1"
WEIGHTS_DIR = os.environ.get("SAROHEAD_WEIGHTS_DIR", "../reptile/backup")
DATA_DIR = os.environ.get("SAROHEAD_DATA_DIR", "../data")
SPLIT_FAMILY = os.environ.get("SAROHEAD_SPLIT_FAMILY", "random")
SEED = int(os.environ.get("SAROHEAD_SEED", "42"))
WEIGHTS_PATTERN = os.environ.get(
    "SAROHEAD_WEIGHTS_PATTERN",
    f"separate_meta_learning_bert_{{category}}_{SPLIT_FAMILY}_seed{SEED}.pt",
)
CATEGORIES = ["social", "politic", "sport"]
MAX_LENGTH = 128

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


class NeuralClassifier(nn.Module):
    def __init__(self, model_name, tokenizer_length, num_labels):
        super().__init__()
        self.backbone_model = AutoModel.from_pretrained(model_name)
        self.backbone_model.resize_token_embeddings(tokenizer_length)
        self.linear_layer = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.backbone_model(input_ids=input_ids, attention_mask=attention_mask)
        cls_embeddings = out.last_hidden_state[:, 0, :]
        return self.linear_layer(cls_embeddings)


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, do_lower_case=True)
    tokenizer.add_tokens(ADDED_TOKENS)
    return tokenizer


def load_model(category, device):
    tokenizer = load_tokenizer()
    state_dict = torch.load(
        f"{WEIGHTS_DIR}/{WEIGHTS_PATTERN.format(category=category)}",
        weights_only=True,
        map_location="cpu",
    )
    state_dict = {k: v.to(torch.float32) for k, v in state_dict.items()}
    model = NeuralClassifier(MODEL_NAME, len(tokenizer), num_labels=2)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model, tokenizer


def load_split(category, split):
    df = read_split(DATA_DIR, split, category)
    return df["new_title"].fillna("").tolist(), df["satiric"].to_numpy()


def make_predict_proba(model, tokenizer, device, batch_size=64):
    @torch.no_grad()
    def predict_proba(texts):
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
        texts = [str(t) for t in texts]
        probs = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            enc = tokenizer(
                batch,
                return_tensors="pt",
                padding="longest",
                max_length=MAX_LENGTH,
                truncation=True,
            ).to(device)
            logits = model(enc["input_ids"], enc["attention_mask"])
            probs.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy())
        return np.concatenate(probs)

    return predict_proba
