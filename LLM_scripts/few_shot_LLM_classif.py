import os
import random
import argparse

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
from torchmetrics import functional
from transformers import set_seed

SEED = 40

set_seed(SEED)

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--model_type")
arg_parser.add_argument("--n_examples", type=int)
parsed_args = arg_parser.parse_args()

model_type = parsed_args.model_type
n_examples = parsed_args.n_examples

model_name = ""
if model_type == "llama2":
    model_name = "OpenLLM-Ro/RoLlama2-7b-Instruct"
elif model_type == "llama3":
    model_name = "OpenLLM-Ro/RoLlama3-8b-Instruct"
elif model_type == "gemma":
    model_name = "OpenLLM-Ro/RoGemma-7b-Instruct"
elif model_type == "mistral":
    model_name = "OpenLLM-Ro/RoMistral-7b-Instruct"
else:
    raise ValueError(f"Unknown model_type: {model_type}")

model = LLM(
    model=model_name,
    dtype=torch.bfloat16,
    seed=SEED,
)

structured_outputs = StructuredOutputsParams(choice=["da", "nu"])

train_df = pd.read_csv("../data/train_data.csv")
train_title_col = "proc_title" if "proc_title" in train_df.columns else ("new_title" if "new_title" in train_df.columns else "title")

train_satire_samples = (
    train_df[train_df["satiric"] == 1]
    .sample(n=n_examples, random_state=SEED)[train_title_col]
    .to_list()
)

non_satire_train_samples = (
    train_df[train_df["satiric"] == 0]
    .sample(n=n_examples, random_state=SEED)[train_title_col]
    .to_list()
)

justification_list = []

def complete_instruction(title):
    return f"""Vei primi un titlu dintr-o știre și trebuie să spui dacă acesta este satiric, sau nu.
Vei răspunde numai cu 'da', sau 'nu', fără a mai fi necesare alte explicații.
Observație: pentru a ține cont exclusiv de structura titlului, entitățile au fost ascunse. Nu cunoști articolul și în stabilirea verdictului te vei folosi exclusiv de titlu, fără a apela la cunoștințe externe. Dacă unele comportamente sunt foarte grave, răspunde cu 'nu'.
Titlu: {title}"""

def classify_category_test(category_name, file_type):
    test_df = pd.read_csv(f"../data/{file_type}_data.csv")
    new_df = test_df[test_df["category"] == category_name].reset_index(drop=True)
    test_title_col = "proc_title" if "proc_title" in new_df.columns else ("new_title" if "new_title" in new_df.columns else "title")

    total_preds, total_GT = [], []
    print("#" * 100 + f" TESTING {category_name} " + "#" * 100)

    for idx in tqdm(range(new_df.shape[0])):
        title = new_df.iloc[idx][test_title_col]
        true_label = int(new_df.iloc[idx]["satiric"])

        conversation = [
            {
                "role": "system",
                "content": "Ești un bun cunoscător al elementelor care definesc satira. Satira are ca scop ridiculizarea unor comportamente, tocmai de aceea se pot regăsi elemente de absurd, ironie, sarcasm."
            }
        ]

        for j in range(n_examples // 2):
            conversation.append(
                {
                    "role": "user",
                    "content": complete_instruction(train_satire_samples[j]),
                }
            )
            conversation.append(
                {
                    "role": "assistant",
                    "content": "da",
                }
            )

        for j in range(n_examples // 2):
            conversation.append(
                {
                    "role": "user",
                    "content": complete_instruction(non_satire_train_samples[j]),
                }
            )
            conversation.append(
                {
                    "role": "assistant",
                    "content": "nu",
                }
            )

        conversation.append(
            {
                "role": "user",
                "content": complete_instruction(title),
            }
        )

        sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            max_tokens=2,
            seed=SEED,
            structured_outputs=structured_outputs,
        )

        outputs = model.chat(conversation, sampling_params)
        generated_text = outputs[0].outputs[0].text.strip().lower()

        boolean_prediction = 1 if generated_text == "da" else 0

        total_preds.append(boolean_prediction)
        total_GT.append(true_label)
        justification_list.append([title, category_name, true_label, boolean_prediction])

    total_preds = torch.tensor(total_preds)
    total_GT = torch.tensor(total_GT)

    return {
        "macro_f1": functional.classification.multiclass_f1_score(
            total_preds, total_GT, num_classes=2, average="macro"
        ).round(decimals=4).item(),
        "satire_recall": functional.classification.binary_recall(
            total_preds, total_GT
        ).round(decimals=4).item(),
        "satire_precision": functional.classification.binary_precision(
            total_preds, total_GT
        ).round(decimals=4).item(),
        "satire_f1": functional.classification.binary_f1_score(
            total_preds, total_GT
        ).round(decimals=4).item(),
        "mainstream_recall": functional.classification.multiclass_recall(
            total_preds, total_GT, average=None, num_classes=2
        )[0].round(decimals=4).item(),
        "mainstream_precision": functional.classification.multiclass_precision(
            total_preds, total_GT, average=None, num_classes=2
        )[0].round(decimals=4).item(),
        "mainstream_f1": functional.classification.multiclass_f1_score(
            total_preds, total_GT, average=None, num_classes=2
        )[0].round(decimals=4).item(),
        "category": f"{category_name}",
    }
test_result_list=[]
val_result_list=[]
keys_list=None      
for category in ['social','politic','sport']:
    category_result_test = classify_category_test(category,"test")
    test_result_list.append(list(category_result_test.values()))
    if(keys_list is None):
        keys_list = list(category_result_test.keys())
test_results = pd.DataFrame(test_result_list,columns=keys_list)
test_results.to_csv(f"~/pentru_articol_satira/results_classif/{model_type}_{n_examples}shot_test_results.csv")