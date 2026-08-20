import argparse
import logging

import pandas as pd
import torch
from new_lora_finetune import build_chat
from peft import PeftModel
from torch.utils.data import DataLoader, Dataset
from torchmetrics import functional
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

torch.backends.cuda.enable_cudnn_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)


class HeadlineDataset(Dataset):
    def __init__(self, category, df_name, tokenizer):
        self.df = pd.read_csv(df_name)
        if category != "all":
            self.df = self.df[self.df["category"] == category]
        title_col = (
            "proc_title"
            if "proc_title" in self.df.columns
            else ("new_title" if "new_title" in self.df.columns else "title")
        )
        self.titles = self.df[title_col].to_list()
        self.labels_binary = self.df["satiric"].to_list()
        self.labels = ["da" if label == 1 else "nu" for label in self.labels_binary]
        self.tokenizer = tokenizer
        self.list_size = len(self.labels)

    def __getitem__(self, idx):
        example = self.df.iloc[idx, :].to_dict()
        title_text = (
            example.get("proc_title")
            or example.get("new_title")
            or example.get("title", "")
        )
        chat = [
            {
                "role": "system",
                "content": "Ești un bun cunoscător al elementelor care definesc satira. Satira are ca scop ridiculizarea unor comportamente, tocmai de aceea se pot regăsi elemente de absurd, ironie, sarcasm.",
            },
            {
                "role": "user",
                "content": f"""Vei primi un titlu dintr-o știre și trebuie să spui dacă acesta este satiric, sau nu. 
Vei răspunde numai cu 'da', sau 'nu', fără a mai fi necesare alte explicații.
Observație: pentru a ține cont exclusiv de structura titlului, entitățile au fost ascunse. Nu cunoști articolul și în stabilirea verdictului te vei folosi exclusiv de titlu, fără a apela la cunoștințe externe.
Titlu:{title_text}""",
            },
        ]
        return chat

    def __len__(self):
        return self.list_size


df_dict = {
    "category": [],
    "model": [],
    "satire recall": [],
    "satire precision": [],
    "satire F1": [],
    "mainstream recall": [],
    "mainstream precision": [],
    "mainstream F1": [],
    "test type": [],
}


def return_model_name(model_type):
    model_name = ""
    if model_type == "llama2":
        model_name = "OpenLLM-Ro/RoLlama2-7b-Instruct"
    elif model_type == "llama3":
        model_name = "OpenLLM-Ro/RoLlama3-8b-Instruct"
    elif model_type == "gemma":
        model_name = "OpenLLM-Ro/RoGemma-7b-Instruct"
    elif model_type == "mistral":
        model_name = "OpenLLM-Ro/RoMistral-7b-Instruct"
    return model_name


def main():
    metrics = ["NS_precision", "NS_recall", "S_precision", "S_recall"]
    logging.basicConfig(
        filename="lora_debug.log",
        format="%(asctime)s: %(message)s",
        level=logging.INFO,
        filemode="w",
    )
    logger = logging.getLogger(__name__)
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--model_name", type=str)
    arg_parser.add_argument("--category", type=str)
    args = arg_parser.parse_args()
    model_list, category_list = [], []
    if args.category == "all":
        category_list = ["social", "politic", "sport"]
    else:
        category_list = [args.category]
    if args.model_name == "all":
        model_list = ["llama2", "llama3", "gemma", "mistral"]
    else:
        model_list = [args.model_name]

    for model_type in model_list:
        model_name = return_model_name(model_type)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="cuda"
        )
        for category in category_list:
            print(f"Ongoing {category}")
            wrapped_model = PeftModel.from_pretrained(
                base_model, f"main_model_{model_type}_{category}", is_trainable=False
            ).cuda()
            wrapped_model.eval()
            hf_pipeline = pipeline(
                task="text-generation",
                model=wrapped_model,
                device="cuda",
                batch_size=1,
                tokenizer=tokenizer,
            )

            def collate_fn(batch):
                chats = []
                for chat in batch:
                    chats.append(chat)
                return chats

            testData = HeadlineDataset(category, "../data/test.csv", tokenizer)
            testLoader = DataLoader(
                testData, batch_size=1, shuffle=False, collate_fn=collate_fn
            )
            to_loader_dict = {"test": [testLoader, testData]}

            for name, loaders in to_loader_dict.items():
                total_GT = loaders[1].labels_binary
                total_preds = []
                print(len(loaders[1]))
                for batch in tqdm(loaders[0]):
                    out = hf_pipeline(
                        batch,
                        return_full_text=False,
                        min_new_tokens=1,
                        max_new_tokens=2,
                        do_sample=False,
                    )
                    """out = wrapped_model.generate(input_ids=input_ids,
                                                do_sample=False,
                                                max_new_tokens=1)
                    logger.info(tokenizer.decode(out[0],skip_special_tokens=True))
                    prediction=tokenizer.decode(out[0],skip_special_tokens=False).strip()"""
                    print(out)
                    out = out[0][0]["generated_text"]
                    # print(out)
                    if out == "da":
                        total_preds.append(1)
                    else:
                        total_preds.append(0)
                total_GT = torch.tensor(total_GT)
                total_preds = torch.tensor(total_preds)
                df_dict["category"].append(category)
                df_dict["model"].append(model_type)
                df_dict["test type"].append(name)
                df_dict["satire recall"].append(
                    functional.classification.binary_recall(total_preds, total_GT)
                    .round(decimals=4)
                    .item()
                )
                df_dict["satire precision"].append(
                    functional.classification.binary_precision(total_preds, total_GT)
                    .round(decimals=4)
                    .item()
                )
                df_dict["satire F1"].append(
                    functional.classification.binary_f1_score(total_preds, total_GT)
                    .round(decimals=4)
                    .item()
                )
                df_dict["mainstream recall"].append(
                    functional.classification.multiclass_recall(
                        total_preds, total_GT, num_classes=2, average=None
                    )[0]
                    .round(decimals=4)
                    .item()
                )
                df_dict["mainstream precision"].append(
                    functional.classification.multiclass_precision(
                        total_preds, total_GT, num_classes=2, average=None
                    )[0]
                    .round(decimals=4)
                    .item()
                )
                df_dict["mainstream F1"].append(
                    functional.classification.multiclass_f1_score(
                        total_preds, total_GT, num_classes=2, average=None
                    )[0]
                    .round(decimals=4)
                    .item()
                )

    df = pd.DataFrame.from_dict(df_dict)
    df.to_csv("../results_classif/lora_results.csv", index=False, index_label=False)


if __name__ == "__main__":
    main()
