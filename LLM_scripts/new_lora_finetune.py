# we import LoRA dependencies
# we import HuggingFace modules
import argparse
import logging

from aim.hugging_face import AimCallback
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer


def return_full_model_name(model_type):
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


def build_chat(example):
    yes_or_no = "da" if example["satiric"] == 1 else "nu"
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
        {"role": "assistant", "content": f"""{yes_or_no}"""},
    ]
    return {"messages": chat}


if __name__ == "__main__":
    logging.basicConfig(
        filename="debug.log", format="%(asctime)s - %(message)s", level=logging.INFO
    )
    logger = logging.getLogger()
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--model_type")
    arg_parser.add_argument("--lora_r", type=int)
    arg_parser.add_argument("--lora_alpha", type=int)
    arg_parser.add_argument("--epochs", type=int)
    arg_parser.add_argument("--warmup_ratio", type=float)
    arg_parser.add_argument("--lr", type=float)
    arg_parser.add_argument("--batch_size", type=int, default=8)
    parsed_args = arg_parser.parse_args()
    full_model_name = return_full_model_name(parsed_args.model_type)
    #### initialize LoRA Config
    lora_config = LoraConfig(
        r=parsed_args.lora_r,
        lora_alpha=parsed_args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
        lora_dropout=0.05,
    )
    ##### init the dataset and tokenizer
    dataset = load_dataset(
        "csv",
        data_files={
            "train": "../data/train.csv",
            "validation": "../data/validation.csv",
            "test": "../data/test.csv",
        },
    )

    tokenizer = AutoTokenizer.from_pretrained(full_model_name)

    categories = ["social", "politic", "sport"]
    for category in categories:
        filtered_data = dataset.filter(lambda example: example["category"] == category)
        cols_to_remove = [
            c for c in filtered_data["train"].column_names if c != "messages"
        ]
        new_filtered_data = filtered_data.map(
            build_chat, batched=False, remove_columns=cols_to_remove
        )

        set_seed(42)

        sft_config = SFTConfig(
            output_dir=f"check_{parsed_args.model_type}_{category}",
            do_train=True,
            do_eval=True,
            num_train_epochs=parsed_args.epochs,
            weight_decay=0.01,
            logging_steps=50,
            seed=42,
            data_seed=42,
            bf16=True,
            bf16_full_eval=True,
            eval_on_start=True,
            eval_strategy="steps",
            eval_steps=100,
            save_total_limit=3,
            learning_rate=parsed_args.lr,
            save_strategy="best",
            greater_is_better=False,
            load_best_model_at_end=True,
            warmup_ratio=parsed_args.warmup_ratio,
            per_device_train_batch_size=parsed_args.batch_size,
            per_device_eval_batch_size=parsed_args.batch_size,
        )
        sft_trainer = SFTTrainer(
            model=full_model_name,
            peft_config=lora_config,
            train_dataset=new_filtered_data["train"],
            eval_dataset=new_filtered_data["validation"],
            callbacks=[
                AimCallback(experiment=f"lora_{parsed_args.model_type}_{category}")
            ],
            args=sft_config,
        )

        sft_trainer.log(
            {
                "lora_r": parsed_args.lora_r,
                "lora_alpha": parsed_args.lora_alpha,
                "lora_dropout": 0.05,
                "warmup_ratio": parsed_args.warmup_ratio,
                "final_lr": parsed_args.lr,
            }
        )
        sft_trainer.train()
        sft_trainer.save_model(f"main_model_{parsed_args.model_type}_{category}")
