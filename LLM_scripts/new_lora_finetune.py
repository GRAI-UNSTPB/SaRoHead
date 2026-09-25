import argparse
import inspect
import logging
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis_scripts.prediction_io import RELEASED_TEXT_COLUMN, TEXT_COLUMN, split_path


def build_sft_config(**kwargs):
    params = inspect.signature(SFTConfig.__init__).parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return SFTConfig(**kwargs)
    return SFTConfig(**{k: v for k, v in kwargs.items() if k in params})


try:
    from aim.hugging_face import AimCallback
except Exception:
    AimCallback = None

MODEL_NAMES = {
    "llama2": "OpenLLM-Ro/RoLlama2-7b-Instruct",
    "llama3": "OpenLLM-Ro/RoLlama3-8b-Instruct",
    "gemma": "OpenLLM-Ro/RoGemma-7b-Instruct",
    "mistral": "OpenLLM-Ro/RoMistral-7b-Instruct",
}
PAPER_NAMES = {
    "llama2": "RoLlama2",
    "llama3": "RoLlama3",
    "gemma": "RoGemma",
    "mistral": "RoMistral",
}

SYSTEM_PROMPT = (
    "Ești un bun cunoscător al elementelor care definesc satira. Satira are ca scop ridiculizarea unor "
    "comportamente, tocmai de aceea se pot regăsi elemente de absurd, ironie, sarcasm."
)


def user_prompt(title):
    return f"""Vei primi un titlu dintr-o știre și trebuie să spui dacă acesta este satiric, sau nu. 
Vei răspunde numai cu 'da', sau 'nu', fără a mai fi necesare alte explicații.
Observație: pentru a ține cont exclusiv de structura titlului, entitățile au fost ascunse. Nu cunoști articolul și în stabilirea verdictului te vei folosi exclusiv de titlu, fără a apela la cunoștințe externe.
Titlu:{title}"""


def return_full_model_name(model_type):
    return MODEL_NAMES[model_type]


def headline(example):
    return example.get(RELEASED_TEXT_COLUMN) or example.get(TEXT_COLUMN) or ""


def build_chat(example):
    yes_or_no = "da" if example["satiric"] == 1 else "nu"
    chat = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt(headline(example))},
        {"role": "assistant", "content": yes_or_no},
    ]
    return {"messages": chat}


def adapter_dir(model_type, category, tag, seed):
    return f"main_model_{model_type}_{category}_{tag}_seed{seed}"


if __name__ == "__main__":
    logging.basicConfig(
        filename="debug.log", format="%(asctime)s - %(message)s", level=logging.INFO
    )
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--model_type", required=True, choices=list(MODEL_NAMES))
    arg_parser.add_argument("--lora_r", type=int, default=8)
    arg_parser.add_argument("--lora_alpha", type=int, default=8)
    arg_parser.add_argument("--lora_dropout", type=float, default=0.05)
    arg_parser.add_argument("--epochs", type=int, default=5)
    arg_parser.add_argument("--warmup_ratio", type=float, default=0.0)
    arg_parser.add_argument("--lr", type=float, default=1e-3)
    arg_parser.add_argument("--batch_size", type=int, default=32)
    arg_parser.add_argument("--seed", type=int, default=42)
    arg_parser.add_argument("--data_dir", default="../data")
    arg_parser.add_argument("--split_family", default="random")
    arg_parser.add_argument("--adapter_tag", default=None)
    arg_parser.add_argument(
        "--categories", nargs="+", default=["social", "politic", "sport"]
    )
    arg_parser.add_argument("--use_aim", action="store_true")
    arg_parser.add_argument("--model_override", default=None)
    arg_parser.add_argument("--max_steps", type=int, default=-1)
    arg_parser.add_argument("--limit", type=int, default=None)
    arg_parser.add_argument("--eval_steps", type=int, default=100)
    arg_parser.add_argument("--grad_accum", type=int, default=1)
    parsed_args = arg_parser.parse_args()

    full_model_name = parsed_args.model_override or return_full_model_name(
        parsed_args.model_type
    )
    lora_config = LoraConfig(
        r=parsed_args.lora_r,
        lora_alpha=parsed_args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
        lora_dropout=parsed_args.lora_dropout,
    )
    dataset = load_dataset(
        "csv",
        data_files={
            "train": str(split_path(parsed_args.data_dir, "train")),
            "validation": str(split_path(parsed_args.data_dir, "val")),
            "test": str(split_path(parsed_args.data_dir, "test")),
        },
    )
    tokenizer = AutoTokenizer.from_pretrained(full_model_name)

    for category in parsed_args.categories:
        filtered_data = dataset.filter(lambda example: example["category"] == category)
        if parsed_args.limit:
            filtered_data["train"] = filtered_data["train"].select(
                range(min(parsed_args.limit, len(filtered_data["train"])))
            )
            filtered_data["validation"] = filtered_data["validation"].select(
                range(min(parsed_args.limit, len(filtered_data["validation"])))
            )
        drop_columns = [c for c in filtered_data["train"].column_names]
        new_filtered_data = filtered_data.map(
            build_chat, batched=False, remove_columns=drop_columns
        )

        set_seed(parsed_args.seed)
        out_name = adapter_dir(
            parsed_args.model_type,
            category,
            parsed_args.adapter_tag or parsed_args.split_family,
            parsed_args.seed,
        )
        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        sft_config = build_sft_config(
            output_dir=f"check_{out_name}",
            do_train=True,
            do_eval=True,
            num_train_epochs=parsed_args.epochs,
            weight_decay=0.01,
            logging_steps=50,
            seed=parsed_args.seed,
            data_seed=parsed_args.seed,
            bf16=use_bf16,
            bf16_full_eval=use_bf16,
            fp16=not use_bf16 and torch.cuda.is_available(),
            eval_on_start=True,
            eval_strategy="steps",
            eval_steps=parsed_args.eval_steps,
            save_strategy="steps",
            save_steps=parsed_args.eval_steps,
            save_total_limit=2,
            learning_rate=parsed_args.lr,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            load_best_model_at_end=True,
            warmup_ratio=parsed_args.warmup_ratio,
            per_device_train_batch_size=parsed_args.batch_size,
            per_device_eval_batch_size=parsed_args.batch_size,
            gradient_accumulation_steps=parsed_args.grad_accum,
            max_steps=parsed_args.max_steps,
            report_to="none",
            loss_type="nll",
        )
        callbacks = []
        if parsed_args.use_aim and AimCallback is not None:
            callbacks.append(AimCallback(experiment=f"lora_{out_name}"))
        sft_trainer = SFTTrainer(
            model=full_model_name,
            peft_config=lora_config,
            train_dataset=new_filtered_data["train"],
            eval_dataset=new_filtered_data["validation"],
            callbacks=callbacks,
            args=sft_config,
        )
        sft_trainer.log(
            {
                "lora_r": parsed_args.lora_r,
                "lora_alpha": parsed_args.lora_alpha,
                "lora_dropout": parsed_args.lora_dropout,
                "warmup_ratio": parsed_args.warmup_ratio,
                "final_lr": parsed_args.lr,
                "seed": parsed_args.seed,
            }
        )
        sft_trainer.train()
        sft_trainer.save_model(out_name)
