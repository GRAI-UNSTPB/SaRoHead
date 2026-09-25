import argparse
import sys
from pathlib import Path

import torch
from new_lora_finetune import (
    MODEL_NAMES,
    PAPER_NAMES,
    SYSTEM_PROMPT,
    adapter_dir,
    user_prompt,
)
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis_scripts.prediction_io import read_split, source_column, write_predictions

torch.backends.cuda.enable_cudnn_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)


def build_prompt(tokenizer, title):
    chat = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt(title)},
    ]
    return tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )


@torch.no_grad()
def classify(model, tokenizer, titles, batch_size):
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    preds = []
    for start in tqdm(range(0, len(titles), batch_size)):
        prompts = [
            build_prompt(tokenizer, t) for t in titles[start : start + batch_size]
        ]
        enc = tokenizer(
            prompts, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(model.device)
        out = model.generate(
            **enc,
            max_new_tokens=2,
            min_new_tokens=1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        generated = tokenizer.batch_decode(
            out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True
        )
        preds.extend(1 if g.strip().lower().startswith("da") else 0 for g in generated)
    return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="all")
    parser.add_argument("--category", nargs="+", default=["all"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", default="../data")
    parser.add_argument("--split_family", default="random")
    parser.add_argument("--eval_splits", nargs="+", default=["test"])
    parser.add_argument("--variant_suffix", default="")
    parser.add_argument("--adapter_tag", default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--model_override", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    categories = (
        ["social", "politic", "sport"]
        if "all" in args.category
        else list(args.category)
    )
    models = list(MODEL_NAMES) if args.model_name == "all" else [args.model_name]
    variant = "lora" + (f"__{args.variant_suffix}" if args.variant_suffix else "")

    for model_type in models:
        model_name = args.model_override or MODEL_NAMES[model_type]
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        device_map = "cuda" if torch.cuda.is_available() else "cpu"
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map=device_map
        )
        for category in categories:
            adapter = adapter_dir(
                model_type, category, args.adapter_tag or args.split_family, args.seed
            )
            print(f"Evaluating {adapter}")
            model = PeftModel.from_pretrained(
                base_model, adapter, is_trainable=False
            ).eval()
            for eval_split in args.eval_splits:
                df = read_split(args.data_dir, eval_split, category)
                if args.limit:
                    df = df.head(args.limit)
                titles = df["new_title"].fillna("").tolist()
                preds = classify(model, tokenizer, titles, args.batch_size)
                path = write_predictions(
                    method="lora",
                    model=PAPER_NAMES[model_type],
                    variant=variant,
                    category=category,
                    seed=args.seed,
                    split_family=args.split_family,
                    eval_split=eval_split,
                    titles=titles,
                    ground_truth=df["satiric"].astype(int),
                    prediction=preds,
                    source=source_column(df),
                )
                print(f"wrote {path}")
            model.unload()


if __name__ == "__main__":
    main()
