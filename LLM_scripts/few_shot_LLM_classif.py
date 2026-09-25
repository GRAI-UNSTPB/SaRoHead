import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis_scripts.prediction_io import read_split, source_column, write_predictions


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def complete_instruction(title):
    return f"""Vei primi un titlu dintr-o știre și trebuie să spui dacă acesta este satiric, sau nu.
Vei răspunde numai cu 'da', sau 'nu', fără a mai fi necesare alte explicații.
Observație: pentru a ține cont exclusiv de structura titlului, entitățile au fost ascunse. Nu cunoști articolul și în stabilirea verdictului te vei folosi exclusiv de titlu, fără a apela la cunoștințe externe. Dacă unele comportamente sunt foarte grave, răspunde cu 'nu'.
Titlu: {title}"""


def build_conversation(title, satire_examples, mainstream_examples, system_role=True):
    conversation = [{"role": "system", "content": SYSTEM_PROMPT}] if system_role else []
    for example in satire_examples:
        conversation.append({"role": "user", "content": complete_instruction(example)})
        conversation.append({"role": "assistant", "content": "da"})
    for example in mainstream_examples:
        conversation.append({"role": "user", "content": complete_instruction(example)})
        conversation.append({"role": "assistant", "content": "nu"})
    conversation.append({"role": "user", "content": complete_instruction(title)})
    if not system_role:
        conversation[0]["content"] = SYSTEM_PROMPT + "\n\n" + conversation[0]["content"]
    return conversation


def parse_da_nu(text):
    text = (text or "").strip().lower()
    if text.startswith("da"):
        return 1
    if text.startswith("nu"):
        return 0
    if " da" in f" {text}" or text == "d":
        return 1
    return 0


def render_prompts(tokenizer, conversations):
    prompts = []
    for conv in conversations:
        try:
            prompts.append(
                tokenizer.apply_chat_template(
                    conv, tokenize=False, add_generation_prompt=True
                )
            )
        except Exception:
            fixed = []
            system = ""
            for msg in conv:
                if msg["role"] == "system":
                    system = msg["content"]
                    continue
                content = msg["content"]
                if system and msg["role"] == "user" and not fixed:
                    content = system + "\n\n" + content
                    system = ""
                fixed.append({"role": msg["role"], "content": content})
            prompts.append(
                tokenizer.apply_chat_template(
                    fixed, tokenize=False, add_generation_prompt=True
                )
            )
    return prompts


def make_vllm_sampling_params(seed):
    from vllm import SamplingParams

    base = {"temperature": 0.0, "top_p": 1.0, "max_tokens": 2, "seed": seed}
    try:
        from vllm.sampling_params import StructuredOutputsParams

        return SamplingParams(
            **base, structured_outputs=StructuredOutputsParams(choice=["da", "nu"])
        )
    except Exception as exc:
        print(f"StructuredOutputsParams unavailable ({exc!r})", flush=True)
    try:
        from vllm.sampling_params import GuidedDecodingParams

        return SamplingParams(
            **base, guided_decoding=GuidedDecodingParams(choice=["da", "nu"])
        )
    except Exception as exc:
        print(f"GuidedDecodingParams unavailable ({exc!r})", flush=True)
    for kwargs in (
        {"guided_choice": ["da", "nu"]},
        {"guided_regex": r"(da|nu)"},
    ):
        try:
            return SamplingParams(**base, **kwargs)
        except Exception as exc:
            print(f"{kwargs} unavailable ({exc!r})", flush=True)
    print("unconstrained greedy decoding", flush=True)
    return SamplingParams(**base)


class VllmBackend:
    def __init__(self, model_name, seed, dtype, enforce_eager=False):
        from vllm import LLM

        llm_kwargs = {
            "model": model_name,
            "dtype": dtype,
            "seed": seed,
            "trust_remote_code": True,
        }
        if enforce_eager:
            llm_kwargs["enforce_eager"] = True
        llm_kwargs["gpu_memory_utilization"] = 0.90
        print(f"loading {model_name} with {llm_kwargs}", flush=True)
        self.model = LLM(**llm_kwargs)
        self.params = make_vllm_sampling_params(seed)
        self.tokenizer = self.model.get_tokenizer()

    def classify(self, conversations):
        try:
            outputs = self.model.chat(
                conversations, sampling_params=self.params, use_tqdm=False
            )
            return [parse_da_nu(o.outputs[0].text) for o in outputs]
        except TypeError:
            try:
                outputs = self.model.chat(conversations, self.params, use_tqdm=False)
                return [parse_da_nu(o.outputs[0].text) for o in outputs]
            except Exception as exc:
                print(f"chat() failed ({exc!r}); using generate()", flush=True)
        except Exception as exc:
            print(f"chat() failed ({exc!r}); using generate()", flush=True)
        prompts = render_prompts(self.tokenizer, conversations)
        outputs = self.model.generate(
            prompts, sampling_params=self.params, use_tqdm=False
        )
        return [parse_da_nu(o.outputs[0].text) for o in outputs]


class HfBackend:
    def __init__(self, model_name, seed, dtype):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        set_seed(seed)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "auto": "auto",
        }[dtype]
        device_map = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"loading {model_name}", flush=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        ).eval()
        self.yes_ids = [
            self.tokenizer.encode(t, add_special_tokens=False)[0]
            for t in ("da", "Da", " da")
        ]
        self.no_ids = [
            self.tokenizer.encode(t, add_special_tokens=False)[0]
            for t in ("nu", "Nu", " nu")
        ]

    @torch.no_grad()
    def classify(self, conversations):
        prompts = render_prompts(self.tokenizer, conversations)
        enc = self.tokenizer(
            prompts, return_tensors="pt", padding=True, add_special_tokens=False
        )
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        logits = self.model(**enc).logits[:, -1, :].float()
        yes = torch.logsumexp(logits[:, self.yes_ids], dim=1)
        no = torch.logsumexp(logits[:, self.no_ids], dim=1)
        return (yes > no).long().tolist()


def build_backend(backend, model_name, seed, dtype):
    if backend == "hf":
        return HfBackend(model_name, seed, dtype)
    try:
        return VllmBackend(model_name, seed, dtype)
    except Exception as exc:
        print(f"vllm failed ({exc!r}); using transformers", flush=True)
        return HfBackend(model_name, seed, dtype if dtype != "auto" else "bfloat16")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", required=True, choices=list(MODEL_NAMES))
    parser.add_argument("--model_override", default=None)
    parser.add_argument("--backend", default="vllm", choices=["vllm", "hf"])
    parser.add_argument(
        "--dtype", default="bfloat16", choices=["bfloat16", "float16", "auto"]
    )
    parser.add_argument("--n_examples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", default="../data")
    parser.add_argument("--split_family", default="random")
    parser.add_argument("--eval_splits", nargs="+", default=["test"])
    parser.add_argument(
        "--categories", nargs="+", default=["social", "politic", "sport"]
    )
    parser.add_argument("--variant_suffix", default="")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--no_system_role", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    train_df = read_split(args.data_dir, "train")
    model_name = args.model_override or MODEL_NAMES[args.model_type]
    no_system = args.no_system_role or args.model_type == "gemma"
    backend = build_backend(args.backend, model_name, args.seed, args.dtype)
    half = args.n_examples // 2
    variant = f"n{args.n_examples}" + (
        f"__{args.variant_suffix}" if args.variant_suffix else ""
    )

    for category in args.categories:
        cat_train = train_df[train_df["category"] == category]
        satire_examples = (
            cat_train[cat_train["satiric"] == 1]
            .sample(n=half, random_state=args.seed)["new_title"]
            .tolist()
            if half
            else []
        )
        mainstream_examples = (
            cat_train[cat_train["satiric"] == 0]
            .sample(n=half, random_state=args.seed)["new_title"]
            .tolist()
            if half
            else []
        )
        print(f"{category} satire examples: {satire_examples}", flush=True)
        print(f"{category} mainstream examples: {mainstream_examples}", flush=True)

        for eval_split in args.eval_splits:
            df = read_split(args.data_dir, eval_split, category)
            if args.limit:
                df = df.head(args.limit)
            titles = df["new_title"].fillna("").tolist()
            preds = []
            for start in range(0, len(titles), args.batch_size):
                batch = titles[start : start + args.batch_size]
                conversations = [
                    build_conversation(
                        t, satire_examples, mainstream_examples, not no_system
                    )
                    for t in batch
                ]
                preds.extend(backend.classify(conversations))
            path = write_predictions(
                method="few_shot",
                model=PAPER_NAMES[args.model_type],
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
            print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
