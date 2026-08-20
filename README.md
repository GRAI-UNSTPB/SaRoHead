# SaRoHead: Detecting Satire in a Multi-Domain Romanian News Headline Dataset

[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue)](https://huggingface.co/datasets/GRAI-UNSTPB/SaRoHead)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

## Abstract

> The primary goal of a news headline is to summarize an event in as few words as possible. Depending on the media outlet, a headline can serve as a means to objectively deliver a summary or improve its visibility. For the latter, specific publications may employ stylistic approaches that incorporate the use of sarcasm, irony, and exaggeration, key elements of a satirical approach. As such, even the headline must reflect the tone of the satirical main content. Current approaches for the Romanian language tend to detect the non-conventional tone (i.e., satire and clickbait) of the news content by combining both the main article and the headline. Because we consider a headline to be merely a brief summary of the main article, we investigate in this paper the presence of satirical tone in headlines alone, testing multiple baselines ranging from standard machine learning algorithms to deep learning models. Our experiments show that Bidirectional Transformer models outperform both standard machine-learning approaches and Large Language Models (LLMs), particularly when the meta-learning Reptile approach is employed.

## Dataset

**SaRoHead** (**Sa**tirical **Ro**manian **Head**lines) is a multi-domain benchmark dataset for satire and sarcasm detection in Romanian news headlines across *social*, *politics*, and *sports* domains. It contains news headlines from various satirical and non-satirical news outlets. While gathering the data, we searched over category keywords from *TimesNewRoman*, *Antena 3*, and *Mediafax* explicitly, whereas *DCNews* did not categorize headlines explicitly. As a result, we looked for keywords relevant to each domain. For regular sports news headlines, we used the *sport.ro* news outlet. The corpus is diverse, comprising headlines spanning 2009 to 2025, with a cutoff date of June 2025. Ultimately, our dataset comprises 20,745 news headlines from publicly available Romanian news outlets. 

### Dataset Statistics

- Language: Romanian (`ro`)
- Total Samples: 20,745
- Task: Binary Text Classification (Satire vs. Regular)
- Domains: `social`, `politic`, `sport`
- Time Span: 2009 - June 2025
- Sources:
    - Satire: `timesnewroman.ro`
    - Mainstream / Non-Satiric: `dcnews.ro`, `mediafax.ro`, `antena3.ro`, `sport.ro`
- Dataset splits:

| Split | Regular (0) | Satiric (1) | Total Samples |
| :--- | :---: | :---: | :---: |
| **Train** | 7,775 | 7,745 | **15,520** |
| **Validation** | 1,877 | 2,043 | **3,920** |
| **Test** | 624 | 681 | **1,305** |
| **Total** | **10,276** | **10,469** | **20,745** |

### Dataset Fields

| Field | Type | Description |
| :--- | :--- | :--- |
| `title` | `string` | The original news headline as published. |
| `proc_title` | `string` | Entity-masked headline where Named Entities (people, locations, organizations, facilities) are replaced with generalized semantic tags (e.g. `[PERSON]`, `[GPE]`, `[ORGANIZATION]`, `[FACILITY]`, `[PRODUCT]`, `[EVENT]`, `[NAT_REL_POL]`, `[LOC]`) via `ro_core_news_lg`. |
| `category` | `string` | Topic category (`social`, `politic`, or `sport`). |
| `satiric` | `int64` | Binary ground-truth label: `0` (Regular news) or `1` (Satiric / Humorous news). |

The dataset is also hosted on [Hugging Face](https://huggingface.co/datasets/GRAI-UNSTPB/SaRoHead):

```python
from datasets import load_dataset

# Load SaRoHead dataset from Hugging Face
dataset = load_dataset("GRAI-UNSTPB/SaRoHead")

# Access splits
train_set = dataset["train"]
val_set = dataset["validation"]
test_set = dataset["test"]

# Example record
print(train_set[0])
```

## Setup & Installation

```bash
git clone https://github.com/GRAI-UNSTPB/SaRoHead.git
cd SaRoHead

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download the Romanian spaCy NER model
python -m spacy download ro_core_news_lg
```

## Reproducing Experiments

### Transformer Models (BERT, XLM-RoBERTa, DistilBERT)

Train Transformer models on each news domain:

```bash
cd bert_scripts

# Train Romanian BERT on Social, Politics, and Sports
python train_BERT_satire.py --model_type bert --lr 1e-3 --sch_type linear --inter_task standard --num_epochs 35

# Evaluate trained models
python make_predictions_satire.py
```

Supported `--model_type` values: `bert`, `xlm-roberta`, `distilled-bert`.

### LLM Experiments (LoRA Fine-Tuning & Few-Shot)

#### LoRA Fine-Tuning:

```bash
cd LLM_scripts

# Fine-tune with LoRA (e.g., RoLlama3)
python new_lora_finetune.py --model_type llama3 --lora_r 8 --lora_alpha 8 --epochs 5 --lr 1e-3 --batch_size 8

# Evaluate fine-tuned checkpoint
python test_lora.py --model_name llama3 --category all
```

Supported LLMs: `llama2` (`RoLlama2-7b`), `llama3` (`RoLlama3-8b`), `gemma` (`RoGemma-7b`), `mistral` (`RoMistral-7b`).

#### Few-Shot Evaluation (vLLM):

```bash
pip install vllm
python few_shot_LLM_classif.py --model_type llama3 --n_examples 5
```

### Meta-Learning (Reptile)

Train and evaluate using the Reptile meta-learning algorithm:

```bash
cd reptile

# Meta-learning training
python train_reptile.py --model_type bert --train_style sequential --K 7

# Evaluate meta-learning checkpoints
python satire_results_reptile.py --model_type bert
```

## License

- **Code:** Distributed under the [MIT License](LICENSE).
- **Dataset:** Distributed under the [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/) license. Note that the original titles remain under the copyright of their respective authors and are permitted for academic use only.
