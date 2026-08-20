# SaRoHead: Detecting Satire in a Multi-Domain Romanian News Headline Dataset

[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue)](https://huggingface.co/datasets/GRAI-UNSTPB/sarohead)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

Official repository containing the code, experimental setups, and benchmark dataset for the paper:  
**"SaRoHead: Detecting Satire in a Multi-Domain Romanian News Headline Dataset"**.

---

## Abstract

> The primary goal of a news headline is to summarize an event in as few words as possible. Depending on the media outlet, a headline can serve as a means to objectively deliver a summary or improve its visibility. For the latter, specific publications may employ stylistic approaches that incorporate the use of sarcasm, irony, and exaggeration, key elements of a satirical approach. As such, even the headline must reflect the tone of the satirical main content. Current approaches for the Romanian language tend to detect the non-conventional tone (i.e., satire and clickbait) of the news content by combining both the main article and the headline. Because we consider a headline to be merely a brief summary of the main article, we investigate in this paper the presence of satirical tone in headlines alone, testing multiple baselines ranging from standard machine learning algorithms to deep learning models. Our experiments show that Bidirectional Transformer models outperform both standard machine-learning approaches and Large Language Models (LLMs), particularly when the meta-learning Reptile approach is employed.

---

## Repository Structure

```text
.
├── data/                       # Dataset splits (train.csv, validation.csv, test.csv)
├── bert_scripts/               # BERT, XLM-RoBERTa, and DistilBERT training & evaluation
│   ├── bert_models.py          # PyTorch Lightning module for BERT classifiers
│   ├── train_BERT_satire.py    # Fine-tuning on domain categories (Social, Politics, Sports)
│   ├── make_predictions_satire.py # Evaluation & metric computation
│   └── job_BERT.sh             # SLURM execution script
├── LLM_scripts/                # Large Language Model experiments
│   ├── new_lora_finetune.py    # LoRA fine-tuning for RoLlama2, RoLlama3, RoGemma, RoMistral
│   ├── few_shot_LLM_classif.py # Few-shot inference via structured outputs
│   ├── test_lora.py            # Evaluation of LoRA fine-tuned LLMs
│   └── job_lora.sh             # SLURM execution script
├── reptile/                    # Meta-Learning & Interpretability
│   ├── train_reptile.py        # Reptile meta-learning algorithm implementation
│   ├── satire_results_reptile.py # Meta-learning evaluation across categories
│   ├── interpret_model.py      # Model interpretability and error analysis
│   └── job_reptile.sh          # SLURM execution script
├── job_TTL.sh                  # Task Transfer Learning runner
├── requirements.txt            # Core dependencies
└── README.md
```

---

## Dataset

The **SaRoHead** dataset comprises **20,745 Romanian news headlines** spanning 2009 to June 2025 across three key domains: **Social**, **Politics**, and **Sports**.

### Dataset Splits

| Split | Regular (`0`) | Satiric (`1`) | Total Samples |
| :--- | :---: | :---: | :---: |
| **Train** | 7,775 | 7,745 | **15,520** |
| **Validation** | 1,877 | 2,043 | **3,920** |
| **Test** | 624 | 681 | **1,305** |
| **Total** | **10,276** | **10,469** | **20,745** |

### Data Fields

| Field | Type | Description |
| :--- | :--- | :--- |
| `title` | `string` | The original raw news headline as published. |
| `proc_title` | `string` | Entity-masked headline where Named Entities (`[PERSON]`, `[GPE]`, `[ORGANIZATION]`, `[FACILITY]`, `[PRODUCT]`, `[EVENT]`) are replaced using `ro_core_news_lg`. |
| `category` | `string` | News domain topic (`social`, `politic`, or `sport`). |
| `satiric` | `int64` | Ground-truth binary label: `0` (Regular news) or `1` (Satiric / Humorous news). |

The dataset is also hosted on [Hugging Face](https://huggingface.co/datasets/GRAI-UNSTPB/sarohead):

```python
from datasets import load_dataset

dataset = load_dataset("GRAI-UNSTPB/sarohead")
```

---

## Setup & Installation

### 1. Environment Setup

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

---

## Reproducing Experiments

### 1. Transformer Models (BERT, XLM-RoBERTa, DistilBERT)

Train Transformer models on each news domain:

```bash
cd bert_scripts

# Train Romanian BERT on Social, Politics, and Sports
python train_BERT_satire.py --model_type bert --lr 1e-3 --sch_type linear --inter_task standard --num_epochs 35

# Evaluate trained models
python make_predictions_satire.py
```

Supported `--model_type` values: `bert`, `xlm-roberta`, `distilled-bert`.

---

### 2. LLM Experiments (LoRA Fine-Tuning & Few-Shot)

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

---

### 3. Meta-Learning (Reptile)

Train and evaluate using the Reptile meta-learning algorithm:

```bash
cd reptile

# Meta-learning training
python train_reptile.py --model_type bert --train_style sequential --K 7

# Evaluate meta-learning checkpoints
python satire_results_reptile.py --model_type bert
```

---

## Citation

If you find this dataset or codebase useful in your research, please cite our paper:

```bibtex
@article{sarohead2026,
  title={SaRoHead: Detecting Satire in a Multi-Domain Romanian News Headline Dataset},
  author={Your Name and Collaborators},
  year={2026},
  publisher={GitHub / Hugging Face}
}
```

---

## License

- **Code:** Distributed under the [MIT License](LICENSE).
- **Dataset:** Distributed under the [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/) license.
