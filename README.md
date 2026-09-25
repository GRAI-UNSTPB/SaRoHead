# SaRoHead: Detecting Satire in a Multi-Domain Romanian News Headline Dataset

[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue)](https://huggingface.co/datasets/GRAI-UNSTPB/SaRoHead)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

## Abstract

> News headlines mainly aim to summarize an event in a news article in a few words. Depending on their goal, media outlets may use headlines to deliver an objective summary or increase the visibility of the news article. For the latter, publications may employ stylistic approaches that incorporate sarcasm, irony, and exaggeration, key elements of a satirical approach. As such, even the headline may reflect the tone of the satirical main content. Current approaches for the Romanian language tend to detect non-conventional tones (i.e., satire and clickbait) by combining the main article and the headline. As a result, investigating headlines in isolation remains largely unexplored. To address this gap, we introduce SaRoHead, a dataset of 20,676 Romanian news headlines spanning social, politics, and sports domains, and analyze satirical tone in headlines alone. We test multiple baselines ranging from standard machine learning algorithms to deep learning models, including transformer-based classifiers and large language models (LLMs), each run with five random seeds. Our experiments show that bidirectional transformer models outperform standard machine-learning approaches. Among BERT-based settings, the meta-learning Reptile approach performs best (pooled F1 90.42 for Romanian BERT). Large language models struggle under the few-shot in-context learning setting. However, when fine-tuned with parameter-efficient techniques, they achieve competitive results, with LoRA RoGemma and RoLlama3 obtaining the highest pooled F1-scores (92.7 and 92.0).

## Dataset

**SaRoHead** (**Sa**tirical **Ro**manian **Head**lines) is a multi-domain benchmark dataset for satire and sarcasm detection in Romanian news headlines across *social*, *politics*, and *sports* domains. It contains news headlines from various satirical and non-satirical news outlets. While gathering the data, we searched over category keywords from *TimesNewRoman*, *Antena 3*, and *Mediafax* explicitly, whereas *DCNews* did not categorize headlines explicitly. As a result, we looked for keywords relevant to each domain. For regular sports news headlines, we used the *sport.ro* news outlet. The corpus is diverse, comprising headlines spanning 2009 to 2025, with a cutoff date of June 2025. We collected 20,745 news headlines from publicly available Romanian news outlets (`data/all_headlines.csv`); after removing near-duplicate headlines, the dataset comprises 20,676 headlines, split 70/15/15 into train/validation/test, stratified by domain and label (see `data_scripts/make_splits.py`).

### Dataset Statistics

- Language: Romanian (`ro`)
- Total Samples: 20,676
- Task: Binary Text Classification (Satire vs. Regular)
- Domains: `social`, `politic`, `sport`
- Time Span: 2009 - June 2025
- Sources:
    - Satire: `timesnewroman.ro`
    - Mainstream / Non-Satiric: `dcnews.ro`, `mediafax.ro`, `antena3.ro`, `sport.ro`
- Dataset splits:

| Split | Regular (0) | Satiric (1) | Total Samples |
| :--- | :---: | :---: | :---: |
| **Train** | 7,169 | 7,303 | **14,472** |
| **Validation** | 1,537 | 1,565 | **3,102** |
| **Test** | 1,537 | 1,565 | **3,102** |
| **Total** | **10,243** | **10,433** | **20,676** |

### Dataset Fields

| Field | Type | Description |
| :--- | :--- | :--- |
| `title` | `string` | The original news headline as published. |
| `proc_title` | `string` | Entity-masked headline where Named Entities are replaced with generalized semantic tags: `[PERSON]`, `[GPE]`, `[ORGANIZATION]`, `[PRODUCT]`, `[NAT_REL_POL]`, `[FACILITY]`, `[LOC]`, `[EVENT]`, `[NAT_GEO_REL]`, `[QUANTITY]`, `[WORK_OF_ART]`, `[LANGUAGE]`, `[PERIOD]`. |
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
```

## Reproducing Experiments

All experiments are run with five seeds (`13, 42, 123, 2024, 7`), one model per news domain. Each run writes its predictions under `results/predictions/<method>/`; the metrics in the paper are computed from these files by the scripts in `analysis_scripts`.

The experiments were run on Google Colab with the notebooks in `colab/` (`01_classic_ml`, `02_bert_ttl`, `03_reptile`, `04_llm_fewshot`, `05_llm_lora`). They clone this repository, read the MLM backbones and the auxiliary datasets from `MyDrive/sarohead/unsupervised_TTL/`, and write the predictions to `MyDrive/sarohead/results/predictions/`; finished configurations are skipped when a notebook is re-run.

### Data Split

`data_scripts/make_splits.py` builds `data/train.csv`, `data/validation.csv` and `data/test.csv` from the 20,745 headlines in `data/all_headlines.csv`. It compares the `proc_title` of every pair of headlines after lowercasing and removing the entity tags, diacritics and punctuation; pairs with a character 5-gram Jaccard similarity of at least 0.8 are grouped as near-duplicates, and only the first headline of each group is kept. This removes 69 headlines.

```bash
python data_scripts/make_splits.py --inputs data/all_headlines.csv
```

### Standard Machine Learning Algorithms

```bash
python ml_scripts/classic_ml.py --seed 42
```

### Transformer Models (BERT, XLM-RoBERTa, DistilBERT)

Train Transformer models on each news domain:

```bash
cd bert_scripts

# Train Romanian BERT on Social, Politics, and Sports
python train_BERT_satire.py --model_type bert --lr 1e-3 --sch_type linear --inter_task standard --num_epochs 35 --seed 42

# Evaluate a saved checkpoint
python make_predictions_satire.py --checkpoint checkpoints/<run>/bert_CKPT.ckpt --model_type bert --category sport --seed 42
```

Supported `--model_type` values: `bert`, `xlm-roberta`, `distilled-bert`.

`--inter_task saroco | click | scitechbait` uses a backbone pretrained with MLM on the headlines of [SaRoCo](https://github.com/MihaelaGaman/SaRoCo), [RoCliCo](https://github.com/dariabroscoteanu/RoCliCo) or [SciTechBaitRo](https://aclanthology.org/2024.nlp4pi-1.17/) (`job_TTL.sh`), expected under `unsupervised_TTL/<task>/backup_<task>_<BERT|Distil|XLM>`.

### LLM Experiments (LoRA Fine-Tuning & Few-Shot)

#### LoRA Fine-Tuning:

```bash
cd LLM_scripts

# Fine-tune with LoRA (e.g., RoLlama3)
python new_lora_finetune.py --model_type llama3 --lora_r 8 --lora_alpha 8 --lora_dropout 0.05 --epochs 5 --lr 1e-3 --batch_size 16 --grad_accum 2 --warmup_ratio 0.2 --seed 42

# Evaluate fine-tuned checkpoint
python test_lora.py --model_name llama3 --category all --seed 42
```

Supported LLMs: `llama2` (`RoLlama2-7b`), `llama3` (`RoLlama3-8b`), `gemma` (`RoGemma-7b`), `mistral` (`RoMistral-7b`).

#### Few-Shot Evaluation (vLLM):

```bash
pip install vllm
python few_shot_LLM_classif.py --model_type llama3 --n_examples 6 --seed 42
```

### Meta-Learning (Reptile)

Train and evaluate using the Reptile meta-learning algorithm. The other tasks are the training headlines of SaRoCo, RoCliCo and SciTechBaitRo, expected as `unsupervised_TTL/{saroco,click,scitechbait}/train.csv`:

```bash
cd reptile

# Meta-learning training (also evaluates the selected checkpoint)
python train_reptile.py --model_type bert --train_style separate --K 7 --seed 42

# Evaluate a saved checkpoint
python satire_results_reptile.py --model_type bert --category sport --seed 42 --checkpoint backup/separate_meta_learning_bert_sport_random_seed42.pt
```

### Results

```bash
python analysis_scripts/aggregate_results.py          # results/metrics_per_run.csv, results/summary.csv, results/tables/*.tex
python analysis_scripts/main_paired_comparisons.py    # paired bootstrap comparisons
python analysis_scripts/make_figures.py               # results/plots/test_set_Reptile_vs_UTTL.pdf
```

### Topic Analysis

The topic model and the TF-IDF analysis use all 20,745 headlines (`data/all_headlines.csv`):

```bash
cd topic_scripts
python bertopic_analysis.py
python tfidf_analysis.py
```

### SHAP

```bash
cd shap_scripts
python reproduce_metrics.py
python run_shap.py
python analyze_shap.py
```

## License

- **Code:** Distributed under the [MIT License](LICENSE).
- **Dataset:** Distributed under the [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/) license. Note that the original titles remain under the copyright of their respective authors and are permitted for academic use only.
