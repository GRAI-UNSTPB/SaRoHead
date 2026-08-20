#!/bin/bash
#SBATCH --job-name=reptile
#SBATCH --time=05:50:00
#SBATCH --gres=gpu:1
#SBATCH --partition=dgxh100
#SBATCH --mem=80GB
#SBATCH --output=out_debug.out
#SBATCH --error=out_err.err
# source /path/to/your_venv/bin/activate
echo 123
#python interpret_model.py
python satire_results_reptile.py --model_type bert
python satire_results_reptile.py --model_type xlm-roberta
python satire_results_reptile.py --model_type distilled-bert
#train_style=("sequential")
#for train_style in "${train_style[@]}"; do
#python train_reptile.py --model_type "$1" --train_style "$2" --K 7
#python train_reptile.py --model_type "$1" --train_style "$2" --K 7
