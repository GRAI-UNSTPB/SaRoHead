#!/bin/bash
#SBATCH --job-name=LLM
#SBATCH --time=10:50:00
#SBATCH --gres=gpu:1
#SBATCH --partition=dgxa100
#SBATCH --mem=80GB
#SBATCH --output=out_debug.out
#SBATCH --error=out_err.err
# source /path/to/your_venv/bin/activate

python test_lora.py --model_name "$1" --category "$2"
#
#python lora_finetune_classif.py --model_type $1
