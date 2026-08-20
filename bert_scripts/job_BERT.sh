#!/bin/bash
#SBATCH --job-name=eval_BERT
#SBATCH --time=09:50:00
#SBATCH --gres=gpu:1
#SBATCH --partition=dgxa100
#SBATCH --mem=80GB
#SBATCH --output=out_debug.out
#SBATCH --error=out_err.err
# source /path/to/your_venv/bin/activate
python make_predictions_satire.py

#inter_tasks=("standard" "click" "saroco" "scitechbait")

#for inter_task in "${inter_tasks[@]}"; do
    #python train_BERT_satire.py  --model_type "$1" \
    #--lr 1e-3 --sch_type "linear" --inter_task $inter_task --num_epochs 35
#done
