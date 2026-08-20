#!/bin/bash
#SBATCH --job-name=LLM
#SBATCH --time=03:50:00
#SBATCH --gres=gpu:1
#SBATCH --partition=dgxa100
#SBATCH --mem=80GB
#SBATCH --output=out_debug.out
#SBATCH --error=out_err.err
# source /path/to/your_venv/bin/activate
#
n_example_arr=(0 2 4 6)
for n_examples in "${n_example_arr[@]}"; do
  if [ $n_examples -eq 0 ]; then
    python zero_shot_LLM_classif.py --model_type $1
  else
    python few_shot_LLM_classif.py --model_type $1 --n_examples $n_examples
  fi
#echo $n_examples
#python test_lora.py
#python lora_finetune_classif.py "$@"

 
done
