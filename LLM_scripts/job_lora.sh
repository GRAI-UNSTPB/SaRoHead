#!/bin/bash
#SBATCH --job-name=satire_LoRA
#SBATCH --time=11:50:00
#SBATCH --gres=gpu:1
#SBATCH --partition=dgxh100
#SBATCH --mem=80GB
#SBATCH --output=out_debug_huh.out
#SBATCH --error=out_err_huh.err
# source /path/to/your_venv/bin/activate
#python test_lora.py
grid_search(){
  r_vals=(8)
  alpha_vals=(8)
  epochs=(5)
  final_lr=(1e-3)
  warmup_ratios=(0.2)
  for epoch in "${epochs[@]}";do
    for r in "${r_vals[@]}"; do
      for alpha in "${alpha_vals[@]}"; do
        for lr in "${final_lr[@]}"; do
           for warmup_ratio in "${warmup_ratios[@]}"; do
              python new_lora_finetune.py --model_type $1 --lora_r $r --lora_alpha $alpha --epochs $epoch \
                                          --lr $lr \
                                          --warmup_ratio $warmup_ratio
           done
        done
      done
    done
  done
}
r=8
alpha=8
epoch=5
#grid_search "$1"
llm_names=("llama2" "llama3" "gemma" "mistral")
for llm_name in "${llm_names[@]}"; do
  python new_lora_finetune.py --model_type "$llm_name" --lora_r $r --lora_alpha $alpha --epochs $epoch --lr 1e-3 --batch_size 32
  python test_lora.py --model_name "$llm_name" --category "all"
done
