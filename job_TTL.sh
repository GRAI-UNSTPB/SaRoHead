#!/bin/bash
#SBATCH --job-name=TTL
#SBATCH --time=11:50:00
#SBATCH --gres=gpu:1
#SBATCH --partition=dgxa100
#SBATCH --mem=80GB
#SBATCH --output=out_TTL
#SBATCH --error=err_TTL.err
# source /path/to/your_venv/bin/activate
train_file=""
num_epochs=0
output_dir=""
if [ "$1" == "click" ]; then
    train_file="../../../../pentru_articol_satira/unsupervised_TTL/click/allRoClick.txt"
    output_dir="../../../../pentru_articol_satira/unsupervised_TTL/click/backup_click"
    num_epochs=50
elif [ "$1" == "saroco" ]; then
    train_file="../../../../pentru_articol_satira/unsupervised_TTL/saroco/titles_saroco.txt"
    output_dir="../../../../pentru_articol_satira/unsupervised_TTL/saroco/backup_saroco"
    num_epochs=40
elif [ "$1" == "bait" ]; then
    train_file="../../../../pentru_articol_satira/unsupervised_TTL/scitechbait/trainSciBait.txt"
    output_dir="../../../../pentru_articol_satira/unsupervised_TTL/scitechbait/backup_scitechbait"
    num_epochs=45
fi

all_names=("dumitrescustefan/bert-base-romanian-uncased-v1" "xlm-roberta-base" "racai/distilbert-base-romanian-uncased")
short_names=("BERT" "Distil" "XLM")
for short_name in "${short_names[@]}"; do
    if [[ "$short_name" == "BERT" ]]; then
      full_name="dumitrescustefan/bert-base-romanian-uncased-v1"
    elif [[ "$short_name" == "Distil" ]]; then
      full_name="racai/distilbert-base-romanian-uncased"
    elif [[ "$short_name" == "XLM" ]]; then
      full_name="xlm-roberta-base"
    fi
    name_dep_out_dir="${output_dir}_${short_name}"
    echo "$full_name"
    echo "$name_dep_out_dir"
    echo "---"
    python run_mlm.py --model_name_or_path "$full_name" \
                    --train_file "$train_file" \
                    --learning_rate 1e-5 \
                    --per_device_train_batch_size 32 \
                    --do_train \
                    --output_dir "$name_dep_out_dir" \
                    --num_train_epochs $num_epochs \
                    --seed 42 \
                    --line_by_line \
                    --save_total_limit 3
done
