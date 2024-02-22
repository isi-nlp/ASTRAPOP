export PYTHONPATH="src"

for tgt_lang in "ARA" "DEU" "FRA" "HIN" "ITA" "JPN" "KOR" "SPA" "TEL" "TUR" "ZHO"
do
    cmd="python src/ets/train_dpo_cpo.py \
        --po_algorithm dpo \
        --dataset data/ets/dpo_cpo_train/${tgt_lang} \
        --llama_model meta-llama/Llama-2-7b-hf \
        --sft_model_dir trained_models/ets/sft/transfer/${tgt_lang}/best_checkpoint_merged \
        --save_path trained_models/ets/dpo/transfer/${tgt_lang} \
        --n_epochs 10 \
        --batch_size 4 \
        --gradient_accumulation_steps 4 \
        --beta 0.5 \
        --lr 2e-6 \
        --disable_tqdm"

    echo $cmd 
    eval $cmd
done