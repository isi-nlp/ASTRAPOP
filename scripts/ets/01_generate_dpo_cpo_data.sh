for tgt_lang in "ARA" "DEU" "FRA" "HIN" "ITA" "JPN" "KOR" "SPA" "TEL" "TUR" "ZHO"
do
    python src/ets/generate_dpo_cpo_data.py \
        --dataset data/ets/train \
        --splits train valid \
        --save_path data/ets/dpo_cpo_train/${tgt_lang} \
        --llama_model meta-llama/Llama-2-7b-hf \
        --sft_model_dir trained_models/ets/sft/transfer/${tgt_lang}/best_checkpoint_merged \
        --cls_model_dir trained_models/ets/sft/classifier/checkpoint-best \
        --tgt_lang ${tgt_lang} \
        --toward_reward \
        --length_penalty \
        --batch_size 4 \
        --n_responses_per_query 3
done
