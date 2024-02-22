for tgt_lang in "ARA" "DEU" "FRA" "HIN" "ITA" "JPN" "KOR" "SPA" "TEL" "TUR" "ZHO"
do
    cmd="python src/ets/train_ppo.py \
        --dataset data/ets/train \
        --llama_model meta-llama/Llama-2-7b-hf \
        --sft_model_dir trained_models/ets/sft/transfer/${tgt_lang}/best_checkpoint_merged \
        --cls_path trained_models/ets/sft/classifier/checkpoint-best \
        --save_path trained_models/ets/ppo/transfer/${tgt_lang} \
        --tgt_lang $tgt_lang \
        --toward_reward \
        --length_penalty \
        --n_epochs 6 \
        --batch_size 32 \
        --lr 1.41e-5"

    echo $cmd 
    eval $cmd
done