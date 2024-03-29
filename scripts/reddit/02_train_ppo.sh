python src/reddit/train_ppo.py \
    --dataset data/reddit/train \
    --llama_model meta-llama/Llama-2-7b-hf \
    --sft_model_dir trained_models/reddit/sft/transfer/best_checkpoint_merged \
    --luar_model rrivera1849/LUAR-MUD \
    --save_path trained_models/reddit/ppo/transfer \
    --toward_reward \
    --away_reward \
    --length_penalty \
    --n_epochs 6 \
    --batch_size 32 \
    --lr 1.41e-5
