python src/reddit/train_sft.py \
    --dataset data/reddit/train \
    --save_path trained_models/reddit/sft/transfer \
    --llama_model meta-llama/Llama-2-7b-hf \
    --mode transfer \
    --n_epochs 6 \
    --batch_size 2 \
    --gradient_accumulation_steps 16 \
    --learning_rate 5e-5 \
    --disable_tqdm
