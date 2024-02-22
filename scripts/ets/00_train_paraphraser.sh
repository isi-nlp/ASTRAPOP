python src/ets/train_sft.py \
    --dataset data/ets/train \
    --save_path trained_models/ets/sft/paraphrase \
    --llama_model meta-llama/Llama-2-7b-hf \
    --mode paraphrase \
    --n_epochs 6 \
    --batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --disable_tqdm