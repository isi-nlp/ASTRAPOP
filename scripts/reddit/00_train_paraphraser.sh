python src/reddit/train_sft.py \
    --dataset data/reddit/train \
    --save_path trained_models/reddit/sft/paraphrase \
    --llama_model meta-llama/Llama-2-7b-hf \
    --mode paraphrase \
    --n_epochs 6 \
    --batch_size 16 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-5 \
    --disable_tqdm
