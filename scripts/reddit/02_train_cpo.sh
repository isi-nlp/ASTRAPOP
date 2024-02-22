export PYTHONPATH="src"

python src/reddit/train_dpo_cpo.py \
    --po_algorithm cpo \
    --dataset data/reddit/dpo_cpo_train \
    --llama_model meta-llama/Llama-2-7b-hf \
    --sft_model_dir trained_models/reddit/sft/transfer/best_checkpoint_merged \
    --save_path trained_models/reddit/cpo/transfer \
    --n_epochs 6 \
    --batch_size 2 \
    --gradient_accumulation_steps 16 \
    --beta 0.5 \
    --lr 2e-6 \
    --disable_tqdm
