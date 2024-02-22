python src/ets/train_classifier.py \
    --dataset data/ets/classification \
    --save_path trained_models/ets/sft/classifier \
    --hf_model_name roberta-large \
    --n_epochs 6 \
    --batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --disable_tqdm
