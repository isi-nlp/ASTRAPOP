cmd="python src/reddit/generate.py \
        --dataset data/reddit/test/random_100_authors.jsonl \
        --save_path outputs/reddit/sft/transferred.jsonl \
        --llama_model meta-llama/Llama-2-7b-hf \
        --paraphrase_model_dir trained_models/reddit/sft/paraphrase/best_checkpoint_merged \
        --sft_model_dir trained_models/reddit/sft/transfer/best_checkpoint_merged \
        --n_refs_per_author 5 \
        --temperature 0.7 \
        --top_p 1.0"
echo $cmd
eval $cmd

cmd="python src/reddit/generate.py \
        --dataset data/reddit/test/random_100_authors.jsonl \
        --save_path outputs/reddit/ppo/transferred.jsonl \
        --llama_model meta-llama/Llama-2-7b-hf \
        --paraphrase_model_dir trained_models/reddit/sft/paraphrase/best_checkpoint_merged \
        --sft_model_dir trained_models/reddit/sft/transfer/best_checkpoint_merged \
        --adapter_dir trained_models/reddit/ppo/transfer/checkpoint-last \
        --n_refs_per_author 5 \
        --temperature 0.7 \
        --top_p 1.0"
echo $cmd
eval $cmd

cmd="python src/reddit/generate.py \
        --dataset data/reddit/test/random_100_authors.jsonl \
        --save_path outputs/reddit/dpo/transferred.jsonl \
        --llama_model meta-llama/Llama-2-7b-hf \
        --paraphrase_model_dir trained_models/reddit/sft/paraphrase/best_checkpoint_merged \
        --sft_model_dir trained_models/reddit/sft/transfer/best_checkpoint_merged \
        --adapter_dir trained_models/reddit/dpo/transfer/checkpoint-best \
        --n_refs_per_author 5 \
        --temperature 0.7 \
        --top_p 1.0"
echo $cmd
eval $cmd

cmd="python src/reddit/generate.py \
        --dataset data/reddit/test/random_100_authors.jsonl \
        --save_path outputs/reddit/cpo/transferred.jsonl \
        --llama_model meta-llama/Llama-2-7b-hf \
        --paraphrase_model_dir trained_models/reddit/sft/paraphrase/best_checkpoint_merged \
        --sft_model_dir trained_models/reddit/sft/transfer/best_checkpoint_merged \
        --adapter_dir trained_models/reddit/cpo/transfer/checkpoint-best \
        --n_refs_per_author 5 \
        --temperature 0.7 \
        --top_p 1.0"
echo $cmd
eval $cmd
