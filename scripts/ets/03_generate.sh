for tgt_lang in "ARA" "DEU" "FRA" "HIN" "ITA" "JPN" "KOR" "SPA" "TEL" "TUR" "ZHO"
do
    cmd="python src/ets/generate.py \
        --dataset data/ets/test/test.jsonl \
        --save_path outputs/ets/sft/transferred/${tgt_lang}.jsonl \
        --llama_model meta-llama/Llama-2-7b-hf \
        --sft_model_dir trained_models/ets/sft/transfer/${tgt_lang}/best_checkpoint_merged \
        --paraphrase_model_dir trained_models/ets/sft/paraphrase/best_checkpoint_merged \
        --tgt_lang $tgt_lang \
        --batch_size 32 \
        --temperature 0.7 \
        --top_p 1.0"

    echo $cmd 
    eval $cmd
done

for tgt_lang in "ARA" "DEU" "FRA" "HIN" "ITA" "JPN" "KOR" "SPA" "TEL" "TUR" "ZHO"
do
    cmd="python src/ets/generate.py \
        --dataset data/ets/test/test.jsonl \
        --save_path outputs/ets/ppo/transferred/${tgt_lang}.jsonl \
        --llama_model meta-llama/Llama-2-7b-hf \
        --sft_model_dir trained_models/ets/sft/transfer/${tgt_lang}/best_checkpoint_merged \
        --paraphrase_model_dir trained_models/ets/sft/paraphrase/best_checkpoint_merged \
        --adapter_dir trained_models/ets/ppo/transfer/${tgt_lang}/checkpoint-last \
        --tgt_lang $tgt_lang \
        --batch_size 32 \
        --temperature 0.7 \
        --top_p 1.0"

    echo $cmd 
    eval $cmd
done

for tgt_lang in "ARA" "DEU" "FRA" "HIN" "ITA" "JPN" "KOR" "SPA" "TEL" "TUR" "ZHO"
do
    cmd="python src/ets/generate.py \
        --dataset data/ets/test/test.jsonl \
        --save_path outputs/ets/dpo/transferred/${tgt_lang}.jsonl \
        --llama_model meta-llama/Llama-2-7b-hf \
        --sft_model_dir trained_models/ets/sft/transfer/${tgt_lang}/best_checkpoint_merged \
        --paraphrase_model_dir trained_models/ets/sft/paraphrase/best_checkpoint_merged \
        --adapter_dir trained_models/ets/dpo/transfer/${tgt_lang}/checkpoint-best \
        --tgt_lang $tgt_lang \
        --batch_size 32 \
        --temperature 0.7 \
        --top_p 1.0"

    echo $cmd 
    eval $cmd
done

for tgt_lang in "ARA" "DEU" "FRA" "HIN" "ITA" "JPN" "KOR" "SPA" "TEL" "TUR" "ZHO"
do
    cmd="python src/ets/generate.py \
        --dataset data/ets/test/test.jsonl \
        --save_path outputs/ets/cpo/transferred/${tgt_lang}.jsonl \
        --llama_model meta-llama/Llama-2-7b-hf \
        --sft_model_dir trained_models/ets/sft/transfer/${tgt_lang}/best_checkpoint_merged \
        --paraphrase_model_dir trained_models/ets/sft/paraphrase/best_checkpoint_merged \
        --adapter_dir trained_models/ets/cpo/transfer/${tgt_lang}/checkpoint-best \
        --tgt_lang $tgt_lang \
        --batch_size 32 \
        --temperature 0.7 \
        --top_p 1.0"

    echo $cmd 
    eval $cmd
done