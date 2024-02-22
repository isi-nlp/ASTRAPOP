import os
import json
import copy
import random
import argparse
from tqdm import tqdm

import torch
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
)


def get_paraphrase_prompt(src):
    prompt = f"[SRC]{src}[/SRC]"
    return prompt

def get_transfer_prompt(src, refs):
    prompt = ''.join([f"[REF]{ref}[/REF]" for ref in refs]) + f"[SRC]{src}[/SRC]"
    return prompt

def generate(
    prompts,
    model,
    tokenizer,
    generation_config,
    max_new_tokens=128,
):
    encoded_prompts = tokenizer(
        prompts,
        add_special_tokens=False,
        max_length=1000,
        padding=True, 
        truncation=True,
        return_tensors='pt',
    )
    encoded_prompts = {k: v.cuda() for k, v in encoded_prompts.items()}
    input_len = encoded_prompts['input_ids'].size(1)
    output = model.generate(
        input_ids=encoded_prompts['input_ids'],
        attention_mask=encoded_prompts['attention_mask'],
        generation_config=generation_config,
        max_new_tokens=max_new_tokens,
    )
    output = tokenizer.batch_decode(output[:, input_len:], skip_special_tokens=True)

    return output

def remove_trailing_char(text, trailing_char='}'):
    if text.endswith(trailing_char):
        return text[:-1]
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="data/reddit/test/random_100_authors.jsonl")
    parser.add_argument('--save_path', type=str, default="test/test.jsonl")
    parser.add_argument('--hf_cache_dir', type=str, default=os.environ.get("TRANSFORMERS_CACHE", None))
    parser.add_argument('--llama_model', type=str, default="meta-llama/Llama-2-7b-hf", help="llama model name or path")
    parser.add_argument('--paraphrase_model_dir', type=str, required=True)
    parser.add_argument('--sft_model_dir', type=str, required=True)
    parser.add_argument('--adapter_dir', type=str, default=None)
    parser.add_argument('--disable_tqdm', action='store_true')
    parser.add_argument('--use_8bit', action='store_true')
    parser.add_argument('--n_refs_per_author', type=int, default=5)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    transformers.set_seed(args.random_seed)

    # load model
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}

    tokenizer = AutoTokenizer.from_pretrained(
        args.llama_model,
        cache_dir=args.hf_cache_dir,
        use_auth_token=os.environ.get('HUGGINGFACE_ACCESS_TOKEN'),
    )
    tokenizer.pad_token_id = (0)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    paraphrase_model = AutoModelForCausalLM.from_pretrained(
        args.paraphrase_model_dir,
        torch_dtype=torch.bfloat16,
        load_in_8bit=args.use_8bit,
        device_map=device_map,
        cache_dir=args.hf_cache_dir,
        use_auth_token=os.environ.get('HUGGINGFACE_ACCESS_TOKEN'),
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.sft_model_dir,
        torch_dtype=torch.bfloat16,
        load_in_8bit=args.use_8bit,
        device_map=device_map,
        cache_dir=args.hf_cache_dir,
        use_auth_token=os.environ.get('HUGGINGFACE_ACCESS_TOKEN'),
    )
    if args.adapter_dir is not None:
        model.load_adapter(args.adapter_dir)

    # load data
    data = []
    with open(args.dataset) as f:
        for line in f:
            data.append(json.loads(line))
    if args.debug:
        data = data[:3]
    output_data = copy.deepcopy(data)

    generation_config = GenerationConfig(
        temperature=args.temperature,
        do_sample=True,
        top_p=args.top_p,
        top_k=0,
    )

    save_dir = os.path.dirname(args.save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(args.save_path, 'w') as f:
        for i, ex in enumerate(tqdm(data, disable=args.disable_tqdm)):
            src_texts = ex['src_texts']
            tgt_texts = ex['tgt_texts']

            refs = random.sample(tgt_texts, args.n_refs_per_author)

            # paraphrase src_texts
            prompts = [f"{tokenizer.bos_token}{get_paraphrase_prompt(text)}" for text in src_texts]
            paraphrased_src_texts = generate(
                prompts=prompts,
                model=paraphrase_model,
                tokenizer=tokenizer,
                generation_config=generation_config,
                max_new_tokens=128,
            )

            # transfer
            prompts = [f"{tokenizer.bos_token}{get_transfer_prompt(text, refs)}" for text in paraphrased_src_texts]

            output_texts = generate(
                prompts=prompts,
                model=model,
                tokenizer=tokenizer,
                generation_config=generation_config,
                max_new_tokens=128,
            )

            output_data[i]['transferred'] = output_texts
            f.write(f"{json.dumps(output_data[i])}\n")

