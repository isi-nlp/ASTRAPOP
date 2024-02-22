import os
import json
import copy
import random
import argparse
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn.functional as F
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    GenerationConfig,
)

LANG2IDX = {
    'ARA': 0,
    'DEU': 1,
    'FRA': 2,
    'HIN': 3,
    'ITA': 4,
    'JPN': 5,
    'KOR': 6,
    'SPA': 7,
    'TEL': 8,
    'TUR': 9,
    'ZHO': 10
}

def get_paraphrase_prompt(src):
    prompt = f"[SRC]{src}[/SRC]"
    return prompt

def get_transfer_prompt(src):
    prompt = f"[SRC]{src}[/SRC]"
    return prompt

def generate(
    prompts,
    model,
    tokenizer,
    generation_config,
    adapter_name=None,
    max_new_tokens=128,
    encoded_refs=None,
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
    if adapter_name is not None:
        model.set_adapter(adapter_name)
    if encoded_refs is None:
        output = model.generate(
            input_ids=encoded_prompts['input_ids'],
            attention_mask=encoded_prompts['attention_mask'],
            generation_config=generation_config,
            max_new_tokens=max_new_tokens,
        )
    else:
        output = model.generate(
            input_ids=encoded_prompts['input_ids'],
            attention_mask=encoded_prompts['attention_mask'],
            refs_input_ids=encoded_refs['input_ids'],
            refs_attn_mask=encoded_refs['attention_mask'],
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
    parser.add_argument('--dataset', type=str, default="data/reddit/transfer")
    parser.add_argument('--splits', nargs='+')
    parser.add_argument('--save_path', type=str, default="test/test.json")
    parser.add_argument('--hf_cache_dir', type=str, default=os.environ.get("TRANSFORMERS_CACHE", None))
    parser.add_argument('--llama_model', type=str, default="meta-llama/Llama-2-7b-hf", help="llama model name or path")
    parser.add_argument('--sft_model_dir', type=str, default=None)
    parser.add_argument('--cls_model_dir', type=str, default="trained_models/ets/classifier/checkpoint-best")
    parser.add_argument('--tgt_lang', type=str, required=True)
    parser.add_argument('--toward_reward', action='store_true')
    parser.add_argument('--away_reward', action='store_true')
    parser.add_argument('--length_penalty', action='store_true')
    parser.add_argument('--disable_tqdm', action='store_true')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_input_len', type=int, default=512)
    parser.add_argument('--n_responses_per_query', type=int, default=2)
    parser.add_argument('--use_8bit', action='store_true')
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    transformers.set_seed(args.random_seed)

    # load model
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

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

    model = AutoModelForCausalLM.from_pretrained(
        args.sft_model_dir,
        torch_dtype=torch.bfloat16,
        load_in_8bit=args.use_8bit,
        device_map=device_map,
        cache_dir=args.hf_cache_dir,
        use_auth_token=os.environ.get('HUGGINGFACE_ACCESS_TOKEN'),
    )
    device = model.device

    # load classifier
    cls_tokenizer = AutoTokenizer.from_pretrained("roberta-large")
    cls_model = AutoModelForSequenceClassification.from_pretrained(args.cls_model_dir).to(device)

    # load data
    data = defaultdict(list)
    data_files = os.listdir(args.dataset)
    for data_file in data_files:
        split = data_file[:data_file.index('.')]
        with open(os.path.join(args.dataset, data_file)) as f:
            for line in f:
                line = json.loads(line)
                if line['native_lang'] != args.tgt_lang:
                    data[split].append(line)
        data[split] = random.sample(data[split], 2000 if split=="train" else 200)
    assert len(data["train"]) == 2000
    assert len(data["valid"]) == 200

    if args.debug:
        for split in data.keys():
            data[split] = data[split][:32]

    generation_config = GenerationConfig(
        temperature=1.0,
        do_sample=True,
        top_p=1.0,
        top_k=0,
        num_return_sequences=args.n_responses_per_query,
    )

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    for split in args.splits:
        outputs = defaultdict(list)
        for i in tqdm(range(0, len(data[split]), args.batch_size), disable=args.disable_tqdm):
            batch = data[split][i:i+args.batch_size]
            batch = {k: [item[k] for item in batch] for k in batch[0].keys()}

            # random tgt_native_langs
            batch['tgt_native_lang'] = [args.tgt_lang] * len(batch['native_lang'])
            assert all([s != t for s, t in zip(batch['native_lang'], batch['tgt_native_lang'])])
            
            assert batch['tgt_native_lang'][0] == args.tgt_lang
            assert len(set(batch['tgt_native_lang'])) == 1

            # generate 
            batch_src = batch['src']
            queries = [get_transfer_prompt(src) for src in batch_src]

            # add bos_token explicitly
            queries = [f"{tokenizer.bos_token}{q}" for q in queries]

            batch_response = generate(
                prompts=queries,
                model=model,
                tokenizer=tokenizer,
                generation_config=generation_config,
                max_new_tokens=128,
            )

            # compute cls scores
            response_cls_tensors = cls_tokenizer(
                batch_response,
                max_length=args.max_input_len, 
                truncation=True, 
                padding=True, 
                return_tensors='pt'
            )
            response_cls_tensors = {k: v.to(device) for k, v in response_cls_tensors.items()}
            response_cls_scores = cls_model(**response_cls_tensors).logits

            src_tgt_ids = torch.tensor(
                [[LANG2IDX[src_lang], LANG2IDX[tgt_lang]]
                for src_lang, tgt_lang in zip(batch['native_lang'], batch['tgt_native_lang'])]
            ).repeat_interleave(args.n_responses_per_query, 0).to(device)
            src_tgt_scores = torch.sigmoid(torch.gather(response_cls_scores, 1, src_tgt_ids))

            # compute reward
            away_reward = 1 - src_tgt_scores[:, 0]
            toward_reward = src_tgt_scores[:, 1]

            reward = torch.zeros_like(toward_reward)
            if args.toward_reward:
                reward += toward_reward
            if args.away_reward:
                reward += away_reward
            if args.length_penalty:
                src_lens = torch.tensor([len(tokenizer.tokenize(src)) for src in batch['src']], dtype=reward.dtype, device=reward.device)
                src_lens = src_lens.repeat_interleave(args.n_responses_per_query, dim=0)
                res_lens = torch.tensor([len(tokenizer.tokenize(res)) for res in batch_response], dtype=reward.dtype, device=reward.device)
                lp = torch.exp(1 - torch.min(src_lens, res_lens) / torch.max(src_lens, res_lens))
                reward = reward - (torch.pow(lp, 0.5) - 1)
            reward = reward.reshape(-1, args.n_responses_per_query)

            # write output
            chosen_index = reward.argmax(dim=1)
            rejected_index = reward.argmin(dim=1)
            grouped_response = [batch_response[i:i+args.n_responses_per_query] 
                                for i in range(0, len(batch_response), args.n_responses_per_query)]

            for i in range(len(queries)):
                outputs['prompt'].append(queries[i])
                outputs['chosen'].append(grouped_response[i][chosen_index[i]])
                outputs['rejected'].append(grouped_response[i][rejected_index[i]])

        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)

        with open(os.path.join(args.save_path, f"{split}.json"), 'w') as f:
            json.dump(outputs, f)