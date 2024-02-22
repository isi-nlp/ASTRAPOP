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
    AutoModel,
    AutoModelForCausalLM,
    GenerationConfig,
)


def get_transfer_prompt(src, refs):
    prompt = ''.join([f"[REF]{ref}[/REF]" for ref in refs]) + f"[SRC]{src}[/SRC]"
    return prompt

def get_paraphrase_prompt(src):
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
    parser.add_argument('--luar_model', type=str, default="rrivera1849/LUAR-MUD", help="luar model name or path")
    parser.add_argument('--sft_model_dir', type=str, default=None)
    parser.add_argument('--toward_reward', action='store_true')
    parser.add_argument('--away_reward', action='store_true')
    parser.add_argument('--length_penalty', action='store_true')
    parser.add_argument('--disable_tqdm', action='store_true')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_input_len', type=int, default=512)
    parser.add_argument('--n_responses_per_query', type=int, default=2)
    parser.add_argument('--use_8bit', action='store_true')
    parser.add_argument('--n_refs_per_author', type=int, default=5)
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
    
    # load luar
    rep_tokenizer = AutoTokenizer.from_pretrained(args.luar_model)
    rep_model = AutoModel.from_pretrained(args.luar_model, trust_remote_code=True).to(model.device)

    generation_config = GenerationConfig(
        temperature=1.0,
        do_sample=True,
        top_p=1.0,
        top_k=0,
        num_return_sequences=args.n_responses_per_query,
    )

    device = model.device

    # load data
    data = defaultdict(list)
    data_files = os.listdir(args.dataset)
    for data_file in data_files:
        split = data_file[:data_file.index('.')]
        with open(os.path.join(args.dataset, data_file)) as f:
            for line in f:
                data[split].append(json.loads(line))

    if args.debug:
        for split in data.keys():
            data[split] = data[split][:32]

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    for split in args.splits:
        outputs = defaultdict(list)
        for i in tqdm(range(0, len(data[split]), args.batch_size), disable=args.disable_tqdm):
            batch = data[split][i:i+args.batch_size]
            batch = {k: [item[k] for item in batch] for k in batch[0].keys()}

            bsz = len(batch['src'])

            # rotate refs
            batch['refs'] = batch['refs'][1:] + batch['refs'][:1]

            # compute src reps
            src_rep_tensors = rep_tokenizer(
                batch['tgt'],
                max_length=args.max_input_len, 
                truncation=True, 
                padding=True, 
                return_tensors='pt'
            )
            max_src_len = src_rep_tensors['input_ids'].size(1)
            src_rep_tensors = {k: v.unsqueeze(1).unsqueeze(1).reshape((bsz, 1, max_src_len)).to(device) for k, v in src_rep_tensors.items()}
            src_rep = rep_model(src_rep_tensors['input_ids'], src_rep_tensors['attention_mask']).detach()

            # compute tgt reps
            tgt_rep_tensors = rep_tokenizer(
                [ref for refs in batch['refs'] for ref in refs],
                max_length=args.max_input_len, 
                truncation=True, 
                padding=True, 
                return_tensors='pt'
            )
            n_refs_per_author = len(batch['refs'][0])
            max_tgt_len = tgt_rep_tensors['input_ids'].size(1)
            tgt_rep_tensors = {k: v.unsqueeze(1).unsqueeze(1).reshape((bsz, n_refs_per_author, max_tgt_len)).to(device) for k, v in tgt_rep_tensors.items()}
            tgt_rep = rep_model(tgt_rep_tensors['input_ids'], tgt_rep_tensors['attention_mask']).detach()

            # generate
            batch_src = batch['src']
            queries = [get_transfer_prompt(src, [ref[1] for ref in refs]) for src, refs in zip(batch_src, batch['refs'])]

            # add bos_token explicitly
            queries = [f"{tokenizer.bos_token}{q}" for q in queries]

            batch_response = generate(
                prompts=queries,
                model=model,
                tokenizer=tokenizer,
                generation_config=generation_config,
                max_new_tokens=128,
            )

            # compute response reps
            response_rep_tensors = rep_tokenizer(
                batch_response,
                max_length=args.max_input_len, 
                truncation=True, 
                padding=True, 
                return_tensors='pt'
            )
            max_response_len = response_rep_tensors['input_ids'].size(1)
            response_rep_tensors = {k: v.unsqueeze(1).unsqueeze(1).reshape((bsz * args.n_responses_per_query, 1, max_response_len)).to(device) for k, v in response_rep_tensors.items()}
            response_rep = rep_model(response_rep_tensors['input_ids'], response_rep_tensors['attention_mask']).detach()
            
            # compute reward
            toward_reward = F.cosine_similarity(tgt_rep.repeat_interleave(args.n_responses_per_query, dim=0), response_rep)
            away_reward = 1 - F.cosine_similarity(src_rep.repeat_interleave(args.n_responses_per_query, dim=0), response_rep)

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

            with open(os.path.join(args.save_path, f"{split}.json"), 'w') as f:
                json.dump(outputs, f)