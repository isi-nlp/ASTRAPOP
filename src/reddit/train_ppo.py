import os
import argparse
from tqdm import tqdm
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    LlamaTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
)
from datasets import load_dataset
from trl import (
    PPOConfig, 
    PPOTrainer,
    AutoModelForCausalLMWithValueHead,
)


@dataclass
class DataCollator:

    def __call__(self, features):
        features = {k: [feat[k] for feat in features] for k in features[0].keys()}

        return features

def get_concat_transfer_prompt(src, refs):
    prompt = ''.join([f"[REF]{ref}[/REF]" for ref in refs]) + f"[SRC]{src}[/SRC]"
    return prompt

def get_paraphrase_prompt(src):
    prompt = f"[SRC]{src}[/SRC]"
    return prompt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="data/reddit/transfer")
    parser.add_argument('--save_path', type=str, default="test")
    parser.add_argument('--hf_cache_dir', type=str, default=os.environ.get("TRANSFORMERS_CACHE", None))
    parser.add_argument('--llama_model', type=str, default="meta-llama/Llama-2-7b-hf", help="llama model name or path")
    parser.add_argument('--sft_model_dir', type=str,required=True) 
    parser.add_argument('--luar_model', type=str, default="rrivera1849/LUAR-MUD", help="luar model name or path")
    parser.add_argument('--toward_reward', action='store_true')
    parser.add_argument('--away_reward', action='store_true')
    parser.add_argument('--length_penalty', action='store_true')
    parser.add_argument('--n_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--ppo_gen_batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2.82e-5)
    parser.add_argument('--max_input_len', type=int, default=512)
    parser.add_argument('--use_8bit', action='store_true')
    parser.add_argument('--disable_tqdm', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    assert args.toward_reward or args.away_reward

    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}

    # load llama models
    tokenizer = AutoTokenizer.from_pretrained(
        args.llama_model,
        cache_dir=args.hf_cache_dir,
        use_auth_token=os.environ.get('HUGGINGFACE_ACCESS_TOKEN'),
    )
    tokenizer.pad_token_id = (0)
    tokenizer.padding_side = "right"

    quantization_config = None
    if args.use_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_skip_modules=["lm_head"],
        )

    # load base sft model
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=['q_proj', 'v_proj'],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.sft_model_dir,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        peft_config=lora_config,
        load_in_8bit=args.use_8bit,
        quantization_config=quantization_config,
    )

    # load luar
    rep_tokenizer = AutoTokenizer.from_pretrained(args.luar_model)
    rep_model = AutoModel.from_pretrained(args.luar_model, trust_remote_code=True).to(model.pretrained_model.device)

    # load dataset
    data_files = {
        "train": "train.jsonl",
        "valid": "valid.jsonl",
        "test": "test.jsonl",
    }
    dataset = load_dataset(
        args.dataset, 
        data_files=data_files, 
        cache_dir=None,
    )

    # ppo training
    config = PPOConfig(
        model_name="llama",
        batch_size=int(args.batch_size / world_size),
        learning_rate=args.lr,
        remove_unused_columns=False,
        use_score_scaling=True,
        use_score_norm=True,
        score_clip=0.5,
    )

    ppo_trainer = PPOTrainer(
        model=model,
        config=config,
        dataset=dataset['train'],
        data_collator=DataCollator(),
        tokenizer=tokenizer,
    )

    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "max_new_tokens": 128,
        "pad_token_id": tokenizer.eos_token_id,
    }

    pbar = tqdm(total=args.n_epochs * len(ppo_trainer.dataloader), disable=args.disable_tqdm)
    iter_i, ave_toward, ave_away = 0, torch.tensor(0), torch.tensor(0)
    for epoch_i in range(args.n_epochs): 
        for batch_i, batch in enumerate(ppo_trainer.dataloader):
            device = model.pretrained_model.device
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
            queries = [get_concat_transfer_prompt(src, [ref[1] for ref in refs]) for src, refs in zip(batch_src, batch['refs'])]

            query_tensors = tokenizer(
                [f"{tokenizer.bos_token}{q}" for q in queries],  
                add_special_tokens=False, 
                max_length=args.max_input_len, 
                truncation=True,
            )
            query_tensors = [torch.tensor(s).to(device) for s in query_tensors['input_ids']]
            response_tensors = ppo_trainer.generate(query_tensors, batch_size=args.ppo_gen_batch_size, return_prompt=False, **generation_kwargs)
            batch_response = [tokenizer.decode(r[r != tokenizer.eos_token_id].squeeze()) for r in response_tensors]
            
            # compute response reps
            response_rep_tensors = rep_tokenizer(
                batch_response,
                max_length=args.max_input_len, 
                truncation=True, 
                padding=True, 
                return_tensors='pt'
            )
            max_response_len = response_rep_tensors['input_ids'].size(1)
            response_rep_tensors = {k: v.unsqueeze(1).unsqueeze(1).reshape((bsz, 1, max_response_len)).to(device) for k, v in response_rep_tensors.items()}
            response_rep = rep_model(response_rep_tensors['input_ids'], response_rep_tensors['attention_mask']).detach()
            
            # compute reward
            toward_reward = F.cosine_similarity(tgt_rep, response_rep)
            away_reward = 1 - F.cosine_similarity(src_rep, response_rep)
            reward = torch.zeros_like(toward_reward)
            if args.toward_reward:
                reward += toward_reward
            if args.away_reward:
                reward += away_reward
            if args.length_penalty:
                src_lens = torch.tensor([len(tokenizer.tokenize(src)) for src in batch['src']], dtype=reward.dtype, device=reward.device)
                res_lens = torch.tensor([len(tokenizer.tokenize(res)) for res in batch_response], dtype=reward.dtype, device=reward.device)
                lp = torch.exp(1 - torch.min(src_lens, res_lens) / torch.max(src_lens, res_lens))
                reward = reward - (torch.pow(lp, 0.5) - 1)
            reward = list(reward)

            stats = ppo_trainer.step(query_tensors, response_tensors, reward)
            ppo_trainer.log_stats(stats, batch, reward)

            pbar.update(1)
            batch_toward = toward_reward.mean().detach().cpu()
            ave_toward = (ave_toward * iter_i + batch_toward) / (iter_i + 1)
            batch_away = away_reward.mean().detach().cpu()
            ave_away = (ave_away * iter_i + batch_away) / (iter_i + 1)
            iter_i += 1
            pbar.set_postfix(ave_toward=f"{ave_toward:.4f}", ave_away=f"{ave_away:.4f}")

            if args.debug and batch_i >= 1:
                break

        if local_rank == 0:
            checkpoint_path = os.path.join(args.save_path, f"checkpoint-{iter_i}")
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)
            model.save_pretrained(checkpoint_path)

        if args.debug:
            break

    if local_rank == 0:
        checkpoint_path = os.path.join(args.save_path, f"checkpoint-last")
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        model.save_pretrained(checkpoint_path)
