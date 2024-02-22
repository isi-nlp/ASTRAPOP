import os
import json
import argparse
from tqdm import tqdm
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaTokenizer,
    TrainingArguments,
)
from trl import DPOTrainer
from utils.cpo_trainer import CPOTrainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--save_path', type=str, default="test")
    parser.add_argument('--hf_cache_dir', type=str, default=os.environ.get("TRANSFORMERS_CACHE", None))
    parser.add_argument('--llama_model', type=str, default="meta-llama/Llama-2-7b-hf", help="llama model name or path")
    parser.add_argument('--sft_model_dir', type=str, required=True) 
    parser.add_argument('--po_algorithm', type=str, choices=["dpo", "cpo"], required=True)
    parser.add_argument('--n_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--lr', type=float, default=5e-7)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--disable_tqdm', action='store_true')
    parser.add_argument('--debug', action='store_true')
    
    args = parser.parse_args()

    # load dataset
    dataset = {}
    data_files = os.listdir(args.dataset)
    for data_file in data_files:
        split = data_file[:data_file.index('.')]
        with open(os.path.join(args.dataset, data_file)) as f:
            split_data = json.load(f)
            dataset[split] = [{key: split_data[key][i] for key in split_data} for i in range(len(split_data['prompt']))]

    # load base sft model and tokenizer
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
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.sft_model_dir,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
    ) 

    per_device_batch_size = int(args.batch_size / world_size)
    num_update_steps_per_epoch = int(len(dataset['train']) / (args.batch_size * args.gradient_accumulation_steps))
    save_steps = int(num_update_steps_per_epoch / 3)
    # warmup_steps = num_update_steps_per_epoch
    warmup_steps = 150
    print(f"\nnum_update_steps_per_epoch: {num_update_steps_per_epoch}\nper_device_batch_size: {per_device_batch_size}\nsave_steps: {save_steps}\n")
    training_args = TrainingArguments(
        max_steps=0 if not args.debug else 3,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        remove_unused_columns=False,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.n_epochs, 
        learning_rate=args.lr,
        evaluation_strategy="steps",
        logging_steps=1,
        eval_steps=save_steps,
        save_steps=save_steps,
        output_dir=args.save_path,
        optim="rmsprop",
        warmup_steps=warmup_steps,
        bf16=True,
        save_total_limit=3,
        load_best_model_at_end=True,
        dataloader_num_workers=8,
        disable_tqdm=args.disable_tqdm,
        report_to="none",
    )

    if args.po_algorithm == "dpo":
        Trainer = DPOTrainer
    elif args.po_algorithm == "cpo":
        Trainer = CPOTrainer

    trainer = Trainer(
        model,
        peft_config=LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=['q_proj', 'v_proj'],
            bias="none",
            task_type="CAUSAL_LM",
        ),
        args=training_args,
        beta=args.beta,
        train_dataset=dataset['train'],
        eval_dataset=dataset['valid'],
        tokenizer=tokenizer,
        max_length=512,
        max_target_length=512,
        max_prompt_length=512,
    )

    trainer.train()
    trainer.save_model(os.path.join(args.save_path, "checkpoint-best"))
