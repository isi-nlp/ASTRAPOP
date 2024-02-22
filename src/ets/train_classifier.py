import os
import argparse
from collections import defaultdict

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
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

def encode_example(example, tokenizer):
    txt = example['txt']
    lang_idx = LANG2IDX[example['native_lang']]

    encoded_input = tokenizer(
        txt,
        max_length=512, 
        truncation=True,
    )
    labels = torch.zeros((len(LANG2IDX), ), dtype=float)
    labels[lang_idx] = 1
    
    return {
        "input_ids": encoded_input['input_ids'],
        "attention_mask": encoded_input["attention_mask"],
        "labels": labels,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="data/ets/classification")
    parser.add_argument('--save_path', type=str, default="trained_models/test")
    parser.add_argument('--hf_cache_dir', type=str, default=os.environ.get("TRANSFORMERS_CACHE", None))
    parser.add_argument('--hf_model_name', type=str, default="roberta-large")
    parser.add_argument('--n_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--use_gradient_checkpointing', action='store_true')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--disable_tqdm', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1

    # load model / tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.hf_model_name,
        problem_type='multi_label_classification',
        num_labels=len(LANG2IDX),
    )

    # load data
    data_files = {
        "train": "train.jsonl",
        "valid": "valid.jsonl",
        "test": "test.jsonl",
    }

    dataset = load_dataset(
        args.dataset, 
        data_files=data_files, 
    )

    fn_kwargs = {"tokenizer": tokenizer}
    dataset = dataset.map(
        encode_example,
        fn_kwargs=fn_kwargs, 
        num_proc=4,
        load_from_cache_file=False,
    )

    data_columns = [
        'input_ids', 
        'attention_mask',
        'labels', 
    ]
    dataset.set_format(
        columns=data_columns,
    )

    per_device_batch_size = int(args.batch_size / world_size)
    num_update_steps_per_epoch = int(len(dataset['train']) / (args.batch_size * args.gradient_accumulation_steps))
    save_steps = int(num_update_steps_per_epoch / 3)
    warmup_steps = num_update_steps_per_epoch
    print(f"\nnum_update_steps_per_epoch: {num_update_steps_per_epoch}\nper_device_batch_size: {per_device_batch_size}\nsave_steps: {save_steps}\nwarmup_steps: {warmup_steps}\n")
    trainer = Trainer(
        model=model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['valid'],
        args=TrainingArguments(
            max_steps=0 if not args.debug else 3,
            per_device_train_batch_size=per_device_batch_size,
            per_device_eval_batch_size=per_device_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            num_train_epochs=args.n_epochs, 
            learning_rate=args.learning_rate,
            fp16=False,
            logging_steps=1,
            optim="adamw_torch",
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=save_steps,
            save_steps=save_steps,
            output_dir=args.save_path,
            save_total_limit=3,
            load_best_model_at_end=True,
            ddp_find_unused_parameters=True if ddp and not args.use_gradient_checkpointing else False,
            group_by_length=False,
            disable_tqdm=args.disable_tqdm,
            dataloader_num_workers=8,
            remove_unused_columns=False,
        ),
        data_collator=DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding=True,
        ),
    )

    trainer.train()
    trainer.save_model(os.path.join(args.save_path, "checkpoint-best"))