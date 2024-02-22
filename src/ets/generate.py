import os
import json
import copy
import nltk
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

def get_transfer_prompt(src, native_lang):
    prompt = f"[SRC]{src}[/SRC]"
    return prompt

def generate(
    prompts,
    model,
    tokenizer,
    generation_config,
    adapter_name=None,
    max_new_tokens=128,
):
    encoded_prompts = tokenizer(
        prompts,
        add_special_tokens=False,
        max_length=384,
        padding=True, 
        truncation=True,
        return_tensors='pt',
    )
    encoded_prompts = {k: v.cuda() for k, v in encoded_prompts.items()}
    input_len = encoded_prompts['input_ids'].size(1)
    if adapter_name is not None:
        model.set_adapter(adapter_name)
    output = model.generate(
        input_ids=encoded_prompts['input_ids'],
        attention_mask=encoded_prompts['attention_mask'],
        generation_config=generation_config,
        max_new_tokens=max_new_tokens,
    )
    output = tokenizer.batch_decode(output[:, input_len:], skip_special_tokens=True)

    return output

class DocSegmenter:

    def __init__(
        self,
        tokenizer,
    ):
        self.tokenizer = tokenizer

    def _segment_doc(self, doc, window_len, overlap_len):
        doc = [sent.strip() for para in doc.splitlines()
               for sent in nltk.sent_tokenize(para.strip()) if len(sent) > 0]

        sent_lens = [len(self.tokenizer(sent).input_ids) for sent in doc]
        segs = []
        curr_len, curr_seg = 0, []
        for i in range(len(doc)):
            sent = doc[i]
            sent_len = sent_lens[i]
            if curr_len + sent_len > window_len and curr_seg:
                segs.append(' '.join(curr_seg))
                next_len, next_seg = 0, []
                for j in reversed(range(len(curr_seg))):
                    temp_len = sent_lens[i - (len(curr_seg) - j)]
                    if temp_len + next_len > overlap_len:
                        curr_len, curr_seg = next_len, next_seg
                        break
                    next_len += temp_len
                    next_seg.insert(0, curr_seg[j])
            curr_len += sent_len
            curr_seg.append(sent)
        segs.append(' '.join(curr_seg))

        return segs

    def segment_docs(self, docs, window_len=32, overlap_len=0, flatten=False):
        segs = [self._segment_doc(doc, window_len=window_len, overlap_len=overlap_len) for doc in docs]
        lens = [len(doc_segs) for doc_segs in segs]

        if flatten:
            segs = [seg for doc_segs in segs for seg in doc_segs]

        return lens, segs

    @staticmethod
    def regroup_segments(segments, lens):
        regrouped_docs = []
        start = 0

        for n_segs in lens:
            doc = ' '.join(segments[start:start + n_segs])
            regrouped_docs.append(doc)
            start += n_segs

        return regrouped_docs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="data/reddit/test/random.jsonl")
    parser.add_argument('--save_path', type=str, default="test/test.jsonl")
    parser.add_argument('--hf_cache_dir', type=str, default=os.environ.get("TRANSFORMERS_CACHE", None))
    parser.add_argument('--llama_model', type=str, default="meta-llama/Llama-2-7b-hf", help="llama model name or path")
    parser.add_argument('--paraphrase_model_dir', type=str, required=True)
    parser.add_argument('--sft_model_dir', type=str, required=True)
    parser.add_argument('--adapter_dir', type=str, default=None)
    parser.add_argument('--disable_tqdm', action='store_true')
    parser.add_argument('--tgt_lang', type=str, default=None, choices=sorted(LANG2IDX.keys()))
    parser.add_argument('--use_8bit', action='store_true')
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=16)
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
            line = json.loads(line)
            data.append({
                'txt': line['fullText'],
                'native_lang': line['source_specific']['native_lang'],
            })
    if args.debug:
        data = data[:3]
    output_data = copy.deepcopy(data)

    # segment data
    segmenter = DocSegmenter(tokenizer)
    seg_lens, segs = segmenter.segment_docs([item['txt'] for item in data], window_len=128, overlap_len=0, flatten=True)

    generation_config = GenerationConfig(
        temperature=args.temperature,
        do_sample=True,
        top_p=args.top_p,
        top_k=0,
    )

    save_dir = os.path.dirname(args.save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    transferred_segs = []
    for i in tqdm(range(0, len(segs), args.batch_size)):
        batch_segs = segs[i:i+args.batch_size]
        
        # paraphrase src_texts
        prompts = [f"{tokenizer.bos_token}{get_paraphrase_prompt(text)}" for text in batch_segs]
        paraphrased_src_texts = generate(
            prompts=prompts,
            model=paraphrase_model,
            tokenizer=tokenizer,
            generation_config=generation_config,
            max_new_tokens=256,
        )

        # transfer
        prompts = [f"{tokenizer.bos_token}{get_transfer_prompt(text, args.tgt_lang)}" for text in paraphrased_src_texts]

        output_texts = generate(
            prompts=prompts,
            model=model,
            tokenizer=tokenizer,
            generation_config=generation_config,
            max_new_tokens=256,
        )

        transferred_segs += output_texts

    transferred_docs = segmenter.regroup_segments(transferred_segs, seg_lens)
    with open(args.save_path, 'w') as f:
        for i, transferred_doc in enumerate(transferred_docs):
            output_data[i]['transferred'] = transferred_doc
            output_data[i]['tgt_native_lang'] = args.tgt_lang
            f.write(f"{json.dumps(output_data[i])}\n")
