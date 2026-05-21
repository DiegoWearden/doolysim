#!/usr/bin/env python3
"""
Generate a dataset from openchat/openchat_sharegpt4_dataset.

Produces the same JSONL format as generate_sharegpt_dataset.py:
    {request_id, prompt, prompt_len, output_len, arrival_time}

Usage:
    python generate_openchat_sharegpt4_dataset.py \
        --output dataset/openchat_sharegpt4.jsonl \
        --num-requests 100 \
        --arrival-rate 5.0
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np


def generate_poisson_arrivals(num_requests: int, arrival_rate: float, seed: int = 42) -> List[float]:
    rng = np.random.default_rng(seed)
    inter = rng.exponential(scale=1.0 / arrival_rate, size=num_requests)
    return np.cumsum(inter).tolist()


def load_openchat_sharegpt4(
    num_requests: int,
    tokenizer_name: str = "meta-llama/Llama-3.1-8B",
    dataset_file: str = "sharegpt_gpt4.json",
    min_input_len: int | None = None,
    max_input_len: int | None = None,
    min_output_len: int | None = None,
    max_output_len: int | None = None,
    custom_output_len: int | None = None,
    seed: int = 42,
) -> List[Dict]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise SystemExit("ERROR: huggingface_hub required. pip install huggingface_hub")
    try:
        from transformers import AutoTokenizer
    except ImportError:
        raise SystemExit("ERROR: transformers required. pip install transformers")

    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    print(f"Downloading openchat/openchat_sharegpt4_dataset::{dataset_file}")
    path = hf_hub_download(
        repo_id="openchat/openchat_sharegpt4_dataset",
        filename=dataset_file,
        repo_type="dataset",
    )
    with open(path, "r") as f:
        raw = json.load(f)
    print(f"Loaded {len(raw)} raw conversations")

    random.Random(seed).shuffle(raw)

    prompts: List[Dict] = []
    skipped_no_turn = 0
    skipped_len = 0

    for entry in raw:
        turns = entry.get("items") or entry.get("conversations") or []
        if len(turns) < 2:
            skipped_no_turn += 1
            continue

        prompt_text = None
        output_text = None
        for turn in turns:
            role = turn.get("from")
            value = turn.get("value") or ""
            if role == "human" and prompt_text is None:
                prompt_text = value
            elif role == "gpt" and prompt_text is not None:
                output_text = value
                break

        if not prompt_text or not output_text:
            skipped_no_turn += 1
            continue

        prompt_len = len(tokenizer.encode(prompt_text))
        if custom_output_len is not None:
            output_len = custom_output_len
        else:
            output_len = len(tokenizer.encode(output_text, add_special_tokens=False))

        if output_len <= 0:
            skipped_len += 1
            continue
        if min_input_len is not None and prompt_len < min_input_len:
            skipped_len += 1
            continue
        if max_input_len is not None and prompt_len > max_input_len:
            skipped_len += 1
            continue
        if min_output_len is not None and output_len < min_output_len:
            skipped_len += 1
            continue
        if max_output_len is not None and output_len > max_output_len:
            skipped_len += 1
            continue

        prompts.append({
            "prompt": prompt_text,
            "prompt_len": prompt_len,
            "output_len": output_len,
        })
        if len(prompts) >= num_requests:
            break

    print(f"Prepared {len(prompts)} prompts "
          f"(skipped_no_turn={skipped_no_turn}, skipped_len={skipped_len})")
    return prompts


def create_dataset(
    output_path: str,
    num_requests: int,
    arrival_rate: float,
    tokenizer_name: str = "meta-llama/Llama-3.1-8B",
    dataset_file: str = "sharegpt_gpt4.json",
    min_input_len: int | None = None,
    max_input_len: int | None = None,
    min_output_len: int | None = None,
    max_output_len: int | None = None,
    custom_output_len: int | None = None,
    seed: int = 42,
):
    prompts = load_openchat_sharegpt4(
        num_requests=num_requests,
        tokenizer_name=tokenizer_name,
        dataset_file=dataset_file,
        min_input_len=min_input_len,
        max_input_len=max_input_len,
        min_output_len=min_output_len,
        max_output_len=max_output_len,
        custom_output_len=custom_output_len,
        seed=seed,
    )

    if len(prompts) < num_requests:
        print(f"WARNING: only found {len(prompts)} prompts (requested {num_requests})")
        num_requests = len(prompts)

    print(f"Generating Poisson arrivals with rate {arrival_rate} req/s")
    arrival_times = generate_poisson_arrivals(num_requests, arrival_rate, seed)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for i in range(num_requests):
            entry = {
                "request_id": f"req_{i}",
                "prompt": prompts[i]["prompt"],
                "prompt_len": prompts[i]["prompt_len"],
                "output_len": prompts[i]["output_len"],
                "arrival_time": arrival_times[i],
            }
            f.write(json.dumps(entry) + "\n")

    print(f"✓ Wrote {num_requests} requests -> {output_path}")
    if num_requests:
        print(f"  Time span: {arrival_times[-1]:.2f}s")

    prompt_lens = [p["prompt_len"] for p in prompts[:num_requests]]
    output_lens = [p["output_len"] for p in prompts[:num_requests]]
    if prompt_lens:
        print("\nPrompt length stats:")
        print(f"  mean={np.mean(prompt_lens):.1f}  median={np.median(prompt_lens):.0f}  "
              f"min={np.min(prompt_lens)}  max={np.max(prompt_lens)}")
        print("Output length stats:")
        print(f"  mean={np.mean(output_lens):.1f}  median={np.median(output_lens):.0f}  "
              f"min={np.min(output_lens)}  max={np.max(output_lens)}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate JSONL dataset from openchat/openchat_sharegpt4_dataset"
    )
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="Output JSONL path")
    parser.add_argument("--num-requests", type=int, default=100)
    parser.add_argument("--arrival-rate", type=float, default=5.0)
    parser.add_argument("--tokenizer", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--dataset-file", type=str, default="sharegpt_gpt4.json",
                        help="File to pull from the HF repo")
    parser.add_argument("--min-input-len", type=int)
    parser.add_argument("--max-input-len", type=int)
    parser.add_argument("--min-output-len", type=int)
    parser.add_argument("--max-output-len", type=int)
    parser.add_argument("--custom-output-len", type=int,
                        help="Force this output_len for every request")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    create_dataset(
        output_path=args.output,
        num_requests=args.num_requests,
        arrival_rate=args.arrival_rate,
        tokenizer_name=args.tokenizer,
        dataset_file=args.dataset_file,
        min_input_len=args.min_input_len,
        max_input_len=args.max_input_len,
        min_output_len=args.min_output_len,
        max_output_len=args.max_output_len,
        custom_output_len=args.custom_output_len,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
