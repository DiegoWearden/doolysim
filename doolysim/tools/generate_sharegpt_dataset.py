#!/usr/bin/env python3
"""
Generate a dataset from ShareGPT using vLLM's data loading utilities.

This creates a JSONL dataset with the same format as our synthetic datasets,
but using real conversational data from ShareGPT.

Usage:
    python generate_sharegpt_dataset.py \
        --sharegpt-path /path/to/ShareGPT_V3_unfiltered_cleaned_split.json \
        --output dataset/sharegpt_test.jsonl \
        --num-requests 100 \
        --arrival-rate 5.0
"""

import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict


def generate_poisson_arrivals(num_requests: int, arrival_rate: float, seed: int = 42) -> List[float]:
    """Generate Poisson arrival times.

    Args:
        num_requests: Number of requests
        arrival_rate: Requests per second (lambda)
        seed: Random seed

    Returns:
        List of arrival times (cumulative)
    """
    np.random.seed(seed)
    # Inter-arrival times follow exponential distribution
    inter_arrival_times = np.random.exponential(1.0 / arrival_rate, num_requests)
    # Cumulative sum gives arrival times
    arrival_times = np.cumsum(inter_arrival_times)
    return arrival_times.tolist()


def load_sharegpt_dataset(
    sharegpt_path: str,
    num_requests: int,
    tokenizer_name: str = "meta-llama/Llama-3.1-8B",
    max_input_len: int = None,
    max_output_len: int = None,
) -> List[Dict]:
    """Load ShareGPT dataset and prepare prompts.

    Args:
        sharegpt_path: Path to ShareGPT JSON file
        num_requests: Number of conversations to load
        tokenizer_name: Tokenizer to use for length estimation
        max_input_len: Max input length filter (optional)
        max_output_len: Max output length filter (optional)

    Returns:
        List of dicts with 'prompt', 'prompt_len', 'output_len'
    """
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("ERROR: transformers not available. Install with: pip install transformers")
        import sys
        sys.exit(1)

    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    print(f"Loading ShareGPT data from {sharegpt_path}")
    with open(sharegpt_path, 'r') as f:
        sharegpt_data = json.load(f)

    print(f"Loaded {len(sharegpt_data)} conversations")

    # Process conversations
    prompts = []
    for item in sharegpt_data:
        conversations = item.get('conversations', [])

        if len(conversations) < 2:
            continue

        # First message is the prompt (user message)
        # Second message is the expected output (assistant response)
        prompt_text = ""
        output_text = ""

        for i, conv in enumerate(conversations):
            if conv['from'] == 'human':
                prompt_text = conv['value']
            elif conv['from'] == 'gpt' and prompt_text:
                output_text = conv['value']
                break

        if not prompt_text or not output_text:
            continue

        # Tokenize to get lengths
        prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=True)
        output_tokens = tokenizer.encode(output_text, add_special_tokens=False)

        prompt_len = len(prompt_tokens)
        output_len = len(output_tokens)

        # Apply filters if specified
        if max_input_len and prompt_len > max_input_len:
            continue
        if max_output_len and output_len > max_output_len:
            continue

        prompts.append({
            'prompt': prompt_text,
            'prompt_len': prompt_len,
            'output_len': output_len,
        })

        if len(prompts) >= num_requests:
            break

    print(f"Prepared {len(prompts)} prompts")
    return prompts


def create_dataset(
    sharegpt_path: str,
    output_path: str,
    num_requests: int,
    arrival_rate: float,
    tokenizer_name: str = "meta-llama/Llama-3.1-8B",
    max_input_len: int = None,
    max_output_len: int = None,
    seed: int = 42,
):
    """Create a JSONL dataset from ShareGPT.

    Args:
        sharegpt_path: Path to ShareGPT JSON
        output_path: Output JSONL path
        num_requests: Number of requests
        arrival_rate: Requests per second
        tokenizer_name: Tokenizer for encoding
        max_input_len: Optional max input length filter
        max_output_len: Optional max output length filter
        seed: Random seed
    """
    # Load ShareGPT prompts
    prompts = load_sharegpt_dataset(
        sharegpt_path=sharegpt_path,
        num_requests=num_requests,
        tokenizer_name=tokenizer_name,
        max_input_len=max_input_len,
        max_output_len=max_output_len,
    )

    if len(prompts) < num_requests:
        print(f"WARNING: Only found {len(prompts)} valid prompts (requested {num_requests})")
        num_requests = len(prompts)

    # Generate arrival times
    print(f"Generating Poisson arrivals with rate {arrival_rate} req/s")
    arrival_times = generate_poisson_arrivals(num_requests, arrival_rate, seed)

    # Create output directory if needed
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Write JSONL
    print(f"Writing dataset to {output_path}")
    with open(output_path, 'w') as f:
        for i in range(num_requests):
            entry = {
                'request_id': f'req_{i}',
                'prompt': prompts[i]['prompt'],
                'prompt_len': prompts[i]['prompt_len'],
                'output_len': prompts[i]['output_len'],
                'arrival_time': arrival_times[i],
            }
            f.write(json.dumps(entry) + '\n')

    print(f"✓ Created dataset with {num_requests} requests")
    print(f"  Output: {output_path}")
    print(f"  Arrival rate: {arrival_rate} req/s")
    print(f"  Time span: {arrival_times[-1]:.2f}s")

    # Print statistics
    prompt_lens = [p['prompt_len'] for p in prompts[:num_requests]]
    output_lens = [p['output_len'] for p in prompts[:num_requests]]

    print(f"\nPrompt length statistics:")
    print(f"  Mean: {np.mean(prompt_lens):.1f} tokens")
    print(f"  Median: {np.median(prompt_lens):.1f} tokens")
    print(f"  Min: {np.min(prompt_lens)} tokens")
    print(f"  Max: {np.max(prompt_lens)} tokens")

    print(f"\nOutput length statistics:")
    print(f"  Mean: {np.mean(output_lens):.1f} tokens")
    print(f"  Median: {np.median(output_lens):.1f} tokens")
    print(f"  Min: {np.min(output_lens)} tokens")
    print(f"  Max: {np.max(output_lens)} tokens")


def main():
    parser = argparse.ArgumentParser(
        description="Generate dataset from ShareGPT with Poisson arrivals"
    )

    parser.add_argument("--sharegpt-path", type=str, required=True,
                       help="Path to ShareGPT JSON file")
    parser.add_argument("--output", "-o", type=str, required=True,
                       help="Output JSONL path")
    parser.add_argument("--num-requests", type=int, default=100,
                       help="Number of requests")
    parser.add_argument("--arrival-rate", type=float, default=5.0,
                       help="Request arrival rate (req/s)")
    parser.add_argument("--tokenizer", type=str, default="meta-llama/Llama-3.1-8B",
                       help="Tokenizer model name")
    parser.add_argument("--max-input-len", type=int,
                       help="Filter prompts longer than this")
    parser.add_argument("--max-output-len", type=int,
                       help="Filter outputs longer than this")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")

    args = parser.parse_args()

    create_dataset(
        sharegpt_path=args.sharegpt_path,
        output_path=args.output,
        num_requests=args.num_requests,
        arrival_rate=args.arrival_rate,
        tokenizer_name=args.tokenizer,
        max_input_len=args.max_input_len,
        max_output_len=args.max_output_len,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
