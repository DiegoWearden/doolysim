"""Request generator using vLLM's dataset generation for reproducibility.

This ensures DoolySim and vLLM benchmarks use identical requests.
"""

import os
from typing import List, Tuple
from dataclasses import dataclass
import numpy as np

try:
    from vllm.benchmarks.datasets import (
        RandomDataset,
        ShareGPTDataset,
        SampleRequest,
        ConversationDataset,
    )
    from vllm import LLM
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    SampleRequest = None
    ConversationDataset = None


@dataclass
class Request:
    """DoolySim request format."""
    request_id: str
    prompt_token_ids: List[int]
    max_tokens: int
    arrival_time: float

def generate_vllm_requests_from_instance(
    llm: LLM,
    num_requests: int,
    arrival_rate: float,
    input_len: int,
    output_len: int,
    range_ratio: float = 0.0,
    fixed_output_len: int | None = None,
    seed: int = 42,
) -> Tuple[List[Request], List[SampleRequest]]:
    """Generate requests with vLLM's RandomDataset.

    If ``fixed_output_len`` is provided, every request's expected output
    length is overridden to that value after generation, while the input
    length still varies according to ``range_ratio``. This is useful when
    you want to vary input length only and hold output length constant
    (e.g., long-context simulator validation).
    """
    tokenizer = llm.get_tokenizer()

    # Create vLLM's RandomDataset
    dataset = RandomDataset(random_seed=seed, disable_shuffle=True)

    print(f"Generating {num_requests} requests with vLLM RandomDataset...")
    print(f"  Input length: {input_len} (range_ratio={range_ratio})")
    if fixed_output_len is not None:
        print(f"  Output length: {fixed_output_len} (fixed; range_ratio ignored for output)")
    else:
        print(f"  Output length: {output_len} (range_ratio={range_ratio})")
    print(f"  Seed: {seed}")

    # Generate requests using vLLM's method
    vllm_requests = dataset.sample(
        tokenizer=tokenizer,
        num_requests=num_requests,
        input_len=input_len,
        output_len=output_len,
        burstiness=5.0,
        range_ratio=range_ratio,
        request_id_prefix="req_",
    )

    # Override output length if a fixed value is requested
    if fixed_output_len is not None:
        for req in vllm_requests:
            req.expected_output_len = fixed_output_len

    print(f"Generated {len(vllm_requests)} requests")

    # Generate arrival times using Poisson process
    if arrival_rate > 0:
        rng = np.random.default_rng(seed)
        inter_arrival_times = rng.exponential(scale=1.0 / arrival_rate, size=num_requests)
        arrival_times = np.cumsum(inter_arrival_times)
    elif arrival_rate == 0:
        arrival_times = np.zeros(num_requests)

    # Convert to DoolySim format
    dooly_requests = []
    for vllm_req, arrival_time in zip(vllm_requests, arrival_times):
        # Tokenize the prompt to get token IDs
        token_ids = tokenizer.encode(vllm_req.prompt)

        dooly_req = Request(
            request_id=vllm_req.request_id,
            prompt_token_ids=token_ids,
            max_tokens=vllm_req.expected_output_len,
            arrival_time=arrival_time,
        )
        dooly_requests.append(dooly_req)

    # Verify lengths match
    print("\nRequest verification:")
    print(f"  vLLM requests: {len(vllm_requests)}")
    print(f"  DoolySim requests: {len(dooly_requests)}")
    print(f"  Sample prompt lengths: {[r.prompt_len for r in vllm_requests[:5]]}")
    print(f"  Sample output lengths: {[r.expected_output_len for r in vllm_requests[:5]]}")
    print(f"  Sample arrival times: {[f'{t:.3f}' for t in arrival_times[:5]]}")

    return dooly_requests, vllm_requests


def generate_sharegpt_requests_from_instance(
    llm: LLM,
    num_requests: int,
    arrival_rate: float,
    sharegpt_path: str,
    min_input_len: int | None = None,
    max_input_len: int | None = None,
    custom_output_len: int | None = None,
    seed: int = 42,
) -> Tuple[List[Request], List[SampleRequest]]:
    """Generate requests from ShareGPT dataset using vLLM's ShareGPTDataset.

    Args:
        llm: vLLM LLM instance (for tokenizer)
        num_requests: Number of requests to sample
        custom_output_len: Custom output length for all requests
        arrival_rate: Requests per second (for Poisson arrivals)
        sharegpt_path: Path to ShareGPT JSON file
        seed: Random seed

    Returns:
        (dooly_requests, vllm_requests) tuple
    """
    tokenizer = llm.get_tokenizer()

    # Create vLLM's ShareGPTDataset
    dataset = ShareGPTDataset(dataset_path=sharegpt_path, random_seed=seed)

    print(f"Sampling {num_requests} requests from ShareGPT dataset...")
    print(f"  Dataset path: {sharegpt_path}")
    print(f"  Seed: {seed}")

    # Sample requests from ShareGPT
    vllm_requests = dataset.sample(
        tokenizer=tokenizer,
        num_requests=num_requests,
        min_input_len=min_input_len,
        max_input_len=max_input_len,
        output_len=custom_output_len,
        request_id_prefix="req_",
    )

    print(vllm_requests)

    print(f"Sampled {len(vllm_requests)} requests from ShareGPT")

    # Generate arrival times using Poisson process
    if arrival_rate > 0.0:
        rng = np.random.default_rng(seed)
        inter_arrival_times = rng.exponential(scale=1.0 / arrival_rate, size=len(vllm_requests))
        arrival_times = np.cumsum(inter_arrival_times)
    else:
        arrival_times = np.zeros(len(vllm_requests)) 
        
    # Convert to DoolySim format
    dooly_requests = []
    for i, vllm_req in enumerate(vllm_requests):
        # Tokenize the prompt
        token_ids = tokenizer.encode(vllm_req.prompt)

        dooly_req = Request(
            request_id=vllm_req.request_id,
            prompt_token_ids=token_ids,
            max_tokens=vllm_req.expected_output_len,
            arrival_time=arrival_times[i],
        )
        dooly_requests.append(dooly_req)

    print(f"\n✓ Generated {len(dooly_requests)} requests")
    print(f"  Sample prompt lengths: {[len(r.prompt_token_ids) for r in dooly_requests[:5]]}")
    print(f"  Sample output lengths: {[r.max_tokens for r in dooly_requests[:5]]}")
    print(f"  Sample arrival times: {[f'{t:.3f}' for t in arrival_times[:5]]}")

    return dooly_requests, vllm_requests


def generate_hf_conversation_requests_from_instance(
    llm: LLM,
    num_requests: int,
    arrival_rate: float,
    dataset_path: str,
    dataset_split: str = "train",
    dataset_subset: str | None = None,
    custom_output_len: int | None = None,
    seed: int = 42,
) -> Tuple[List[Request], List[SampleRequest]]:
    """Generate requests from any HuggingFace conversation dataset via vLLM's ConversationDataset.

    Works with HF datasets that have the ShareGPT `conversations` schema, e.g.
    `Aeala/ShareGPT_Vicuna_unfiltered` or `anon8231489123/ShareGPT_Vicuna_unfiltered`.

    Datasets with a different schema (e.g. `openchat/openchat_sharegpt4_dataset`,
    which is pre-tokenized) will fail inside ConversationDataset.sample().

    Args:
        llm: vLLM LLM instance (for tokenizer)
        num_requests: Number of requests to sample
        arrival_rate: Requests per second (Poisson). 0 = all arrive at t=0.
        dataset_path: HuggingFace dataset path (e.g. "Aeala/ShareGPT_Vicuna_unfiltered")
        dataset_split: HF split to load (default "train")
        dataset_subset: Optional HF config/subset name
        custom_output_len: Force a fixed output length; None = use dataset's completion len
        seed: Random seed

    Returns:
        (dooly_requests, vllm_requests)
    """
    if ConversationDataset is None:
        raise ImportError(
            "vllm.benchmarks.datasets.ConversationDataset is unavailable. "
            "Upgrade vLLM or check the install."
        )

    tokenizer = llm.get_tokenizer()

    dataset = ConversationDataset(
        dataset_path=dataset_path,
        dataset_split=dataset_split,
        dataset_subset=dataset_subset,
        random_seed=seed,
    )

    print(f"Sampling {num_requests} requests from HF dataset {dataset_path}...")
    print(f"  split={dataset_split} subset={dataset_subset} seed={seed}")

    vllm_requests = dataset.sample(
        tokenizer=tokenizer,
        num_requests=num_requests,
        output_len=custom_output_len,
        request_id_prefix="req_",
    )

    print(f"Sampled {len(vllm_requests)} requests")

    if arrival_rate > 0.0:
        rng = np.random.default_rng(seed)
        inter = rng.exponential(scale=1.0 / arrival_rate, size=len(vllm_requests))
        arrival_times = np.cumsum(inter)
    else:
        arrival_times = np.zeros(len(vllm_requests))

    dooly_requests = []
    for vllm_req, arrival_time in zip(vllm_requests, arrival_times):
        token_ids = tokenizer.encode(vllm_req.prompt)
        dooly_requests.append(Request(
            request_id=vllm_req.request_id,
            prompt_token_ids=token_ids,
            max_tokens=vllm_req.expected_output_len,
            arrival_time=arrival_time,
        ))

    print(f"\n✓ Generated {len(dooly_requests)} requests")
    print(f"  Sample prompt lengths: {[len(r.prompt_token_ids) for r in dooly_requests[:5]]}")
    print(f"  Sample output lengths: {[r.max_tokens for r in dooly_requests[:5]]}")
    print(f"  Sample arrival times: {[f'{t:.3f}' for t in arrival_times[:5]]}")

    return dooly_requests, vllm_requests


def generate_openchat_sharegpt4_requests_from_instance(
    llm: LLM,
    num_requests: int,
    arrival_rate: float,
    min_input_len: int | None = None,
    max_input_len: int | None = None,
    custom_output_len: int | None = None,
    dataset_file: str = "sharegpt_gpt4.json",
    seed: int = 42,
) -> Tuple[List[Request], List[SampleRequest]]:
    """Generate requests from openchat/openchat_sharegpt4_dataset.

    Pulls `sharegpt_gpt4.json` from the HF Hub. That file is structured like
    ShareGPT V3 but uses `items` instead of `conversations`:
        [{"items": [{"from": "human", "value": "..."},
                    {"from": "gpt",   "value": "..."}, ...], ...}, ...]

    Output format matches `generate_sharegpt_dataset.py`: the first human turn
    becomes the prompt, the following gpt turn becomes the expected output, and
    both are tokenized with the target model's tokenizer.

    Args:
        llm: vLLM LLM instance (for tokenizer).
        num_requests: Number of requests to return.
        arrival_rate: Requests per second (Poisson). 0 = all arrive at t=0.
        min_input_len: Drop samples with prompt_len < this.
        max_input_len: Drop samples with prompt_len > this.
        custom_output_len: Force a fixed output length; None = use dataset's output len.
        dataset_file: Filename inside the HF repo to load.
        seed: Random seed for shuffling and arrival times.

    Returns:
        (dooly_requests, vllm_requests)
    """
    import json
    import random
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required. Install with: pip install huggingface_hub"
        ) from e

    tokenizer = llm.get_tokenizer()

    print(f"Downloading openchat/openchat_sharegpt4_dataset::{dataset_file}...")
    path = hf_hub_download(
        repo_id="openchat/openchat_sharegpt4_dataset",
        filename=dataset_file,
        repo_type="dataset",
    )
    with open(path, "r") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} raw conversations")

    rng = random.Random(seed)
    rng.shuffle(data)

    collected: List[SampleRequest] = []
    skipped_no_turn = 0
    skipped_len = 0

    for entry in data:
        turns = entry.get("items") or entry.get("conversations") or []
        if len(turns) < 2:
            skipped_no_turn += 1
            continue

        # First human turn -> first gpt turn following it.
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

        prompt_ids = tokenizer.encode(prompt_text)
        prompt_len = len(prompt_ids)

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

        collected.append(SampleRequest(
            prompt=prompt_text,
            prompt_len=prompt_len,
            expected_output_len=output_len,
            request_id=f"req_{len(collected)}",
        ))
        if len(collected) >= num_requests:
            break

    if len(collected) < num_requests:
        print(
            f"WARNING: only collected {len(collected)}/{num_requests} "
            f"(skipped_no_turn={skipped_no_turn}, skipped_len={skipped_len})"
        )

    n = len(collected)
    if arrival_rate > 0.0:
        np_rng = np.random.default_rng(seed)
        inter = np_rng.exponential(scale=1.0 / arrival_rate, size=n)
        arrival_times = np.cumsum(inter)
    else:
        arrival_times = np.zeros(n)

    dooly_requests: List[Request] = []
    for vllm_req, arrival_time in zip(collected, arrival_times):
        dooly_requests.append(Request(
            request_id=vllm_req.request_id,
            prompt_token_ids=tokenizer.encode(vllm_req.prompt),
            max_tokens=vllm_req.expected_output_len,
            arrival_time=float(arrival_time),
        ))

    print(f"\n✓ Generated {len(dooly_requests)} requests from openchat_sharegpt4_dataset")
    print(f"  Sample prompt lengths: {[len(r.prompt_token_ids) for r in dooly_requests[:5]]}")
    print(f"  Sample output lengths: {[r.max_tokens for r in dooly_requests[:5]]}")
    print(f"  Sample arrival times: {[f'{t:.3f}' for t in arrival_times[:5]]}")

    return dooly_requests, collected


def generate_vllm_requests(
    model: str,
    num_requests: int,
    arrival_rate: float,
    input_len: int,
    output_len: int,
    range_ratio: float = 0.0,
    seed: int = 42,
) -> Tuple[List[Request], List[SampleRequest]]:
    if not VLLM_AVAILABLE:
        raise ImportError("vLLM is not installed. Install with: pip install vllm")

    print(f"Initializing tokenizer for {model}...")
    # Initialize LLM to get tokenizer
    llm = LLM(
        model=model,
        download_dir=None,
        load_format="dummy",  # Don't load weights, just tokenizer
        enforce_eager=True,
    )
    tokenizer = llm.get_tokenizer()

    # Create vLLM's RandomDataset
    dataset = RandomDataset(random_seed=seed, disable_shuffle=True)

    print(f"Generating {num_requests} requests with vLLM RandomDataset...")
    print(f"  Input length: {input_len} (range_ratio={range_ratio})")
    print(f"  Output length: {output_len} (range_ratio={range_ratio})")
    print(f"  Seed: {seed}")

    # Generate requests using vLLM's method
    vllm_requests = dataset.sample(
        tokenizer=tokenizer,
        num_requests=num_requests,
        input_len=input_len,
        output_len=output_len,
        burstiness=5.0,
        range_ratio=range_ratio,
        request_id_prefix="req_",
    )

    print(f"Generated {len(vllm_requests)} requests")

    # Generate arrival times using Poisson process
    rng = np.random.default_rng(seed)
    inter_arrival_times = rng.exponential(scale=1.0 / arrival_rate, size=num_requests)
    arrival_times = np.cumsum(inter_arrival_times)

    # Convert to DoolySim format
    dooly_requests = []
    for vllm_req, arrival_time in zip(vllm_requests, arrival_times):
        # Tokenize the prompt to get token IDs
        token_ids = tokenizer.encode(vllm_req.prompt)

        dooly_req = Request(
            request_id=vllm_req.request_id,
            prompt_token_ids=token_ids,
            max_tokens=vllm_req.expected_output_len,
            arrival_time=arrival_time,
        )
        dooly_requests.append(dooly_req)

    # Verify lengths match
    print("\nRequest verification:")
    print(f"  vLLM requests: {len(vllm_requests)}")
    print(f"  DoolySim requests: {len(dooly_requests)}")
    print(f"  Sample prompt lengths: {[r.prompt_len for r in vllm_requests[:5]]}")
    print(f"  Sample output lengths: {[r.expected_output_len for r in vllm_requests[:5]]}")
    print(f"  Sample arrival times: {[f'{t:.3f}' for t in arrival_times[:5]]}")

    return dooly_requests, vllm_requests


def export_vllm_dataset_jsonl(
    vllm_requests: List[SampleRequest],
    arrival_times: List[float],
    output_path: str,
):
    """Export vLLM requests to JSONL format for vllm bench serve.

    Args:
        vllm_requests: List of SampleRequest from generate_vllm_requests()
        arrival_times: List of arrival times for each request
        output_path: Path to output JSONL file
    """
    import json
    import os

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with open(output_path, 'w') as f:
        for req, arrival_time in zip(vllm_requests, arrival_times):
            entry = {
                "prompt": req.prompt,
                "output_len": req.expected_output_len,
                "request_id": req.request_id,
                "prompt_len": req.prompt_len,
                "arrival_time": arrival_time,
            }
            f.write(json.dumps(entry) + "\n")

    print(f"Exported {len(vllm_requests)} requests to {output_path}")


if __name__ == "__main__":
    # Long-context synthetic dataset for §7 evaluation:
    #   500 requests, input ~2048-8192 tokens, output fixed at 128, RPS=0.5
    OUTPUT_PATH = os.environ.get(
        "DOOLYSIM_DATASET_PATH",
        "./dataset/longctx_2k-8k_rps0.5_out128.jsonl",
    )

    llm = LLM(model="meta-llama/Llama-3.1-8B", load_format="dummy", enforce_eager=True)

    dooly_reqs, vllm_reqs = generate_vllm_requests_from_instance(
        llm=llm,
        num_requests=500,
        arrival_rate=0.5,
        input_len=5120,             # midpoint of 2K-8K
        output_len=128,             # consumed by RandomDataset; overridden below
        range_ratio=0.6,            # input ~ [2048, 8192]
        fixed_output_len=128,       # every request capped at exactly 128 output tokens
        seed=42,
    )

    arrival_times = [r.arrival_time for r in dooly_reqs]
    export_vllm_dataset_jsonl(vllm_reqs, arrival_times, OUTPUT_PATH)

    prompt_lens = [r.prompt_len for r in vllm_reqs]
    output_lens = sorted(set(r.expected_output_len for r in vllm_reqs))
    span = arrival_times[-1] if arrival_times else 0.0
    print()
    print(f"Dataset summary ({OUTPUT_PATH}):")
    print(f"  requests       : {len(vllm_reqs)}")
    print(f"  input tokens   : min={min(prompt_lens)}, max={max(prompt_lens)}, "
          f"mean={sum(prompt_lens)/len(prompt_lens):.0f}")
    print(f"  output tokens  : {output_lens} (single value expected)")
    print(f"  arrival span   : {span:.1f} s  (effective RPS={len(vllm_reqs)/span:.2f})")
