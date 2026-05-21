#!/usr/bin/env python3
"""
Test script to run vLLM and Dooly benchmarks based on comparison-configs.json
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any

def parse_config_file(config_path: str) -> List[Dict[str, Any]]:
    """Parse the comparison-configs.json file"""
    with open(config_path, 'r') as f:
        content = f.read()

    # Fix the malformed JSON by converting to proper array format
    lines = content.strip().split('\n')
    configs = []
    current_config = {}

    for line in lines:
        line = line.strip()
        if line.startswith('{'):
            current_config = {}
        elif line.startswith('}'):
            if current_config:
                configs.append(current_config)
                current_config = {}
        elif ':' in line:
            # Parse key-value pairs
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip().rstrip(',')

            if value.startswith('[') and value.endswith(']'):
                # Parse array
                value = value[1:-1]  # Remove brackets
                value = [float(v.strip()) for v in value.split(',')]
            else:
                # Parse single value
                try:
                    value = float(value)
                except ValueError:
                    value = value.strip('"\'')

            current_config[key] = value

    return configs

def create_directories():
    """Create results directories"""
    os.makedirs("results/vllm", exist_ok=True)
    os.makedirs("results/dooly", exist_ok=True)
    os.makedirs("dataset", exist_ok=True)

def run_vllm_benchmark(max_num_batched_tokens: int, input_len: int, output_len: int, arrival_rate: float) -> str:
    """Run vLLM streaming benchmark and return output file path"""
    dataset_file = f"dataset/syn-{arrival_rate}rps-uniform-{input_len}-{output_len}.jsonl"

    cmd = [
        sys.executable, "-m", "doolysim.benchmark.measure_vllm_streaming",
        "--num-requests", "100",
        "--input-len", str(int(input_len)),
        "--output-len", str(int(output_len)),
        "--arrival-rate", str(arrival_rate),
        "--max-num-batched-tokens", str(max_num_batched_tokens),
        "--max-model-len", "16384",
        "--max-num-seqs", "256",
        "--gpu-memory-utilization", "0.9",
        "--seed", "42",
        "--export-requests", dataset_file,
        "--verbose",
    ]

    print(f"Running vLLM benchmark: input_len={input_len}, output_len={output_len}, arrival_rate={arrival_rate}")
    print(f"Command: {' '.join(cmd)}")

    try:
        # Remove capture_output=True to show stdout in real-time
        result = subprocess.run(cmd, check=True, text=True)
        return dataset_file
    except subprocess.CalledProcessError as e:
        print(f"✗ vLLM benchmark failed: {e}")
        return None

def run_dooly_benchmark(max_num_batched_tokens: int, input_len: int, output_len: int, arrival_rate: float, dataset_file: str):
    """Run Dooly benchmark using the generated dataset"""

    db_path = os.environ.get("DOOLYSIM_DB_PATH", "")
    cache_dir = os.environ.get("DOOLYSIM_CACHE_DIR", "./cache/tp4/meta-llama/Llama-3.1-70B")
    cmd = [
        sys.executable, "-m", "doolysim.cli",
        "--model", "meta-llama/Llama-3.1-70B",
        "--backend", "FLASHINFER",
        "--cache-dir", cache_dir,
        "--db-path", db_path,
        "--tp", "4",  # Add TP=4 to match vLLM benchmark
        "--num-requests", "100",
        "--max-num-batched-tokens", str(max_num_batched_tokens),
        "--max-model-len", "16384",
        "--max-num-seqs", "256",
        "--gpu-mem-util", "0.9",
        "--seed", "42",
        "--arrival-rate", str(arrival_rate),
        "--load-dataset", dataset_file,
        "--export-batch-composition",
        "--verbose"
    ]

    print(f"Running Dooly benchmark: input_len={input_len}, output_len={output_len}, arrival_rate={arrival_rate}")
    print(f"Command: {' '.join(cmd)}")

    try:
        # Remove capture_output=True to show stdout in real-time
        result = subprocess.run(cmd, check=True, text=True)
        print(f"✓ Dooly benchmark completed")
    except subprocess.CalledProcessError as e:
        print(f"✗ Dooly benchmark failed: {e}")

def main():
    """Main execution function"""
    import argparse
    parser = argparse.ArgumentParser(description="Run vLLM and Dooly benchmarks based on comparison-configs.json")
    parser.add_argument("--config-file", type=str, required=True, help="Path to comparison-configs.json")
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192, help="Maximum number of batched tokens for Dooly benchmark")
    args = parser.parse_args()

    if not os.path.exists(config_file):
        print(f"Error: Config file not found: {config_file}")
        sys.exit(1)

    # Parse configuration
    configs = parse_config_file(config_file)
    print(f"Loaded {len(configs)} configurations")

    # Create directories
    create_directories()

    # Run benchmarks for each configuration
    total_runs = sum(len(config.get('arrival-rate', [])) for config in configs)
    current_run = 0

    for config in configs:
        input_len = config['input-len']
        output_len = config['output-len']
        arrival_rates = config['arrival-rate']

        for arrival_rate in arrival_rates:
            current_run += 1
            print(f"\n=== Run {current_run}/{total_runs} ===")

            # Run vLLM benchmark
            dataset_file = run_vllm_benchmark(args.max_num_batched_tokens, input_len, output_len, arrival_rate)

            # Run Dooly benchmark if vLLM succeeded
            if dataset_file and os.path.exists(dataset_file):
                run_dooly_benchmark(args.max_num_batched_tokens, input_len, output_len, arrival_rate, dataset_file)
            else:
                print(f"✗ Skipping Dooly benchmark due to vLLM failure")

    print(f"\n=== Completed {current_run} benchmark runs ===")
    print("Results stored in:")
    print("  - results/vllm/ (vLLM benchmark outputs)")
    print("  - results/dooly/ (Dooly benchmark outputs)")
    print("  - dataset/ (Generated request datasets)")

if __name__ == "__main__":
    main()