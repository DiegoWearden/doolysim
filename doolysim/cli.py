"""Command-line interface for DoolySim."""

import argparse
import os
from pathlib import Path
from .core.dooly_sim import DoolySim
from .request import RequestStream
from .analysis.reporters import print_summary
from .utils import load_requests_from_jsonl

VERBOSE = False

def main():
    parser = argparse.ArgumentParser(description="DoolySim - vLLM Scheduler Simulator")

    # Model configuration ################################################
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B",
                       help="Model name to simulate")
    parser.add_argument("--backend", type=str, default="FLASHINFER",
                       help="Backend used for profiling")
    parser.add_argument("--tp", type=int, default=1,
                       help="Tensor parallel size (sets VLLM_FAKE_TP env var)")
    parser.add_argument("--dtype", type=str, default="auto",
                        help="Model dtype passed to the emulated vLLM engine (auto, bfloat16, float16); selects the profiled dtype")
    parser.add_argument("--kv-cache-dtype", type=str, default="auto",
                        help="vLLM KV-cache dtype literal (auto, fp8, fp8_e4m3, fp8_e5m2); passed to the emulated engine and "
                             "used to select the profiled KV-cache dtype")
    parser.add_argument("--topology", type=str, default="mew1",
                       help="GPU interconnect topology for collective operations (default: mew1)")
    # Paths ################################################
    parser.add_argument("--cache-dir", type=str,
                       default=os.environ.get("DOOLYSIM_CACHE_DIR"),
                       help="Directory containing cached regression models "
                            "(or set DOOLYSIM_CACHE_DIR env var)")
    parser.add_argument("--db-path", type=str,
                       default=os.environ.get("DOOLYSIM_DB_PATH"),
                       help="Path to profiling database "
                            "(or set DOOLYSIM_DB_PATH env var)")

    # vLLM configuration
    parser.add_argument("--max-num-seqs", type=int, default=256,
                       help="Maximum number of sequences")
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192,
                       help="Maximum number of batched tokens (set high enough to avoid prefill chunking)")
    parser.add_argument("--max-model-len", type=int, default=16384,
                       help="Maximum model context length") 
    parser.add_argument("--gpu-mem-util", type=float, default=0.9,
                       help="GPU memory utilization")


    # GENERATION OPTIONS ################################################
    # Loading pre-generated dataset 
    parser.add_argument("--load-dataset", type=str,
                       help="Load requests from a pre-generated JSONL file (for exact comparison)\
                           This ignores num requests and arrival rate set in other options")
    
    # Request Generation Options
    parser.add_argument("--export-dataset-path", type=str,
                       help="Export vLLM dataset to JSONL file for vllm bench serve")
    parser.add_argument("--synthetic", action="store_true",
                       help="Use vLLM's RandomDataset generator for reproducible benchmarks")
    parser.add_argument("--input-len", type=int, default=1024,
                       help="Mean input length")
    parser.add_argument("--output-len", type=int, default=128,
                       help="Mean output length (use this for sharegpt too if you want to override actual output lengths in dataset)")
    parser.add_argument("--range-ratio", type=float, default=0.0,
                       help="Length variability (0.0=fixed, 1.0=uniform 0 to 2x mean)")
    
    # Common parameters for both generators
    parser.add_argument("--num-requests", type=int, default=20,
                       help="Number of requests to simulate")
    parser.add_argument("--arrival-rate", type=float, default=5.0,
                       help="Request arrival rate (requests/sec)")
    
    # ShareGPT dataset option
    parser.add_argument("--sharegpt-path", type=str, metavar="SHAREGPT_PATH",
                       help="Use ShareGPT dataset (path to ShareGPT JSON file)")

    # DoolySim Native RequestStream parameters (range-based)
    parser.add_argument("--prompt-len-min", type=int, default=256,
                       help="[RequestStream] Minimum prompt length")
    parser.add_argument("--prompt-len-max", type=int, default=4096,
                       help="[RequestStream] Maximum prompt length")
    parser.add_argument("--output-len-min", type=int, default=64,
                       help="[RequestStream] Minimum output length")
    parser.add_argument("--output-len-max", type=int, default=512,
                       help="[RequestStream] Maximum output length")


    # Simulator Operations ###############################################
    parser.add_argument("--max-steps", type=int, default=10000000,
                       help="Maximum simulation steps")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--show-per-request", action="store_true",
                       help="Show detailed per-request statistics")

    # Export options #######################################################
    parser.add_argument("--export-csv", action="store_true",
                       help="Export CSV files for analysis")
    parser.add_argument("--csv-output-dir", type=str, default="./csv_output",
                       help="Directory to save CSV files")
    parser.add_argument("--csv-prefix", type=str, default="simulation",
                       help="Prefix for CSV filenames")
    parser.add_argument("--export-json", action="store_true",
                       help="Export JSON in vLLM benchmark format")
    parser.add_argument("--json-output", type=str,
                       help="Output path for vLLM-format JSON file")
    parser.add_argument("--export-batch-composition", action="store_true",
                       help="Export detailed batch composition to JSON")
    parser.add_argument("--batch-composition-output", type=str,
                       help="Output path for batch composition JSON file")

    # Fallback ################################################
    parser.add_argument("--fallback-duration-ms", type=float, default=0.0,
                       help="Fallback step duration in ms if models fail to load")
    parser.add_argument("--cpu-overhead-ms", type=float, default=0.0)

    args = parser.parse_args()

    # Auto-generate output filenames based on dataset parameters if not explicitly provided
    if args.load_dataset and args.load_dataset.endswith('.jsonl'):
        # Extract dataset name from load-dataset path
        dataset_name = Path(args.load_dataset).stem

        # Auto-set output files if they're not explicitly provided
        if not args.json_output:
            args.json_output = f"./comparison/dooly_{dataset_name}.json"

        if not args.batch_composition_output and args.export_batch_composition:
            args.batch_composition_output = f"./comparison/dooly_batch_{dataset_name}.json"

    elif args.sharegpt_path:
        # ShareGPT dataset: sharegpt-{rate}rps-{num_requests}reqs
        dataset_name = f"sharegpt-{args.arrival_rate:.0f}rps-{args.num_requests}reqs"

        # Auto-set output files if they're not explicitly provided
        if not args.json_output:
            args.json_output = f"./comparison/dooly_{dataset_name}.json"

        if not args.batch_composition_output and args.export_batch_composition:
            args.batch_composition_output = f"./comparison/dooly_batch_{dataset_name}.json"

    elif not args.load_dataset:
        # Generate dataset name pattern based on request generation parameters
        if args.synthetic:
            # vLLM RandomDataset: syn-{rate}rps-uniform-{input_len}-{output_len}
            dataset_name = f"syn-{args.arrival_rate:.0f}rps-uniform-{args.input_len}-{args.output_len}"
        else:
            # DoolySim RequestStream: range-based naming
            dataset_name = f"dooly-{args.arrival_rate:.0f}rps-{args.prompt_len_min}-{args.prompt_len_max}-{args.output_len_min}-{args.output_len_max}"

        # Auto-set output files if they're not explicitly provided
        if not args.json_output:
            args.json_output = f"./comparison/dooly_{dataset_name}.json"

        if not args.batch_composition_output and args.export_batch_composition:
            args.batch_composition_output = f"./comparison/dooly_batch_{dataset_name}.json"

    # End of CLI Options ################################################

    # Set global VERBOSE flag
    global VERBOSE
    VERBOSE = args.verbose

    # Print header
    print()
    print("=" * 60)
    print("DoolySim (w Emulated vLLM scheduler)")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Backend: {args.backend}")
    print(f"TP: {args.tp}  dtype: {args.dtype}  kv_cache_dtype: {args.kv_cache_dtype}")
    print(f"Cache dir: {args.cache_dir}")
    print(f"DB path: {args.db_path}")
    print()

    # Create simulator first (so we have tokenizer for loading dataset)
    simulator = DoolySim(
        step_duration_ms=args.fallback_duration_ms,
        model=args.model,
        args={
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_mem_util,
            "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "enable_prefix_caching": False,
            "dtype": args.dtype,
            "kv_cache_dtype": args.kv_cache_dtype,
        },
        cache_dir=args.cache_dir,
        db_path=args.db_path,
        backend=args.backend,
        topology=args.topology,  # Pass topology for collective operations
        tensor_parallel_size=args.tp,  # Set TP size for profiling and simulation
        cpu_overhead=args.cpu_overhead_ms,
    )

    simulator.dump_config()

    # Generate or load requests
    if args.load_dataset:
        # Load from pre-generated JSONL file
        print(f"Loading dataset from {args.load_dataset}")
        dooly_requests = load_requests_from_jsonl(args.load_dataset, simulator.tokenizer)
        requests = iter(dooly_requests)
    elif args.sharegpt_path:
        print(f"Using ShareGPT dataset for request generation: {args.sharegpt_path}")
        from doolysim.tools.vllm_request_generator import generate_sharegpt_requests_from_instance, export_vllm_dataset_jsonl

        dooly_requests, vllm_requests = generate_sharegpt_requests_from_instance(
            llm=simulator.llm,
            num_requests=args.num_requests,
            arrival_rate=args.arrival_rate,
            sharegpt_path=args.sharegpt_path,
            custom_output_len=args.output_len,
            seed=args.seed,
        )

        # Export dataset if requested
        if args.export_dataset_path:
            arrival_times = [r.arrival_time for r in dooly_requests]
            export_vllm_dataset_jsonl(vllm_requests, arrival_times, args.export_dataset_path)

        requests = iter(dooly_requests)
    elif args.synthetic:
        print("Using vLLM RandomDataset for request generation")
        from doolysim.tools.vllm_request_generator import generate_vllm_requests, export_vllm_dataset_jsonl

        dooly_requests, vllm_requests = generate_vllm_requests(
            model=args.model,
            num_requests=args.num_requests,
            arrival_rate=args.arrival_rate,
            input_len=args.input_len,
            output_len=args.output_len,
            range_ratio=args.range_ratio,
            seed=args.seed,
        )

        # Export vLLM dataset if requested
        if args.export_dataset_path:
            arrival_times = [r.arrival_time for r in dooly_requests]
            export_vllm_dataset_jsonl(vllm_requests, arrival_times, args.export_dataset_path)

        requests = iter(dooly_requests)
    else:
        # Use DoolySim's own RequestStream
        print("Using DoolySim RequestStream for request generation")
        stream = RequestStream(
            num_requests=args.num_requests,
            arrival_rate=args.arrival_rate,
            prompt_len_range=(args.prompt_len_min, args.prompt_len_max),
            output_len_range=(args.output_len_min, args.output_len_max),
            seed=args.seed,
        )
        requests = stream.generate()

    # Run simulation
    result = simulator.run(
        request_stream=requests,
        max_steps=args.max_steps,
        verbose=args.verbose,
    )

    # Print results
    print_summary(result, show_per_request=args.show_per_request)

    # Export CSV files if requested
    if args.export_csv:
        result.export_all_csvs(args.csv_output_dir, args.csv_prefix)

    # Export JSON in vLLM format if requested
    if args.export_json or args.json_output:
        json_path = args.json_output if args.json_output else os.path.join(args.csv_output_dir, f"{args.csv_prefix}.json")
        result.export_vllm_json(json_path, args.model, args.arrival_rate)

    # Export batch composition if requested
    if args.export_batch_composition or args.batch_composition_output:
        batch_path = args.batch_composition_output if args.batch_composition_output else os.path.join(args.csv_output_dir, "batch-composition.json")
        result.export_batch_composition_json(batch_path)


if __name__ == "__main__":
    main()
