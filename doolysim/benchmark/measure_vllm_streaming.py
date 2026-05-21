#!/usr/bin/env python3
"""
Measure vLLM performance with streaming requests (arrival times).

This script uses vLLM's LLMEngine directly to add requests over time,
mimicking the online serving behavior that dooly_sim.py simulates.

This provides a fair baseline comparison where both systems:
1. Use the same request stream (from RandomDataset)
2. Feed requests at the same arrival times
3. Use the same vLLM scheduler logic
4. Use scheduler.schedule() output for metrics (matches Dooly's approach)
"""

import os
import sys
import time
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import torch
# Kineto profiling imports for CPU/GPU timing separation
from torch.profiler import profile, ProfilerActivity, record_function

# Disable torch.compile / Dynamo globally before vLLM imports so the
# measurement side matches the profiler-side trace, which is captured under
# taintracer.hooks.install_hooks (that utility also sets this flag for its
# own reasons). Without this, real vLLM runs modules like GemmaRMSNorm
# through a fused Triton kernel while the profiler recorded the individual
# aten kernels, so the simulator's predicted-latency sum wouldn't match
# measured single-kernel cost.
# import torch._dynamo
# torch._dynamo.config.disable = True

try:
    from vllm import LLM
    from vllm import EngineArgs, LLMEngine, SamplingParams
    from vllm.benchmarks.serve import calculate_metrics, BenchmarkMetrics
    from vllm.benchmarks.datasets import SampleRequest
    from vllm.benchmarks.lib.endpoint_request_func import RequestFuncOutput
except ImportError:
    print("ERROR: vLLM not available. Install with: pip install vllm")
    sys.exit(1)

import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"  

# Add simulator to path for request generation
simulator_dir = Path(__file__).parent.parent
if str(simulator_dir) not in sys.path:
    sys.path.insert(0, str(simulator_dir))

from doolysim.tools.vllm_request_generator import (
    generate_vllm_requests_from_instance,
    generate_sharegpt_requests_from_instance,
    generate_hf_conversation_requests_from_instance,
)


def run_streaming_benchmark(
    llm: LLM,
    requests: List[tuple],  # (request_id, prompt_token_ids, max_tokens, arrival_time)
    warmup: int = 0,
    verbose: bool = False,
) -> tuple[List[RequestFuncOutput], float, dict]:

    engine = llm.llm_engine
    outputs = []
    completed_requests = {}

    # Track batch information for each request
    request_batch_info = {}  # req_id -> {"ttft_batch_size": N, "ttft_step": M, "first_token_generated": False}

    # Track batch composition per step
    step_batches = []  # List of {"step": N, "batch_size": M, "req_ids": [...]}

    # Track wall-clock timestamps for each token (matching Dooly's token_events approach).
    # ITL = gap between consecutive token timestamps, avoiding step_duration attribution
    # errors from vLLM's pipelined execution (where engine.step() includes processing
    # the previous step's outputs before executing the current batch).
    request_token_timestamps = {}  # req_id -> [wall_clock_sec, ...]

    # CPU/GPU timing data collection disabled - using timestamp instrumentation instead
    cpu_gpu_timing_data = []

    # Track first arrival time to normalize all times (like measure_vllm_direct.py does on line 142)
    first_request_arrival_time = None

    # Sort by arrival time
    requests = sorted(requests, key=lambda x: x[3])

    if verbose:
        print(f"\n{'='*80}")
        print(f"Starting streaming benchmark with {len(requests)} requests")
        print(f"{'='*80}\n")

    # warm up
    warmup_sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        ignore_eos=True,
    )

    print(f"Warming up ({warmup} iterations)...")
    for _ in range(warmup):
        # Pass token IDs as TextTokensPrompt objects
        from vllm.inputs import TokensPrompt
        prompts = [TokensPrompt(prompt_token_ids=token_ids) for _, token_ids, _, _ in requests]
        llm.generate(prompts, sampling_params=warmup_sampling_params)
        torch.cuda.synchronize()

    # ------------------------------------------------------------------
    # Streaming-path warmup: one dummy request through the exact
    # engine.add_request + engine.step() loop the real benchmark uses.
    # The llm.generate() warmup above autotunes kernels at packed-batch
    # shapes (e.g. M=max_num_batched_tokens); it does NOT warm the
    # "single isolated request" path the streaming loop hits at idx=0.
    # For MoE models (Marlin MXFP4) this path must be autotuned before
    # the real run or idx=0's prefill pays hundreds of ms of autotune.
    #
    # Side-effect safety: uses a unique request_id so it cannot collide
    # with real requests; drains engine before returning; never touches
    # the main-loop tracking dicts (request_token_timestamps,
    # request_batch_info, step_batches, request_arrival_times — all
    # still-empty at this point) or benchmark timers (start_time /
    # benchmark_start_time set AFTER this block).
    # ------------------------------------------------------------------
    _warmup_params = SamplingParams(
        temperature=0.0,
        max_tokens=8,
        ignore_eos=True,
    )
    # Warm every unique prompt length in the workload through an isolated
    # engine.step() so Triton FP8 MoE autotune (cached exact-by-M) hits on
    # every length the real streaming loop will later encounter. Quantile
    # sampling missed near-by lengths (e.g. M=1232 vs warmup M=1295), so
    # workload-driven exact coverage is the only robust option.
    # Cost: ~50ms per unique length × ~N_unique lengths in dataset.
    _sorted_tokens = sorted((r[1] for r in requests), key=len)
    _warmup_tokens_list = []
    _seen_lens: set = set()
    for tokens in _sorted_tokens:
        if len(tokens) in _seen_lens:
            continue
        _seen_lens.add(len(tokens))
        _warmup_tokens_list.append(tokens)

    print(
        f"Streaming-path warmup: {len(_warmup_tokens_list)} isolated-request "
        f"shapes through engine.step() "
        f"(M={sorted(len(t) for t in _warmup_tokens_list)})..."
    )
    for i, tokens in enumerate(_warmup_tokens_list):
        _warmup_req_id = f"__dooly_streaming_warmup_{i}__"
        engine.add_request(
            request_id=_warmup_req_id,
            prompt=tokens,
            params=_warmup_params,
        )
        while engine.has_unfinished_requests():
            engine.step()
    torch.cuda.synchronize()
    print("Streaming-path warmup done.")

    print(f"Warm up done. Running real runs.")
    # tqdm
    pbar = tqdm(total=len(requests), desc="Processing requests")
    start_time = time.perf_counter()

    # Record wall-clock time at benchmark start (for converting our relative times to wall-clock)
    benchmark_start_time = time.monotonic()

    # Map request_id to its arrival_time from requests list (relative to benchmark start)
    request_arrival_times = {}

    # Track time base offset (wall-clock - engine_time), set from first completed request
    time_base_offset = None

    request_idx = 0
    step_count = 0
    while request_idx < len(requests) or engine.has_unfinished_requests():
        # Calculate elapsed wall-clock time since benchmark start
        # This ensures requests are fed at the same absolute rate regardless of execution speed
        elapsed_time = time.perf_counter() - start_time

        # Add any requests that have arrived based on wall-clock time
        while request_idx < len(requests):
            request_id, prompt_token_ids, max_tokens, arrival_time = requests[request_idx]

            if arrival_time <= elapsed_time:
                # Create sampling params
                sampling_params = SamplingParams(
                    temperature=0.0,
                    max_tokens=max_tokens,
                    ignore_eos=True,
                )
                print(f"req_id: {request_id} - max_tokens: {max_tokens}")

                # Add request to engine
                engine.add_request(
                    request_id=request_id,
                    prompt=prompt_token_ids,
                    params=sampling_params,
                    arrival_time=benchmark_start_time+arrival_time
                )

                # Store the arrival_time from our requests list
                request_arrival_times[request_id] = arrival_time

                if verbose:
                    print(f"[t={elapsed_time:.3f}s] Added request {request_id} "
                          f"(prompt_len={len(prompt_token_ids)}, max_tokens={max_tokens})")

                request_idx += 1
            else:
                break

        # Step the engine if there are requests to process
        # if engine.has_unfinished_requests():
        scheduler = engine.engine_core.engine_core.scheduler  # Get first scheduler (for TP=1)

        # Store original schedule method to intercept the scheduling output
        original_schedule = scheduler.schedule
        captured_sched_output = None

        def capture_schedule():
            nonlocal captured_sched_output
            captured_sched_output = original_schedule()
            return captured_sched_output

        # Temporarily replace schedule method to capture output
        scheduler.schedule = capture_schedule

        # Original timing code (commented out for CPU/GPU split timing)
        torch.cuda.synchronize()
        step_start = time.perf_counter()

        # Debug: Check request scheduling behavior
        # print(f"\n[DEBUG] Step {step_count} - before engine.step()")
        # print(f"  Scheduler state: running={len(scheduler.running)}, waiting={len(scheduler.waiting)}")

        request_outputs = engine.step()

        torch.cuda.synchronize()  # Synchronize after to ensure this step's work is complete
        step_end = time.perf_counter()
        step_duration = step_end - step_start

        # Debug: Show what was actually scheduled and executed in this step
        # if captured_sched_output:
        #     print(f"[DEBUG] Step {step_count} - after engine.step()")
        #     print(f"  Scheduled new reqs: {len(captured_sched_output.scheduled_new_reqs)}")
        #     print(f"  Scheduled cached reqs: {captured_sched_output.scheduled_cached_reqs.num_reqs}")
        #     print(f"  Total tokens scheduled: {captured_sched_output.total_num_scheduled_tokens}")

        #     # Show request outputs
        #     print(f"  Request outputs returned: {len(request_outputs)}")
        #     for i, output in enumerate(request_outputs[:2]):
        #         print(f"    output[{i}]: id={output.request_id[:15]}, finished={output.finished}")
        #         if hasattr(output, 'outputs') and output.outputs:
        #             print(f"      tokens_generated={len(output.outputs[0].token_ids)}")

        print(f"  Step duration: {step_duration*1000:.2f}ms")

        # More debug for first step
        # if step_count == 0:
        #     print(f"[DEBUG] First step completed:")
        #     print(f"  Duration: {step_duration*1000:.2f}ms")
        #     print(f"  Outputs returned: {len(request_outputs)}")
        #     if request_outputs and hasattr(request_outputs[0], 'finished'):
        #         print(f"  First output finished: {request_outputs[0].finished}")

        current_time = time.perf_counter() - start_time
        # print(f"[t={current_time:.3f}s] Step {step_count} completed in {step_duration*1000:.2f}ms")

        # Track token timestamps for ITL calculation (matching Dooly's approach).
        # Records wall-clock time when each token is observed in request_outputs,
        # so ITL = gap between consecutive observations. This avoids step_duration
        # attribution issues from vLLM's pipelined engine.step().
        step_wall_time_sec = step_end - start_time
        for request_output in request_outputs:
            out_req_id = request_output.request_id
            if hasattr(request_output, 'outputs') and request_output.outputs:
                current_token_count = len(request_output.outputs[0].token_ids)
                if out_req_id not in request_token_timestamps:
                    request_token_timestamps[out_req_id] = []
                prev_count = len(request_token_timestamps[out_req_id])
                for _ in range(current_token_count - prev_count):
                    request_token_timestamps[out_req_id].append(step_wall_time_sec)

        # Restore original schedule method
        scheduler.schedule = original_schedule

        # Extract scheduling metrics from captured scheduler output (matching Dooly's approach)
        if captured_sched_output:
            num_scheduled_new = len(captured_sched_output.scheduled_new_reqs)
            num_scheduled_cached = captured_sched_output.scheduled_cached_reqs.num_reqs
            total_tokens_scheduled = captured_sched_output.total_num_scheduled_tokens
            scheduled_req_ids = list(captured_sched_output.num_scheduled_tokens.keys())
            current_batch_size = len(scheduled_req_ids)
            preempted_req_ids = list(captured_sched_output.preempted_req_ids) if captured_sched_output.preempted_req_ids else []
        else:
            # Fallback to old method if capture failed
            running_reqs = {req.request_id for req in scheduler.running}
            new_reqs = [req.request_id for req in scheduler.running if req.num_computed_tokens == 0]
            cached_reqs = [req.request_id for req in scheduler.running if req.num_computed_tokens > 0]

            num_scheduled_new = len(new_reqs)
            num_scheduled_cached = len(cached_reqs)
            total_tokens_scheduled = sum(req.num_computed_tokens for req in scheduler.running)
            scheduled_req_ids = list(running_reqs)
            current_batch_size = len(scheduled_req_ids)
            preempted_req_ids = []

        # Get KV cache info
        kv_cache_manager = scheduler.kv_cache_manager
        free_blocks = kv_cache_manager.block_pool.get_num_free_blocks()

        step_count += 1

        # Get finished/preempted from request_outputs (will populate below)
        finished_this_step = []

        # Record batch composition for this step (detailed)
        # IMPORTANT: Using scheduler.schedule() output for fair comparison with Dooly
        if total_tokens_scheduled > 0:
            step_batches.append({
                "step_id": step_count,  # Match Dooly's field name
                "sim_time_ms": (step_end - start_time) * 1000.0,  # step end time, matches dooly_batch convention
                "elapsed_time_ms": step_duration * 1000.0,  # **Add actual step duration**
                "batch_size": current_batch_size,
                "scheduled_req_ids": sorted(scheduled_req_ids),
                "num_scheduled_new": num_scheduled_new,
                "num_scheduled_cached": num_scheduled_cached,
                "total_tokens_scheduled": total_tokens_scheduled,
                "free_blocks": free_blocks,
                "waiting_count": len(scheduler.waiting),
                "running_count": len(scheduler.running),
                "finished_req_ids": finished_this_step,  # Will update after processing outputs
                "preempted_req_ids": preempted_req_ids,
            })

        # CPU/GPU timing collection disabled - using timestamp instrumentation instead
        # cpu_gpu_timing_data.append({
        #     "step": step_count,
        #     "wall_time_ms": wall_time_ms,
        #     "gpu_time_ms": gpu_time_ms,
        #     "cpu_overhead_ms": cpu_overhead_ms,
        #     "batch_size": current_batch_size,
        #     "is_prefill": num_scheduled_new > 0,
        #     "num_scheduled_new": num_scheduled_new,
        #     "num_scheduled_cached": num_scheduled_cached,
        # })

        if verbose:
            # Original logging
            print(f"[Step {step_count}] SCHED: reqs={sorted(scheduled_req_ids)}, "
                    f"tokens={total_tokens_scheduled}, "
                    f"new={num_scheduled_new}, cached={num_scheduled_cached}")

        # Process outputs
        for request_output in request_outputs:
            # if verbose:
            #     print(f"Step: {step_count}, request id: {request_output.request_id}, finished: {request_output.finished}")

            if request_output.finished:
                # Track finished requests for this step
                finished_this_step.append(request_output.request_id)
                # Request completed
                req_id = request_output.request_id
                
                # Skip first request for debugging (can be outlier due to initialization)
                # if req_id == "req_0":
                #     continue  

                # Get metrics from request_output
                metrics = request_output.metrics

                # Calculate per-token ITLs from wall-clock timestamps.
                # Uses gaps between consecutive token observation times (matching
                # Dooly's inter_token_latencies_ms = token_events[i].timestamp -
                # token_events[i-1].timestamp). Gives N-1 values for N tokens.
                # This avoids step_duration attribution errors from vLLM's
                # pipelined engine.step() which includes previous step's output
                # processing in the current step's wall-clock time.
                itls = []
                if req_id in request_token_timestamps:
                    timestamps = request_token_timestamps[req_id]
                    itls = [timestamps[i] - timestamps[i-1]
                            for i in range(1, len(timestamps))]
                elif metrics.num_generation_tokens > 1:
                    # Fallback: average ITL from vLLM internal metrics
                    avg_itl = ((metrics.last_token_ts - metrics.first_token_ts)
                            / (metrics.num_generation_tokens - 1))
                    itls = [avg_itl] * (metrics.num_generation_tokens - 1)

                # if len(outputs) < 3:
                #     print(f"\n[DEBUG] {req_id} ITL: {len(itls)} values, "
                #         f"first 3: {[f'{x*1000:.2f}ms' for x in itls[:3]]}")

                # Use arrival_time from our requests list (ground truth)
                relative_start_time = request_arrival_times[req_id]
                
                # Set time base offset from first completed request
                # Offset = wall_clock_time - engine_time (constant for all requests)
                if time_base_offset is None:
                    time_base_offset = metrics.arrival_time - metrics.queued_ts

                # For THIS request, convert its actual arrival time (from vLLM metrics) to engine time
                # metrics.arrival_time = wall-clock when add_request() was called for THIS request
                # engine_time_arrival = metrics.arrival_time - time_base_offset
                engine_time_arrival = metrics.arrival_time

                # TTFT = time from when THIS request was added to first token (includes queue + prefill)
                ttft_with_queue = metrics.first_token_ts - engine_time_arrival
                # ttft_with_queue = metrics.first_token_ts - metrics.queued_ts
                
                # E2E latency = time from when THIS request was added to last token
                e2e_latency = metrics.last_token_ts - engine_time_arrival

                output = RequestFuncOutput(
                    generated_text=request_output.outputs[0].text if request_output.outputs else "",
                    success=True,
                    latency=e2e_latency,  # E2E latency from arrival to completion
                    output_tokens=metrics.num_generation_tokens,
                    ttft=ttft_with_queue,  # Include queue time for fair comparison
                    itl=itls,
                    tpot=0.0,  # Will be calculated by calculate_metrics
                    prompt_len=len(request_output.prompt_token_ids),
                    error="",
                    start_time=relative_start_time,  # Use relative time, not absolute
                    input_audio_duration=0.0,
                )

                outputs.append(output)
                completed_requests[req_id] = output

                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({
                    'TTFT': f'{output.ttft*1000:.1f}ms',
                    'tokens': output.output_tokens,
                    'steps': step_count
                })

                if verbose:
                    current_time = time.perf_counter() - start_time
                    itl_str = f"itl={output.itl[0]*1000:.1f}ms" if output.itl else "itl=N/A"
                    print(f"[t={current_time:.3f}s] Request {req_id} completed "
                            f"(latency={output.latency*1000:.1f}ms, "
                            f"ttft={output.ttft*1000:.1f}ms, "
                            f"{itl_str}, "
                            f"tokens={output.output_tokens})")

    # Close progress bar
    pbar.close()

    end_time = time.perf_counter()
    duration = end_time - start_time

    return outputs, duration, step_batches

def main():
    parser = argparse.ArgumentParser(
        description="Measure vLLM performance with streaming requests"
    )

    # Model configuration
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-70B",
                       help="Model name")
    parser.add_argument("--backend", type=str, default="FLASHINFER",
                       help="Attention backend (FLASHINFER or FLASH_ATTN)")
    parser.add_argument("--tp", type=int, default=4)

    # Request generation
    parser.add_argument("--load-dataset", type=str,
                       help="Load requests from JSONL file (for exact comparison)")
    parser.add_argument("--use-sharegpt", type=str, metavar="SHAREGPT_PATH",
                       help="Use ShareGPT dataset (path to ShareGPT JSON)")
    parser.add_argument("--use-hf-dataset", type=str, metavar="HF_DATASET_PATH",
                       help=("Use a HuggingFace conversation dataset via vLLM's "
                             "ConversationDataset (e.g. Aeala/ShareGPT_Vicuna_unfiltered)"))
    parser.add_argument("--hf-dataset-split", type=str, default="train",
                       help="HF dataset split (default: train)")
    parser.add_argument("--hf-dataset-subset", type=str, default=None,
                       help="HF dataset subset/config name (optional)")
    parser.add_argument("--num-requests", type=int, default=100,
                       help="Number of requests")
    parser.add_argument("--arrival-rate", type=float, default=5.0,
                       help="Request arrival rate (req/s)")
    parser.add_argument("--input-len", type=int, default=1024,
                       help="Mean input length")
    parser.add_argument("--min-input-len", type=int, default=None,
                       help="Min input length")
    parser.add_argument("--max-input-len", type=int, default=None,
                       help="Max input length")
    parser.add_argument("--output-len", type=int, default=None,
                       help="Mean output length")
    parser.add_argument("--range-ratio", type=float, default=0.0,
                       help="Length variability (0.0=fixed, 1.0=uniform 0 to 2x)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")

    # vLLM engine configuration
    parser.add_argument("--max-model-len", type=int, default=16384,
                       help="Maximum model length")
    parser.add_argument("--max-num-seqs", type=int, default=256,
                       help="Maximum number of sequences")
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192,
                       help="Maximum batched tokens")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                       help="GPU memory utilization")
    parser.add_argument("--enable-chunked-prefill", action="store_true",
                       help="Enable chunked prefill")
    parser.add_argument("--moe-routing-sim", type=str, default="",
                       choices=["", "uniform_random", "normal_routing"],
                       help="If set, replaces real MoE routing with a simulator. "
                            "Sets VLLM_MOE_ROUTING_SIMULATION_STRATEGY before LLM "
                            "construction. 'uniform_random' matches the profiler's "
                            "torch.randn router_logits regime (uniform expert load). "
                            "Output text becomes invalid; latency measurements remain valid.")

    # Output options
    parser.add_argument("--output", type=str, default="vllm_streaming.json",
                       help="Output JSON file")
    parser.add_argument("--export-requests", type=str,
                       help="Export request dataset to JSONL")
    parser.add_argument("--batch-composition-output", type=str,
                       help="Export batch composition to separate JSON file")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose output")
    parser.add_argument("--quantization", type=str, default=None,
                       choices=["fp8", "awq", "gptq"],
                       help="Optional quantization scheme. 'fp8' requires Hopper+. "
                            "With load_format=dummy, vLLM initializes random quantized weights.")

    args = parser.parse_args()

    # Apply MoE routing simulation BEFORE constructing LLM. vLLM reads this env
    # var lazily at FusedMoE router construction (router_factory.py:93), which
    # happens inside LLM(...). Setting it here is in time.
    if args.moe_routing_sim:
        os.environ["VLLM_MOE_ROUTING_SIMULATION_STRATEGY"] = args.moe_routing_sim
        print(f"[moe-routing-sim] VLLM_MOE_ROUTING_SIMULATION_STRATEGY={args.moe_routing_sim} "
              f"-- runtime routing replaced; output text will be invalid; latency is valid.",
              flush=True)

    # Auto-generate output filenames based on dataset parameters if not explicitly provided
    if args.use_sharegpt:
        # ShareGPT dataset: sharegpt-{rate}rps-{num_requests}reqs
        dataset_name = f"sharegpt-{args.arrival_rate:.0f}rps-{args.num_requests}reqs"

        # Auto-set export-requests if not provided
        if not args.export_requests:
            args.export_requests = f"./dataset/{dataset_name}.jsonl"

        # Auto-set output files if they're still defaults
        if args.output == "vllm_streaming.json":
            args.output = f"./vllm_{dataset_name}.json"

        if not args.batch_composition_output:
            args.batch_composition_output = f"./comparison/vllm_batch_{dataset_name}.json"

    elif args.use_hf_dataset:
        # HF conversation dataset: hf-{basename}-{rate}rps-{num_requests}reqs
        hf_basename = args.use_hf_dataset.replace('/', '_')
        dataset_name = f"hf-{hf_basename}-{args.arrival_rate:.0f}rps-{args.num_requests}reqs"

        if not args.export_requests:
            args.export_requests = f"./dataset/{dataset_name}.jsonl"

        if args.output == "vllm_streaming.json":
            args.output = f"./vllm_{dataset_name}.json"

        if not args.batch_composition_output:
            args.batch_composition_output = f"./comparison/vllm_batch_{dataset_name}.json"

    elif not args.export_requests and not args.load_dataset:
        # Synthetic dataset: syn-{rate}rps-uniform-{input_len}-{output_len}
        dataset_name = f"syn-{args.arrival_rate:.0f}rps-uniform-{args.input_len}-{args.output_len or 128}"

        # Auto-set export-requests if not provided
        args.export_requests = f"./dataset/{dataset_name}.jsonl"

        # Auto-set output files if they're still defaults
        if args.output == "vllm_streaming.json":
            args.output = f"./comparison/vllm_{dataset_name}.json"

        if not args.batch_composition_output:
            args.batch_composition_output = f"./comparison/vllm_batch_{dataset_name}.json"

    elif args.load_dataset and args.load_dataset.endswith('.jsonl'):
        # Extract dataset name from load-dataset path
        dataset_name = Path(args.load_dataset).stem

        # Auto-set output files if they're still defaults
        if args.output == "vllm_streaming.json":
            args.output = f"./comparison/vllm_{dataset_name}.json"

        if not args.batch_composition_output:
            args.batch_composition_output = f"./comparison/vllm_batch_{dataset_name}.json"

    elif args.export_requests and args.export_requests.endswith('.jsonl'):
        # Extract dataset name from export-requests path
        dataset_name = Path(args.export_requests).stem

        # Auto-set output files if they're still defaults
        if args.output == "vllm_streaming.json":
            args.output = f"./comparison/vllm_{dataset_name}.json"

        if not args.batch_composition_output:
            args.batch_composition_output = f"./comparison/vllm_batch_{dataset_name}.json"

    print("=" * 80)
    print("vLLM STREAMING BENCHMARK (Arrival Time Simulation)")
    print("=" * 80)
    print(f"Model:              {args.model}")
    print(f"Backend:            {args.backend}")
    print(f"Requests:           {args.num_requests}")
    print(f"Arrival rate:       {args.arrival_rate} req/s")
    print(f"Input length:       {args.input_len} (range_ratio={args.range_ratio})")
    print(f"Output length:      {args.output_len} (range_ratio={args.range_ratio})")
    print(f"Max model length:   {args.max_model_len}")
    print(f"Max seqs:           {args.max_num_seqs}")
    print(f"Max batched tokens: {args.max_num_batched_tokens}")
    print(f"Chunked prefill:    {args.enable_chunked_prefill}")
    print("=" * 80)
    print()
    
    logging.getLogger("vllm.v1.metrics.loggers").setLevel(logging.CRITICAL)

    llm_kwargs = dict(
            model=args.model,
            dtype="bfloat16",
            attention_backend=args.backend,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=args.max_num_batched_tokens,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enable_prefix_caching=False,
            enforce_eager=False,
            load_format="dummy",
            disable_log_stats=False,
            tensor_parallel_size=args.tp,
            attention_config=({"flash_attn_version": 2} if args.backend == 'FLASH_ATTN' else None),
            # Disable async scheduling to ensure each engine.step() synchronously
            # executes the batch it schedules (matching Dooly's behavior).
            async_scheduling=False,
        )
    if args.quantization:
        llm_kwargs["quantization"] = args.quantization
    llm = LLM(**llm_kwargs)

    # Load or generate requests
    if args.load_dataset:
        # Load from pre-generated JSONL file
        print(f"Loading requests from {args.load_dataset}...")

        tokenizer = llm.get_tokenizer()
        requests = []
        sample_requests = []

        with open(args.load_dataset, 'r') as f:
            for line in f:
                entry = json.loads(line)
                prompt = entry['prompt']
                token_ids = tokenizer.encode(prompt)

                # Streaming request tuple
                requests.append((
                    entry['request_id'],
                    token_ids,
                    entry['output_len'],
                    entry['arrival_time'],
                ))

                # SampleRequest for metrics
                sample_requests.append(SampleRequest(
                    prompt=prompt,
                    prompt_len=entry.get('prompt_len', len(token_ids)),
                    expected_output_len=entry['output_len'],
                ))

        print(f"✓ Loaded {len(requests)} requests from dataset")
    elif args.use_sharegpt:
        # Generate from ShareGPT dataset
        print(f"Generating requests from ShareGPT: {args.use_sharegpt}")
        dooly_requests, vllm_requests = generate_sharegpt_requests_from_instance(
            llm=llm,
            num_requests=args.num_requests,
            arrival_rate=args.arrival_rate,
            sharegpt_path=args.use_sharegpt,
            min_input_len=args.min_input_len,
            max_input_len=args.max_input_len,
            custom_output_len=args.output_len,
            seed=args.seed,
        )

        # Export requests if requested
        if args.export_requests:
            from doolysim.tools.vllm_request_generator import export_vllm_dataset_jsonl
            arrival_times = [r.arrival_time for r in dooly_requests]
            export_vllm_dataset_jsonl(vllm_requests, arrival_times, args.export_requests)

        # Convert to format for streaming benchmark
        requests = [
            (req.request_id, req.prompt_token_ids, req.max_tokens, req.arrival_time)
            for req in dooly_requests
        ]

        sample_requests = [
            SampleRequest(
                prompt=vllm_req.prompt,
                prompt_len=vllm_req.prompt_len,
                expected_output_len=vllm_req.expected_output_len,
            )
            for vllm_req in vllm_requests
        ]

        print(f"[REQUESTS] Req_id: {requests[-1][0]} - output len: {requests[-1][2]}")
        print(f"[VLLM_REQUESTS] Req_id: {vllm_requests[-1].request_id} - output len: {vllm_requests[-1].expected_output_len}")
        # exit(0)
    elif args.use_hf_dataset:
        # Generate from a HuggingFace conversation dataset
        print(f"Generating requests from HF dataset: {args.use_hf_dataset}")
        dooly_requests, vllm_requests = generate_hf_conversation_requests_from_instance(
            llm=llm,
            num_requests=args.num_requests,
            arrival_rate=args.arrival_rate,
            dataset_path=args.use_hf_dataset,
            dataset_split=args.hf_dataset_split,
            dataset_subset=args.hf_dataset_subset,
            custom_output_len=args.output_len,
            seed=args.seed,
        )

        if args.export_requests:
            from doolysim.tools.vllm_request_generator import export_vllm_dataset_jsonl
            arrival_times = [r.arrival_time for r in dooly_requests]
            export_vllm_dataset_jsonl(vllm_requests, arrival_times, args.export_requests)

        requests = [
            (req.request_id, req.prompt_token_ids, req.max_tokens, req.arrival_time)
            for req in dooly_requests
        ]
        sample_requests = [
            SampleRequest(
                prompt=vllm_req.prompt,
                prompt_len=vllm_req.prompt_len,
                expected_output_len=vllm_req.expected_output_len,
            )
            for vllm_req in vllm_requests
        ]
    else:
        # Generate requests using RandomDataset (same as dooly_sim)
        print("Generating requests with vLLM RandomDataset...")
        dooly_requests, vllm_requests = generate_vllm_requests_from_instance(
            llm=llm,
            num_requests=args.num_requests,
            arrival_rate=args.arrival_rate,
            input_len=args.input_len,
            output_len=args.output_len,
            range_ratio=args.range_ratio,
            seed=args.seed,
        )

        # Export requests if requested
        if args.export_requests:
            from doolysim.tools.vllm_request_generator import export_vllm_dataset_jsonl
            arrival_times = [r.arrival_time for r in dooly_requests]
            export_vllm_dataset_jsonl(vllm_requests, arrival_times, args.export_requests)

        # Convert to format for streaming benchmark
        requests = [
            (req.request_id, req.prompt_token_ids, req.max_tokens, req.arrival_time)
            for req in dooly_requests
        ]

        sample_requests = [
            SampleRequest(
                prompt=vllm_req.prompt,
                prompt_len=vllm_req.prompt_len,
                expected_output_len=vllm_req.expected_output_len,
            )
            for vllm_req in vllm_requests
        ]
    

    print(f"\n✓ Ready with {len(requests)} requests")
    print(f"  First arrival: {requests[0][3]:.3f}s")
    print(f"  Last arrival:  {requests[-1][3]:.3f}s")
    print()

    # Run streaming benchmark
    print("Running streaming benchmark...")
    outputs, duration, step_batches = run_streaming_benchmark(
        llm=llm,
        warmup=2,
        requests=requests,
        verbose=args.verbose,
    )

    # Sort outputs by start_time to match input_requests order
    # CRITICAL: calculate_metrics assumes outputs[i] corresponds to input_requests[i]
    # But outputs list is in completion order! Must sort by arrival time (start_time)
    outputs_sorted = sorted(outputs, key=lambda x: x.start_time)

    # **PRESERVE our step-based ITL data before calculate_metrics() potentially overwrites it**
    step_based_itls = [output.itl for output in outputs_sorted]

    # print(f"\n[DEBUG] Verifying index alignment after sorting:")
    # print(f"  outputs_sorted[99]: start={outputs_sorted[99].start_time:.3f}s, tokens={outputs_sorted[99].output_tokens}, prompt={outputs_sorted[99].prompt_len}")
    # print(f"  sample_requests[99]: prompt={sample_requests[99].prompt_len}, expected_output={sample_requests[99].expected_output_len}")

    print(f"\n[DEBUG] Checking timestamp ranges:")
    print(f"  Min start_time: {min(o.start_time for o in outputs_sorted):.6f}s")
    print(f"  Max start_time: {max(o.start_time for o in outputs_sorted):.6f}s")
    print(f"  Sample latencies (first 5): {[f'{o.latency:.3f}s' for o in outputs_sorted[:5]]}")
    print(f"  Sample TTFTs (first 5): {[f'{o.ttft:.3f}s' for o in outputs_sorted[:5]]}")

    # Calculate metrics using vLLM's standard function
    print("\nCalculating metrics...")
    bench_metrics, actual_output_lens = calculate_metrics(
        input_requests=sample_requests,
        outputs=outputs_sorted,
        dur_s=duration,
        tokenizer=None,
        selected_percentiles=[50.0, 95.0, 99.0],
        goodput_config_dict={},
    )

    print(f"\n[DEBUG] actual_output_lens from calculate_metrics: {actual_output_lens[-5:]}")
    print(f"[DEBUG] output.output_tokens from outputs list: {[o.output_tokens for o in outputs[-5:]]}")

    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Completed:             {bench_metrics.completed}/{args.num_requests}")
    print(f"Failed:                {bench_metrics.failed}")
    print(f"Duration:              {duration:.2f}s")
    print(f"Request throughput:    {bench_metrics.request_throughput:.2f} req/s")
    print(f"Output throughput:     {bench_metrics.output_throughput:.2f} tok/s")
    print(f"Total token throughput:{bench_metrics.total_token_throughput:.2f} tok/s")
    print()
    print(f"TTFT (Time to First Token):")
    print(f"  Mean:   {bench_metrics.mean_ttft_ms:.2f} ms")
    print(f"  Median: {bench_metrics.median_ttft_ms:.2f} ms")
    print(f"  P95:    {dict(bench_metrics.percentiles_ttft_ms).get(95.0, 0):.2f} ms")
    print(f"  P99:    {dict(bench_metrics.percentiles_ttft_ms).get(99.0, 0):.2f} ms")
    print()
    print(f"TPOT (Time Per Output Token):")
    print(f"  Mean:   {bench_metrics.mean_tpot_ms:.2f} ms")
    print(f"  Median: {bench_metrics.median_tpot_ms:.2f} ms")
    print(f"  P95:    {dict(bench_metrics.percentiles_tpot_ms).get(95.0, 0):.2f} ms")
    print(f"  P99:    {dict(bench_metrics.percentiles_tpot_ms).get(99.0, 0):.2f} ms")
    print()
    print(f"ITL (Inter-Token Latency):")
    print(f"  Mean:   {bench_metrics.mean_itl_ms:.2f} ms")
    print(f"  Median: {bench_metrics.median_itl_ms:.2f} ms")
    print()
    print(f"E2E Latency:")
    print(f"  Mean:   {bench_metrics.mean_e2el_ms:.2f} ms")
    print(f"  Median: {bench_metrics.median_e2el_ms:.2f} ms")
    print(f"  P95:    {dict(bench_metrics.percentiles_e2el_ms).get(95.0, 0):.2f} ms")
    print(f"  P99:    {dict(bench_metrics.percentiles_e2el_ms).get(99.0, 0):.2f} ms")
    print("=" * 80)

    # Save results in vLLM-compatible format
    from datetime import datetime
    output_data = {
        'date': datetime.now().strftime("%Y%m%d-%H%M%S"),
        'backend': 'vllm_streaming',
        'model_id': args.model,
        'num_prompts': args.num_requests,
        'arrival_rate': args.arrival_rate,
        'input_len': args.input_len,
        'output_len': args.output_len,
        'range_ratio': args.range_ratio,
        'duration': duration,
        'completed': bench_metrics.completed,
        'failed': bench_metrics.failed,
        'request_throughput': bench_metrics.request_throughput,
        'output_throughput': bench_metrics.output_throughput,
        'total_throughput': bench_metrics.total_token_throughput,
        'mean_ttft_ms': bench_metrics.mean_ttft_ms,
        'median_ttft_ms': bench_metrics.median_ttft_ms,
        'p95_ttft_ms': dict(bench_metrics.percentiles_ttft_ms).get(95.0, 0),
        'p99_ttft_ms': dict(bench_metrics.percentiles_ttft_ms).get(99.0, 0),
        'mean_tpot_ms': bench_metrics.mean_tpot_ms,
        'median_tpot_ms': bench_metrics.median_tpot_ms,
        'p95_tpot_ms': dict(bench_metrics.percentiles_tpot_ms).get(95.0, 0),
        'p99_tpot_ms': dict(bench_metrics.percentiles_tpot_ms).get(99.0, 0),
        'mean_itl_ms': bench_metrics.mean_itl_ms,
        'median_itl_ms': bench_metrics.median_itl_ms,
        'mean_e2el_ms': bench_metrics.mean_e2el_ms,
        'median_e2el_ms': bench_metrics.median_e2el_ms,
        'p95_e2el_ms': dict(bench_metrics.percentiles_e2el_ms).get(95.0, 0),
        'p99_e2el_ms': dict(bench_metrics.percentiles_e2el_ms).get(99.0, 0),
        'input_lens': [req.prompt_len for req in sample_requests],
        'output_lens': actual_output_lens,
        'ttfts': [output.ttft for output in outputs_sorted],
        'itls': step_based_itls,  # **Use our preserved step-based ITL data, not outputs_sorted**
        'start_times': [output.start_time for output in outputs_sorted],

        # Batch information for each request
        # 'batch_info': [dict(step_batch) for step_batch in step_batches],
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Results saved to {args.output}")

    # Export batch composition to separate file if requested
    if args.batch_composition_output:
        batch_composition_data = []
        for step_batch in step_batches:
            batch_composition_data.append({
                "step_id": step_batch["step_id"],
                "sim_time_ms": round(step_batch["sim_time_ms"], 2),
                "elapsed_time_ms": round(step_batch["elapsed_time_ms"], 2),  # **Add elapsed_time_ms**
                "batch_size": step_batch["batch_size"],
                "scheduled_req_ids": step_batch["scheduled_req_ids"],
                "num_scheduled_new": step_batch["num_scheduled_new"],
                "num_scheduled_cached": step_batch["num_scheduled_cached"],
                "total_tokens_scheduled": step_batch["total_tokens_scheduled"],
                "free_blocks": step_batch["free_blocks"],
                "waiting_count": step_batch["waiting_count"],
                "running_count": step_batch["running_count"],
                "finished_req_ids": step_batch["finished_req_ids"],
                "preempted_req_ids": step_batch["preempted_req_ids"],
            })

        with open(args.batch_composition_output, 'w') as f:
            json.dump(batch_composition_data, f, indent=2)

        print(f"✓ Batch composition saved to {args.batch_composition_output}")


if __name__ == "__main__":
    main()
