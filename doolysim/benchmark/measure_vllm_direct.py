import re
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

# vLLM annotates each forward pass on the GPU as
#   execute_context_<n_prefill_seqs>(<n_prefill_tokens>)_generation_<n_decode_seqs>(<n_decode_tokens>)
# A step is a prefill iff <n_prefill_tokens> > 0; otherwise it's a decode.
# Capturing groups: (n_prefill_seqs, n_prefill_tokens, n_decode_seqs, n_decode_tokens)
_EXECUTE_CONTEXT_RE = re.compile(
    r'execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)'
)

try:
    import torch
    from vllm import LLM, SamplingParams
    from vllm.benchmarks.serve import calculate_metrics, BenchmarkMetrics
    from vllm.benchmarks.datasets import SampleRequest
    from vllm.benchmarks.lib.endpoint_request_func import RequestFuncOutput
    VLLM_AVAILABLE = True
except ImportError:
    print("ERROR: vLLM not available. Install with: pip install vllm")
    sys.exit(1)


def convert_to_request_func_output(vllm_output, start_time: float) -> RequestFuncOutput:
    """Convert vLLM RequestOutput to RequestFuncOutput format for calculate_metrics."""
    metrics = vllm_output.metrics

    # Handle dummy load format where metrics might be None
    if metrics is None:
        # Return dummy output
        num_output_tokens = len(vllm_output.outputs[0].token_ids) if vllm_output.outputs else 1
        return RequestFuncOutput(
            generated_text=vllm_output.outputs[0].text if vllm_output.outputs else "",
            success=True,
            latency=0.02,  # 20ms dummy latency
            output_tokens=num_output_tokens,
            ttft=0.01,  # 10ms dummy TTFT
            itl=[0.005] * num_output_tokens,  # 5ms dummy ITL
            tpot=0.0,
            prompt_len=len(vllm_output.prompt_token_ids),
            error="",
            start_time=start_time,
            input_audio_duration=0.0,
        )

    # Calculate ITLs from metrics
    itls = []
    if metrics.num_generation_tokens > 1:
        # Average ITL across all tokens except first
        avg_itl = (metrics.last_token_ts - metrics.first_token_ts) / metrics.num_generation_tokens
        itls = [avg_itl] * metrics.num_generation_tokens

    # Calculate total latency (E2E from queued to last token)
    # Note: vLLM metrics use monotonic clock, so we use the deltas not absolute values
    # latency = time from request queued until last token generated
    latency = metrics.last_token_ts - metrics.queued_ts

    return RequestFuncOutput(
        generated_text=vllm_output.outputs[0].text if vllm_output.outputs else "",
        success=True,
        latency=latency,
        output_tokens=metrics.num_generation_tokens,
        ttft=metrics.first_token_latency,
        itl=itls,
        tpot=0.0,  # Will be calculated by calculate_metrics
        prompt_len=len(vllm_output.prompt_token_ids),
        error="",
        start_time=start_time,
        input_audio_duration=0.0,
    )

def measure_batch_request(
    llm: LLM,
    batch_config: Dict,
    warmup: int = 3,
    iterations: int = 10,
    profile_dir: str = './vllm_profiles'
) -> Dict:
    prompt_lens = batch_config['prompt_lens']
    num_prefills = len(prompt_lens)
    output_lens = batch_config['output_lens']

    # Create prompts using explicit token IDs to ensure exact length
    # This guarantees we get exactly the desired token count
    prompt_token_ids_list = []
    for target_len in prompt_lens:
        # Create token IDs from 1 to target_len (avoiding special tokens like 0)
        token_ids = list(range(1, target_len + 1))
        prompt_token_ids_list.append(token_ids)

    print(f"  Created {len(prompt_token_ids_list)} prompts with exact token counts:")
    for i, (target_len, token_ids, out_len) in enumerate(zip(prompt_lens, prompt_token_ids_list, output_lens)):
        print(f"    Prompt {i+1}: {len(token_ids)} tokens (target: {target_len}), output_len: {out_len}")

    # Use max output_len for generation
    max_output_len = max(output_lens)
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_output_len,
        ignore_eos=True,
    )
    print(f"  Using max_tokens={max_output_len}")

    # Warmup - run WITHOUT profiling
    print(f"  Warming up ({warmup} iterations)...")
    for i in range(warmup):
        from vllm.inputs import TokensPrompt
        prompts = [TokensPrompt(prompt_token_ids=token_ids) for token_ids in prompt_token_ids_list]
        llm.generate(prompts, sampling_params=sampling_params)
        torch.cuda.synchronize()

    # Measure with profiling
    print(f"  Profiling prefill + decode phases...")
    latencies = []

    # Start profiling BEFORE the first generate call
    torch.cuda.synchronize()
    llm.start_profile()

    # Run iterations
    per_request_ttfts = []  # Track TTFT for each request
    per_request_itls = []   # Track ITL for each request
    all_request_func_outputs = []  # For calculate_metrics compatibility

    benchmark_start_time = time.perf_counter()
    first_request_arrival_time = None  # Track the first arrival time to make times relative

    for i in range(iterations):
        # Pass token IDs as TextTokensPrompt objects
        from vllm.inputs import TokensPrompt
        prompts = [TokensPrompt(prompt_token_ids=token_ids) for token_ids in prompt_token_ids_list]
        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params=sampling_params)
        torch.cuda.synchronize()
        end = time.perf_counter()

        latency_ms = (end - start) * 1000
        latencies.append(latency_ms)

        # Extract per-request metrics from vLLM
        print(f"    Per-request metrics:")

        for j, output in enumerate(outputs):
            # Check if metrics are available
            if output.metrics is None:
                print(f"      Request {j+1}: No metrics available (using dummy load)")
                # Create dummy metrics for testing with dummy load
                per_request_ttfts.append(0.01)  # 10ms dummy TTFT
                per_request_itls.append(0.005)  # 5ms dummy ITL
                # Still convert to RequestFuncOutput for compatibility
                if first_request_arrival_time is None:
                    first_request_arrival_time = 0.0
                request_func_output = convert_to_request_func_output(output, 0.0)
                all_request_func_outputs.append(request_func_output)
                continue

            # Track first arrival time to make all times relative
            if first_request_arrival_time is None:
                first_request_arrival_time = output.metrics.arrival_time

            # Verify actual token count
            actual_prompt_len = len(output.prompt_token_ids)
            print(f"      Request {j+1}: {actual_prompt_len} tokens")
            print(f"         num_generation_tokens: {output.metrics.num_generation_tokens}")
            print(f"         arrival_time: {output.metrics.arrival_time:.3f} s")
            print(f"         queued_ts: {output.metrics.queued_ts:.3f} s")
            print(f"         scheduled_ts: {output.metrics.scheduled_ts:.3f} s")
            print(f"         first_token_ts: {output.metrics.first_token_ts:.3f} s")
            print(f"         last_token_ts: {output.metrics.last_token_ts:.3f} s")
            print(f"         first_token_latency: {output.metrics.first_token_latency * 1000.0} ms")
            print(f"         inter-token latency: {(output.metrics.last_token_ts - output.metrics.first_token_ts) / output.metrics.num_generation_tokens * 1000.0} ms")

            per_request_ttfts.append(output.metrics.first_token_latency)
            per_request_itls.append(((output.metrics.last_token_ts - output.metrics.first_token_ts) / output.metrics.num_generation_tokens))

            # Convert to RequestFuncOutput for calculate_metrics (use relative time)
            relative_start_time = output.metrics.arrival_time - first_request_arrival_time
            request_func_output = convert_to_request_func_output(output, relative_start_time)
            all_request_func_outputs.append(request_func_output)

    benchmark_end_time = time.perf_counter()
    benchmark_duration_s = benchmark_end_time - benchmark_start_time

    llm.stop_profile()

    # Create SampleRequest objects for calculate_metrics
    sample_requests = [
        SampleRequest(
            prompt="",  # We use token IDs directly
            prompt_len=len(token_ids),
            expected_output_len=output_lens[idx] if idx < len(output_lens) else 1,
        )
        for idx, token_ids in enumerate(prompt_token_ids_list)
    ]

    # Debug: Print some sample data
    if all_request_func_outputs:
        print(f"  Debug: Total outputs: {len(all_request_func_outputs)}")
        print(f"  Debug: First output - start_time: {all_request_func_outputs[0].start_time:.6f}s, latency: {all_request_func_outputs[0].latency:.6f}s")
        print(f"  Debug: Last output - start_time: {all_request_func_outputs[-1].start_time:.6f}s, latency: {all_request_func_outputs[-1].latency:.6f}s")
        print(f"  Debug: Benchmark duration: {benchmark_duration_s:.6f}s")

    # Find the generated trace file. vLLM ≥0.17 emits filenames like
    # 'dp0_pp0_tp0_dcp0_ep0_rank0.<id>.pt.trace.json'; older versions used
    # 'rank0.<id>.pt.trace.json'. Match both.
    trace_files = sorted(Path(profile_dir).glob("*rank*.pt.trace.json"))
    gpu_times = {}
    trace_file_path = None
    if trace_files:
        latest_trace = trace_files[-1]
        print(f"  ✓ Trace saved to: {latest_trace}")
        trace_file_path = str(latest_trace)

        # Parse trace to extract gpu_user_annotation times
        gpu_times = extract_gpu_times_from_trace(trace_file_path)

    # Use GPU timing data from execute_model_stats
    execute_model_stats = gpu_times.get('execute_model_details', [])

    # Helper for the per-stat-list median that we use in several places.
    def _med(stats, key):
        return float(np.median([s[key] for s in stats])) if stats else None

    # Default field values (overwritten below if we have any execute_model_stats).
    prefill_latency = decode_latency = first_decode_latency = None
    n_prefill_steps = n_decode_steps = 0

    if execute_model_stats:
        # Sort chronologically so "first decode after prefill" is well-defined.
        sorted_stats = sorted(execute_model_stats, key=lambda s: s.get('ts', 0))

        # Partition by prefill/decode using the annotation classification.
        # If `is_prefill` is None (regex didn't match — older vLLM annotation
        # format), fall back to the historic "median of all" behavior so old
        # traces still produce a usable number.
        prefill_stats = [s for s in sorted_stats if s.get('is_prefill') is True]
        decode_stats  = [s for s in sorted_stats if s.get('is_prefill') is False]
        unknown_stats = [s for s in sorted_stats if s.get('is_prefill') is None]

        n_prefill_steps = len(prefill_stats)
        n_decode_steps  = len(decode_stats)

        prefill_latency = _med(prefill_stats, 'computation_kernel_time_ms')
        decode_latency  = _med(decode_stats,  'computation_kernel_time_ms')

        # The first decode step in chronological order — its KV cache equals
        # exactly the prefill prompt length (no growth yet). This is the value
        # that matches db_predictor's "decode at KV=K exactly" prediction.
        if decode_stats:
            first_decode_latency = (decode_stats[0]['computation_kernel_time_ms']
                                    + decode_stats[0]['communication_kernel_time_ms'])

        # Pick the workload-appropriate aggregate.
        # - Decode-mode trace (any decode steps present): use the first decode
        #   so the comparison to db_predictor lands at the configured KV.
        # - Prefill-only trace: median of prefills (typically just one per iter).
        # - Unknown classification: fall back to the historic "median of all"
        #   so old traces still produce a usable number.
        if decode_stats:
            # Communication latency for the same first-decode step.
            comm_latency = decode_stats[0]['communication_kernel_time_ms']
            gpu_latency  = decode_stats[0]['computation_kernel_time_ms']
        elif prefill_stats:
            gpu_latency  = prefill_latency
            comm_latency = _med(prefill_stats, 'communication_kernel_time_ms') or 0.0
        else:
            # Unknown — fall back to old behavior across all stats.
            gpu_latency  = _med(sorted_stats, 'computation_kernel_time_ms') or 0.0
            comm_latency = _med(sorted_stats, 'communication_kernel_time_ms') or 0.0

        # Comm-aggregate fields stay as medians across all events (these are
        # diagnostic rather than the headline number).
        all_reduce_latency      = float(np.median([s['communication']['all_reduce']['total_ms']  for s in execute_model_stats]))
        all_reduce_count        = float(np.median([s['communication']['all_reduce']['count']     for s in execute_model_stats]))
        per_all_reduce_latency  = float(np.median([s['communication']['all_reduce']['median_ms'] for s in execute_model_stats]))
        all_gather_latency      = float(np.median([s['communication']['all_gather']['total_ms']  for s in execute_model_stats]))
        all_gather_count        = float(np.median([s['communication']['all_gather']['count']     for s in execute_model_stats]))
        per_all_gather_latency  = float(np.median([s['communication']['all_gather']['median_ms'] for s in execute_model_stats]))
        total_latency = gpu_latency + comm_latency
    else:
        # Fallback to end-to-end measurement
        total_latency = np.mean(latencies)
        gpu_latency = total_latency
        comm_latency = 0.0
        all_reduce_latency = 0.0
        all_reduce_count = 0
        per_all_reduce_latency = 0.0
        all_gather_latency = 0.0
        all_gather_count = 0
        per_all_gather_latency = 0.0

    # Return results matching batch_predictor.py format
    result = {
        'total_latency_ms': total_latency,
        'gpu_latency_ms': gpu_latency,
        'comm_latency_ms': comm_latency,
        'all_reduce_latency_ms': all_reduce_latency,
        'all_reduce_count': all_reduce_count,
        'per_all_reduce_latency_ms': per_all_reduce_latency,
        'all_gather_latency_ms': all_gather_latency,
        'all_gather_count': all_gather_count,
        'per_all_gather_latency_ms': per_all_gather_latency,
        'batch_size': num_prefills,
        'total_tokens': sum(prompt_lens),
        # Counts come from the trace classification — these reflect what
        # actually executed on the GPU, not the input config's request lists.
        'num_prefill_steps': n_prefill_steps,
        'num_decode_steps':  n_decode_steps,
        # Per-workload latencies. `prefill_latency_ms` is the median across
        # all prefill steps in this trace; `decode_latency_ms` is the median
        # across all decode steps; `first_decode_latency_ms` is the very
        # first decode step (KV cache = configured prompt length, no growth)
        # — this is the value that matches db_predictor's prediction at
        # `kv_cache=K` exactly.
        'prefill_latency_ms':      prefill_latency,
        'decode_latency_ms':       decode_latency,
        'first_decode_latency_ms': first_decode_latency,
        # Backward-compat fields (kept so older readers don't break, but the
        # values now reflect the trace, not the config).
        'num_prefill_requests': n_prefill_steps,
        'num_decode_requests':  n_decode_steps,
    }

    if result['total_tokens'] > 0:
        result['ms_per_token'] = total_latency / result['total_tokens']

    # Add detailed trace analysis including module-to-kernel mappings
    if gpu_times:
        result['trace_analysis'] = gpu_times

    return result


def extract_module_id(module_name: str) -> int:
    """Extract module ID from names like 'nn.Module: RMSNorm_132'"""
    try:
        if 'nn.Module:' in module_name:
            module_part = module_name.split('nn.Module:')[1].strip()
            if '_' in module_part:
                _, module_id_str = module_part.rsplit('_', 1)
                return int(module_id_str)
    except:
        pass
    return 0

def extract_module_type(module_name: str) -> str:
    """Extract module type from names like 'nn.Module: RMSNorm_132' or 'COMMUNICATION: AllReduce_0 (Layer 5)'"""
    try:
        if 'COMMUNICATION:' in module_name:
            # Extract communication type like "AllReduce_0" or "AllGather_1"
            if 'AllReduce' in module_name:
                if '_0' in module_name:
                    return 'AllReduce_0'
                elif '_1' in module_name:
                    return 'AllReduce_1'
                else:
                    return 'AllReduce'
            elif 'AllGather' in module_name:
                if '_0' in module_name:
                    return 'AllGather_0'
                elif '_1' in module_name:
                    return 'AllGather_1'
                else:
                    return 'AllGather'
            else:
                return 'Communication'
        elif 'nn.Module:' in module_name:
            module_part = module_name.split('nn.Module:')[1].strip()
            if '_' in module_part:
                module_type, _ = module_part.rsplit('_', 1)
                return module_type
    except:
        pass
    return 'Unknown'

def analyze_module_to_kernel_from_trace(trace_path: str) -> Dict:
    """Analyze module-to-kernel mapping from PyTorch trace."""

    with open(trace_path, 'r') as f:
        trace = json.load(f)

    # Collect events by type
    modules = []
    cpu_operations = []
    kernel_events = []
    external_id_to_kernel = {}

    for event in trace.get('traceEvents', []):
        cat = event.get('cat', '')
        name = event.get('name', '')
        ts = event.get('ts', 0)
        dur = event.get('dur', 0)
        external_id = event.get('args', {}).get('External id')

        if cat == 'python_function':
            if 'nn.Module' in name:
                modules.append({
                    'name': name,
                    'cat': cat,
                    'ts': ts,
                    'dur': dur,
                    'end_ts': ts + dur,
                    'external_id': external_id,
                    'leaf_module': True,
                    'is_layer': 'DecoderLayer_' in name,
                    'layer_num': -1
                })

        elif cat == 'cpu_op':
            cpu_operations.append({
                'name': name,
                'cat': cat,
                'ts': ts,
                'dur': dur,
                'end_ts': ts + dur,
                'external_id': external_id
            })
        elif cat == 'kernel':
            kernel_event = {
                'name': name,
                'cat': cat,
                'ts': ts,
                'dur': dur,
                'end_ts': ts + dur,
                'external_id': external_id
            }
            kernel_events.append(kernel_event)

            if external_id is not None:
                if external_id not in external_id_to_kernel:
                    external_id_to_kernel[external_id] = []
                external_id_to_kernel[external_id].append(kernel_event)

    # First pass: Assign layer numbers to all modules based on their parent DecoderLayer
    for module in modules:
        if 'DecoderLayer_' in module['name']:
            # Extract layer number from DecoderLayer
            layer_num = int(module['name'].split('DecoderLayer_')[-1].split('/')[0])
            module_start = module['ts']
            module_end = module['end_ts']

            # Mark all child modules with this layer number
            for child_module in modules:
                if (child_module['ts'] >= module_start and
                    child_module['end_ts'] <= module_end and
                    child_module['name'] != module['name']):
                    child_module['layer_num'] = layer_num

    # Second pass: Find leaf modules
    leaf_modules = []
    for module in modules:
        module_start = module['ts']
        module_end = module['end_ts']

        for child_module in modules:
            if (child_module['ts'] >= module_start and
                child_module['end_ts'] <= module_end and
                child_module['name'] != module['name']):
                module['leaf_module'] = False
                break

        if module['leaf_module']:
            leaf_modules.append(module)

    # Extract module-to-kernel mappings with communication isolation
    module_to_kernel = {}
    comm_counts_per_layer = {}

    for leaf_module in leaf_modules:
        for cpu_op in cpu_operations:
            if cpu_op['ts'] >= leaf_module['ts'] and cpu_op['end_ts'] <= leaf_module['end_ts']:
                if cpu_op['external_id'] is not None:
                    kernels = external_id_to_kernel.get(cpu_op['external_id'])
                    if kernels:
                        # Check if this is a communication operation
                        cpu_op_lower = cpu_op['name'].lower()
                        is_communication = any(comm_op in cpu_op_lower for comm_op in ['all_reduce', 'allreduce', 'all_gather', 'allgather'])

                        if is_communication:
                            # Determine communication type
                            if 'all_reduce' in cpu_op_lower or 'allreduce' in cpu_op_lower:
                                comm_type = 'AllReduce'
                            elif 'all_gather' in cpu_op_lower or 'allgather' in cpu_op_lower:
                                comm_type = 'AllGather'
                            else:
                                comm_type = 'Unknown'

                            # Track occurrence count for this communication type in this layer
                            layer_comm_key = (leaf_module['layer_num'], comm_type)
                            if layer_comm_key not in comm_counts_per_layer:
                                comm_counts_per_layer[layer_comm_key] = 0

                            # Create unique communication module name with occurrence count
                            occurrence = comm_counts_per_layer[layer_comm_key]
                            comm_module_name = f"COMMUNICATION: {comm_type}_{occurrence} (Layer {leaf_module['layer_num']})"
                            comm_counts_per_layer[layer_comm_key] += 1

                            if comm_module_name not in module_to_kernel:
                                module_to_kernel[comm_module_name] = []

                            # Handle all kernels for communication
                            for kernel in kernels:
                                module_to_kernel[comm_module_name].append({
                                    'layer_num': leaf_module['layer_num'],
                                    'cpu_op': cpu_op['name'],
                                    'kernel': kernel['name'],
                                    'kernel_dur_us': kernel['dur']
                                })
                        else:
                            # Regular computation module
                            if leaf_module['name'] not in module_to_kernel:
                                module_to_kernel[leaf_module['name']] = []

                            # Handle all kernels with the same External_id
                            for kernel in kernels:
                                module_to_kernel[leaf_module['name']].append({
                                    'layer_num': leaf_module['layer_num'],
                                    'cpu_op': cpu_op['name'],
                                    'kernel': kernel['name'],
                                    'kernel_dur_us': kernel['dur']
                                })

    # Calculate total latencies per module
    module_total_latencies = {}
    for module, ops in module_to_kernel.items():
        total_latency_us = sum(op['kernel_dur_us'] for op in ops)
        kernel_count = len(ops)

        module_total_latencies[module] = {
            'layer_num': ops[0]['layer_num'] if ops else -1,
            'total_latency_us': total_latency_us,
            'kernel_count': kernel_count,
        }

    return module_total_latencies

def compute_median_latencies_by_position(module_total_latencies: Dict) -> Dict:
    """Group modules by their execution order position within each layer and compute median latency."""

    # Group modules by layer first (including layer -1)
    layers = {}
    non_layer_modules = []
    for module_name, data in module_total_latencies.items():
        layer_num = data['layer_num']
        if layer_num == -1:  # Non-layer modules (embeddings, logits, etc.)
            non_layer_modules.append((module_name, data))
        else:
            if layer_num not in layers:
                layers[layer_num] = []
            layers[layer_num].append((module_name, data))

    # Sort modules within each layer by module_id to get execution order
    for layer_num in layers:
        layers[layer_num].sort(key=lambda x: extract_module_id(x[0]))

    # Group by position across all layers
    position_groups = {}
    for layer_num, modules_in_layer in layers.items():
        # Track occurrence count for each module type within this layer
        module_type_counts = {}

        for position, (module_name, data) in enumerate(modules_in_layer):
            module_type = extract_module_type(module_name)

            # Count occurrence of this module type in this layer
            if module_type not in module_type_counts:
                module_type_counts[module_type] = 0
            occurrence = module_type_counts[module_type]
            module_type_counts[module_type] += 1

            # Create position key based on module type and occurrence order
            position_key = f"{module_type}_occurrence_{occurrence}"

            if position_key not in position_groups:
                position_groups[position_key] = []

            position_groups[position_key].append(data['total_latency_us'])

    # Compute median for each position
    result = {}
    for position_key, latencies in position_groups.items():
        result[position_key] = {
            'median_latency_us': np.median(latencies),
            'median_latency_ms': np.median(latencies) / 1000.0,
            'mean_latency_us': np.mean(latencies),
            'num_layers': len(latencies),
        }

    # Add non-layer modules (layer -1) as individual entries
    for module_name, data in non_layer_modules:
        module_type = extract_module_type(module_name)
        position_key = f"{module_type}_non_layer"
        result[position_key] = {
            'median_latency_us': data['total_latency_us'],
            'median_latency_ms': data['total_latency_us'] / 1000.0,
            'mean_latency_us': data['total_latency_us'],
            'num_layers': 1,
        }

    return result

def print_module_analysis_summary(module_analysis: Dict):
    """Print a formatted summary of the module analysis results."""
    if not module_analysis or 'module_position_analysis' not in module_analysis:
        print("  No module analysis data available")
        return

    position_analysis = module_analysis['module_position_analysis']

    if not position_analysis:
        print("  No module position analysis found")
        return

    print("\n  📊 Module-to-Kernel Analysis:")
    print("  " + "="*60)

    # Keep original order, separate communication, computation, and non-layer modules
    communication_ops = []
    computation_ops = []
    non_layer_ops = []

    for pos, data in position_analysis.items():
        if 'AllReduce' in pos or 'AllGather' in pos:
            communication_ops.append((pos, data))
        elif '_non_layer' in pos:
            non_layer_ops.append((pos, data))
        else:
            computation_ops.append((pos, data))

    # Print communication operations
    if communication_ops:
        print("\n  🔄 Communication Operations:")
        for pos, data in communication_ops:
            median_ms = data['median_latency_ms']
            num_layers = data['num_layers']
            print(f"    {pos:<35} {median_ms:>7.3f} ms  ({num_layers} layers)")

    # Print computation operations
    if computation_ops:
        print("\n  ⚡ Computation Operations:")
        for pos, data in computation_ops:
            median_ms = data['median_latency_ms']
            num_layers = data['num_layers']
            print(f"    {pos:<35} {median_ms:>7.3f} ms  ({num_layers} layers)")

    # Print non-layer operations
    if non_layer_ops:
        print("\n  🎯 Non-Layer Operations (embeddings, logits, etc.):")
        for pos, data in non_layer_ops:
            median_ms = data['median_latency_ms']
            print(f"    {pos:<35} {median_ms:>7.3f} ms")

    # Calculate totals
    total_comm_time = sum(data['median_latency_ms'] for _, data in communication_ops)
    total_comp_time = sum(data['median_latency_ms'] for _, data in computation_ops)
    total_non_layer_time = sum(data['median_latency_ms'] for _, data in non_layer_ops)
    total_time = total_comm_time + total_comp_time + total_non_layer_time

    print("\n  📈 Summary:")
    print(f"    Communication time: {total_comm_time:>7.3f} ms ({total_comm_time/total_time*100:.1f}%)")
    print(f"    Computation time:   {total_comp_time:>7.3f} ms ({total_comp_time/total_time*100:.1f}%)")
    if total_non_layer_time > 0:
        print(f"    Non-layer time:     {total_non_layer_time:>7.3f} ms ({total_non_layer_time/total_time*100:.1f}%)")
    print(f"    Total time:         {total_time:>7.3f} ms")
    print("  " + "="*60)

def extract_gpu_times_from_trace(trace_path: str) -> Dict:
    return analyze_execute_model_from_trace(trace_path)

def analyze_execute_model_from_trace(trace_path: str) -> Dict:
    """Analyze execute_model events and communication operations within them.

    Returns detailed statistics about kernel execution and communication patterns
    within execute_model calls for comparison with Dooly simulator.
    """
    try:
        with open(trace_path, 'r') as f:
            trace = json.load(f)

        # Build maps of events
        kernel_events_by_ts = {}
        kernel_name_to_dur = {}
        execute_model_events = []
        execute_context_events = []
        comm_kernel_events = []

        for event in trace.get('traceEvents', []):
            cat = event.get('cat', '')
            name = event.get('name', '')
            ts = event.get('ts', 0)

            if cat == 'kernel':
                if ts not in kernel_events_by_ts:
                    kernel_events_by_ts[ts] = []
                    
                kernel_name_to_dur[name] = event.get('dur', 0)
                kernel_events_by_ts[ts].append(event)
                
                if 'all_gather' in name.lower() or 'allgather' in name.lower() or 'all_reduce' in name.lower() or 'allreduce' in name.lower():
                    comm_kernel_events.append({
                        'name': name,
                        'ts': ts,
                        'dur': event.get('dur', 0),
                        'type': 'all_gather' if 'all_gather' in name.lower() or 'allgather' in name.lower() else 'all_reduce'
                    })
                    
            elif cat == 'gpu_user_annotation':
                if 'execute_context' in name:
                    # Parse out prefill vs decode classification from the
                    # annotation name (vLLM emits this format on every
                    # forward pass — see _EXECUTE_CONTEXT_RE definition).
                    m = _EXECUTE_CONTEXT_RE.search(name)
                    if m:
                        n_prefill_seqs   = int(m.group(1))
                        n_prefill_tokens = int(m.group(2))
                        n_decode_seqs    = int(m.group(3))
                        n_decode_tokens  = int(m.group(4))
                        is_prefill       = n_prefill_tokens > 0
                    else:
                        n_prefill_seqs = n_prefill_tokens = 0
                        n_decode_seqs  = n_decode_tokens  = 0
                        is_prefill     = None  # unknown
                    execute_context_events.append({
                        'name': name,
                        'ts': ts,
                        'dur': event.get('dur', 0),
                        'type': cat,
                        'is_prefill':       is_prefill,
                        'n_prefill_seqs':   n_prefill_seqs,
                        'n_prefill_tokens': n_prefill_tokens,
                        'n_decode_seqs':    n_decode_seqs,
                        'n_decode_tokens':  n_decode_tokens,
                    })
                    
                    
        # Analyze each execute_model event
        execute_model_stats = []
        
        for i, exec_model in enumerate(execute_context_events):
            start_ts = exec_model['ts']
            end_ts = start_ts + exec_model['dur']

            # Find all kernels within this execute_model
            kernels_within = []
            comm_kernels = {'all_gather': [], 'all_reduce': [], 'others': []}

            for ts in kernel_events_by_ts:
                if start_ts <= ts <= end_ts:
                    for kernel in kernel_events_by_ts[ts]:
                        kernel_name = kernel.get('name', '').lower()
                        kernel_dur_us = kernel.get('dur', 0)

                        kernels_within.append({
                            'name': kernel.get('name', ''),
                            'dur_us': kernel_dur_us
                        })

                        # Check if it's a communication kernel
                        if 'nccl' in kernel_name:
                            if 'allgather' in kernel_name:
                                comm_kernels['all_gather'].append(kernel_dur_us)
                            elif 'allreduce' in kernel_name:
                                comm_kernels['all_reduce'].append(kernel_dur_us)
                        elif 'cross_device_reduce' in kernel_name:
                            comm_kernels['all_reduce'].append(kernel_dur_us)
                            
            exec_model['kernels_within'] = kernels_within
            exec_model['comm_kernels'] = comm_kernels

        execute_context_events.sort(key=lambda x: x['ts'])  # Ensure execute_model events are in chronological order
        for i, exec_model in enumerate(execute_context_events):
            start_ts = exec_model['ts']
            end_ts = start_ts + exec_model['dur']
            comm_kernels = exec_model.get('comm_kernels', {'all_gather': [], 'all_reduce': []})
            kernels_within = exec_model.get('kernels_within', [])
            
            for comm_kernel in comm_kernel_events:
                if comm_kernel['ts'] >= end_ts and comm_kernel['ts'] < execute_context_events[i+1]['ts'] if i+1 < len(execute_context_events) else True:
                    if 'all_gather' in comm_kernel['name'].lower() or 'allgather' in comm_kernel['name'].lower():
                        comm_kernels['all_gather'].append(comm_kernel['dur'])
                        comm_kernel_events.remove(comm_kernel)  # Remove to avoid double counting
 

            # Calculate statistics
            total_kernel_time_us = sum(k['dur_us'] for k in kernels_within)
            total_comm_kernel_time_us = (sum(comm_kernels['all_gather']) +
                                        sum(comm_kernels['all_reduce']))

            exec_stats = {
                'duration_ms': exec_model['dur'] / 1000,
                'total_kernel_time_ms': total_kernel_time_us / 1000,
                'computation_kernel_time_ms': (total_kernel_time_us - total_comm_kernel_time_us) / 1000,
                'communication_kernel_time_ms': total_comm_kernel_time_us / 1000,
                'kernel_count': len(kernels_within),
                'is_prefill':       exec_model.get('is_prefill'),
                'n_prefill_seqs':   exec_model.get('n_prefill_seqs', 0),
                'n_prefill_tokens': exec_model.get('n_prefill_tokens', 0),
                'n_decode_seqs':    exec_model.get('n_decode_seqs', 0),
                'n_decode_tokens':  exec_model.get('n_decode_tokens', 0),
                'ts':               exec_model['ts'],
                'communication': {
                    'all_gather': {
                        'count': len(comm_kernels['all_gather']),
                        'total_ms': sum(comm_kernels['all_gather']) / 1000,
                        'median_ms': np.median(comm_kernels['all_gather']) / 1000 if comm_kernels['all_gather'] else 0,
                    },
                    'all_reduce': {
                        'count': len(comm_kernels['all_reduce']),
                        'total_ms': sum(comm_kernels['all_reduce']) / 1000,
                        'median_ms': np.median(comm_kernels['all_reduce']) / 1000 if comm_kernels['all_reduce'] else 0,
                    },
                    'comm_kernel_ratio': total_comm_kernel_time_us / total_kernel_time_us if total_kernel_time_us > 0 else 0,
                }
            }

            execute_model_stats.append(exec_stats)

        # Add module-to-kernel analysis
        try:
            module_total_latencies = analyze_module_to_kernel_from_trace(trace_path)
            module_position_analysis = compute_median_latencies_by_position(module_total_latencies)
        except Exception as e:
            print(f"  ⚠ Warning: Module analysis failed: {e}")
            module_total_latencies = {}
            module_position_analysis = {}

        # Calculate aggregate statistics
        result = {
            'execute_model_count': len(execute_model_stats),
            'execute_model_details': execute_model_stats,
            'module_latency': module_position_analysis
        }

        return result

    except Exception as e:
        print(f"  ⚠ Error analyzing execute_model: {e}")
        import traceback
        traceback.print_exc()
        return {}


def load_batch_config_from_json(json_path: str) -> Tuple[str, Dict]:
    """Load batch configuration from JSON file compatible with batch_predictor.py.

    Args:
        json_path: Path to JSON configuration file

    Returns:
        Tuple of (test_name, config_dict) compatible with measure_batch_request
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Extract prompt lengths and output lengths from the JSON format
    prompt_lens = []
    output_lens = []
    
    
    # Process prefill requests
    if 'prefill_requests' in data:
        for req in data['prefill_requests']:
            tokens = req.get('tokens', req.get('num_tokens', 0))
            # For prefill requests, we'll generate a small number of tokens
            # This simulates the prefill -> first decode token transition
            prompt_lens.append(tokens)
            output_lens.append(1)  # Default to 1 token output for prefill

    # Process decode requests
    # For decode requests in the JSON, we interpret them as requests that have
    # already completed prefill and need to generate additional tokens
    if 'decode_requests' in data:
        for req in data['decode_requests']:
            if isinstance(req, dict):
                kv_cache = req.get('kv_cache', req.get('kv_cache_size', 0))
            else:
                kv_cache = int(req)
            # For decode requests, the "prompt" is the KV cache size
            # and we generate a configurable number of output tokens
            prompt_lens.append(kv_cache)
            # Use a default output length or from config
            output_lens.append(data.get('default_output_len', 32))

    # Create test name from description or filename
    test_name = data.get('description', Path(json_path).stem)

    # Build config in the format expected by measure_batch_request
    config = {
        'prompt_lens': prompt_lens,
        'output_lens': output_lens,
        'description': f"Loaded from {json_path}: {len(prompt_lens)} requests",
    }

    # Add metadata if present
    if 'model' in data:
        config['model'] = data['model']
    if 'backend' in data:
        config['backend'] = data['backend']
    if 'tensor_parallel_size' in data:
        config['tensor_parallel_size'] = data['tensor_parallel_size']

    return test_name, config

def main():
    """Main measurement entry point."""

    import argparse
    parser = argparse.ArgumentParser(description="Measure vLLM latencies directly")
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.1-8B',
                       help='Model to benchmark')
    parser.add_argument('--warmup', type=int, default=3,
                       help='Number of warmup iterations')
    parser.add_argument('--iterations', type=int, default=1,
                       help='Number of measurement iterations')
    parser.add_argument('--output', type=str, default='vllm_measurements.json',
                       help='Output JSON file')
    parser.add_argument('--gpu-memory', type=float, default=0.9,
                       help='GPU memory utilization')
    parser.add_argument('--batch-config', type=str,
                       help='Load batch configuration from JSON file (compatible with batch_predictor.py)')
    parser.add_argument('--tensor-parallel-size', type=int, default=2,
                       help='Tensor parallel size')
    parser.add_argument('--attention-backend', type=str, default='FLASHINFER',
                       help='Attention backend (FLASHINFER or FLASH_ATTN)')
    parser.add_argument('--max-model-len', type=int, default=8192,
                       help='Maximum model sequence length')
    parser.add_argument('--max-num-seqs', type=int, default=32,
                       help='Maximum number of sequences')
    parser.add_argument('--out-dir', type=str, default='./result/')
    parser.add_argument('--quantization', type=str, default=None, choices=["fp8", "awq", "gptq"],
                       help="Optional quantization scheme. 'fp8' requires Hopper+. With load_format=dummy, vLLM initializes random quantized weights.")
    args = parser.parse_args()

    print("=" * 80)
    print("DIRECT VLLM LATENCY MEASUREMENT")
    print("=" * 80)
    print(f"Model:               {args.model}")
    print(f"Warmup:              {args.warmup}")
    print(f"Iterations:          {args.iterations}")
    print(f"Output:              {args.output}")
    print(f"Tensor Parallel:     {args.tensor_parallel_size}")
    print(f"Attention Backend:   {args.attention_backend}")
    print(f"Max Model Len:       {args.max_model_len}")
    if args.batch_config:
        print(f"Batch Config File:   {args.batch_config}")
    print("=" * 80)
    print()

    # Initialize vLLM
    print("Initializing vLLM (this may take a minute)...")
    test_name, config = load_batch_config_from_json(args.batch_config)

    # Load original config data for JSON output
    with open(args.batch_config, 'r') as f:
        original_config = json.load(f)

    model = config.get('model', args.model)
    backend = config.get('backend', args.attention_backend)
    tp = config.get('tensor_parallel_size', args.tensor_parallel_size)

    llm_kwargs = dict(
        model=model,
        dtype="bfloat16",  # Ensure consistent dtype with Dooly
        attention_backend=backend,
        tensor_parallel_size=tp,
        gpu_memory_utilization=args.gpu_memory,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_model_len,
        load_format="dummy",
        enforce_eager=False,
        enable_prefix_caching=False,
        attention_config=({"flash_attn_version": 2} if args.attention_backend == 'FLASH_ATTN' else None),
        # compilation_config=CompilationConfig(
        #     mode=CompilationMode.NONE,
        #     cudagraph_mode=CUDAGraphMode.FULL
        # ),
        profiler_config={
            'profiler': 'torch',
            'torch_profiler_dir': './vllm_profiles',
            'torch_profiler_use_gzip': False,
            'torch_profiler_with_stack': True,
            'torch_profiler_record_shapes': True,
        },
    )
    if args.quantization:
        llm_kwargs["quantization"] = args.quantization
    llm = LLM(**llm_kwargs)

    result = measure_batch_request(
        llm=llm,
        batch_config=config,
        warmup=args.warmup,
        iterations=args.iterations,
        profile_dir='./vllm_profiles'
    )

    # Create output directory if it doesn't exist
    import os
    os.makedirs(args.out_dir, exist_ok=True)

    # Create JSON output in the same format as batch_predictor.py
    output_data = {
        'model': model,
        'backend': backend,
        'tensor_parallel_size': tp,
        'topology': 'mew1',  # Default topology
        'batch_config': {
            'prefill_requests': original_config.get('prefill_requests', []),
            'decode_requests': original_config.get('decode_requests', []),
        },
        'results': result
    }

    output_file = os.path.join(args.out_dir, args.output)
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
