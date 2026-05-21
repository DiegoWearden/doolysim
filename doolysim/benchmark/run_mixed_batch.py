#!/usr/bin/env python3
"""
Run a single CONTROLLED mixed-batch step on vLLM and produce a kineto trace.

The standard `measure_vllm_direct.py` path uses `llm.generate(...)` which submits
all prompts at once and lets vLLM's scheduler decide composition. That can't
produce a true "fresh prefill batched with ongoing decodes" step. This driver
controls execution via `engine.step()` directly:

  Phase 1: pre-warm. Submit each `decode_requests[i]` as a prompt of length
           `kv_cache_i` and step the engine until that request has produced
           its first output token (i.e., it's now in decode phase, KV at
           target).
  Phase 2: inject. Submit each `prefill_requests[i]` as a fresh prompt of
           length `tokens_i`.
  Phase 3: profile exactly ONE engine.step() inside torch.profiler. That step
           is the actual mixed batch.
  Phase 4: verify the captured step really is mixed (scheduler may have
           decided otherwise — abort with diagnostics if so).
  Phase 5: parse the trace, aggregate per-module GPU kernel time, save JSON.

Usage:
  python -m doolysim.benchmark.run_mixed_batch \\
      --config batch_config.json \\
      --out-dir results/mixed_batch/ \\
      [--enforce-eager] [--moe-routing-sim uniform_random]

batch_config.json schema:
  {
    "model": "Qwen/Qwen3-30B-A3B",
    "backend": "FLASHINFER",
    "tensor_parallel_size": 1,
    "prefill_requests": [{"tokens": 65,  "kv_cache": 0}, ...],
    "decode_requests":  [{"tokens": 1,   "kv_cache": 512}, ...]
  }
"""

import argparse
import json
import os
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", required=True, help="Path to batch_config JSON")
    p.add_argument("--out-dir", default="./mixed_batch_results")
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--max-num-batched-tokens", type=int, default=8192)
    p.add_argument("--max-num-seqs", type=int, default=32)
    p.add_argument("--gpu-memory", type=float, default=0.9)
    p.add_argument("--enforce-eager", action="store_true", default=True,
                   help="Run in eager mode (default True). Required for visible "
                        "per-kernel timing in the kineto trace; CUDA-graph capture "
                        "would inline kernels and obscure module annotations.")
    p.add_argument("--no-enforce-eager", dest="enforce_eager", action="store_false",
                   help="Disable enforce_eager (use compiled mode + CUDA graphs).")
    p.add_argument("--moe-routing-sim", default="",
                   choices=["", "uniform_random", "normal_routing"],
                   help="If set, sets VLLM_MOE_ROUTING_SIMULATION_STRATEGY before "
                        "vLLM constructs the FusedMoE router.")
    return p.parse_args()


def aggregate_kernel_time_by_module(trace_path: str) -> dict:
    """Group GPU kernel time by the leaf cpu_op that launched each kernel,
    using External-id correlation (NOT timestamp containment).

    Why External id and not ts: kernel events are timestamped on the GPU
    clock; cpu_ops on the CPU clock. With async dispatch, a kernel often
    completes after its launching cpu_op closes, so ts-based containment
    misattributes or drops it. External id is set when the kernel is
    launched and is a deterministic kernel→cpu_op link, independent of
    clock alignment. Same algorithm measure_vllm_direct.py uses.

    Returns {op_name: total_kernel_time_ms} sorted descending.
    """
    with open(trace_path) as f:
        trace = json.load(f)
    events = trace.get("traceEvents", [])

    # External id -> innermost (smallest-span) cpu_op. PyTorch occasionally
    # propagates the same id from a parent cpu_op to a child that just
    # delegates; the leaf is the smallest-dur match.
    cpu_by_ext = {}
    for ev in events:
        if ev.get("cat") != "cpu_op":
            continue
        ext = ev.get("args", {}).get("External id")
        if ext is None:
            continue
        dur = ev.get("dur", 0)
        if ext not in cpu_by_ext or dur < cpu_by_ext[ext].get("dur", 1e30):
            cpu_by_ext[ext] = ev

    bucket_us = defaultdict(float)
    bucket_count = defaultdict(int)
    no_ext_us = 0.0
    for ev in events:
        if ev.get("cat") != "kernel":
            continue
        dur = ev.get("dur", 0)
        ext = ev.get("args", {}).get("External id")
        leaf = cpu_by_ext.get(ext)
        if leaf is None:
            no_ext_us += dur
        else:
            bucket_us[leaf["name"]] += dur
            bucket_count[leaf["name"]] += 1

    out = {k: v / 1000.0 for k, v in bucket_us.items()}
    if no_ext_us > 0:
        out["__no_external_id__"] = no_ext_us / 1000.0
    return dict(sorted(out.items(), key=lambda x: -x[1]))


def aggregate_kernel_time_by_module_class(trace_path: str) -> dict:
    """Roll cpu_op-attributed kernel time up to nn.Module class level.

    Uses External id for kernel → cpu_op (clock-safe), then CPU-clock ts
    containment to assign each cpu_op to the innermost containing
    `nn.Module:` annotation (CPU-clock vs CPU-clock — no async issue).
    Strips trailing `_N` instance index so all 48 layer instances of a
    class collapse into one row. Requires the trace to have been captured
    with `with_stack=True` / `with_modules=True`.
    """
    with open(trace_path) as f:
        trace = json.load(f)
    events = trace.get("traceEvents", [])

    cpu_ops = []     # (name, ts, end, ext_id)
    modules = []     # (class_name, ts, end, span)
    ext_id_to_kernels_us = defaultdict(float)

    for ev in events:
        cat = ev.get("cat", "")
        name = ev.get("name", "")
        ts = ev.get("ts")
        dur = ev.get("dur")
        if ts is None or dur is None:
            continue
        ext = ev.get("args", {}).get("External id")
        if cat == "cpu_op":
            cpu_ops.append((name, ts, ts + dur, ext))
        elif cat == "kernel" and ext is not None:
            ext_id_to_kernels_us[ext] += dur
        elif cat == "python_function" and name.startswith("nn.Module: "):
            cls = name[len("nn.Module: "):].strip().rsplit("_", 1)[0]
            modules.append((cls, ts, ts + dur, dur))

    modules.sort(key=lambda m: m[1])

    bucket_us = defaultdict(float)
    outside_us = 0.0
    for op_name, op_ts, op_end, ext in cpu_ops:
        if ext is None:
            continue
        kernel_us = ext_id_to_kernels_us.get(ext, 0)
        if kernel_us == 0:
            continue
        best_cls, best_span = None, float("inf")
        for cls, m_ts, m_end, m_span in modules:
            if m_ts > op_ts:
                break
            if m_end >= op_end and m_span < best_span:
                best_cls, best_span = cls, m_span
        if best_cls is None:
            outside_us += kernel_us
        else:
            bucket_us[best_cls] += kernel_us

    out = {k: v / 1000.0 for k, v in bucket_us.items()}
    if outside_us > 0:
        out["__outside_any_module__"] = outside_us / 1000.0
    return dict(sorted(out.items(), key=lambda x: -x[1]))


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Env vars must be set BEFORE vllm import (FusedMoE router factory reads
    # VLLM_MOE_ROUTING_SIMULATION_STRATEGY at module-construction time inside
    # LLM init; we are still upstream of that here).
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    if args.moe_routing_sim:
        os.environ["VLLM_MOE_ROUTING_SIMULATION_STRATEGY"] = args.moe_routing_sim
        print(f"[mixed_batch] VLLM_MOE_ROUTING_SIMULATION_STRATEGY={args.moe_routing_sim}",
              flush=True)

    print("[mixed_batch] Importing vLLM...", flush=True)
    import torch
    from torch.profiler import profile, ProfilerActivity
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    with open(args.config) as f:
        cfg = json.load(f)
    pr_list = cfg.get("prefill_requests", [])
    dr_list = cfg.get("decode_requests", [])
    if not pr_list or not dr_list:
        print("[mixed_batch] ERROR: batch_config must have BOTH prefill_requests "
              "AND decode_requests for mixed-batch mode")
        return 1

    model = cfg["model"]
    backend = cfg.get("backend", "FLASHINFER")
    tp = int(cfg.get("tensor_parallel_size", 1))

    print(f"[mixed_batch] Target composition:")
    print(f"  prefill_requests: {[(p['tokens'], p.get('kv_cache', 0)) for p in pr_list]}")
    print(f"  decode_requests:  {[(d.get('tokens', 1), d['kv_cache']) for d in dr_list]}")
    print(f"[mixed_batch] LLM: model={model}, backend={backend}, tp={tp}, "
          f"enforce_eager={args.enforce_eager}", flush=True)

    t0 = time.perf_counter()
    llm = LLM(
        model=model,
        dtype="bfloat16",
        load_format="dummy",
        attention_backend=backend,
        attention_config=({"flash_attn_version": 2} if backend == "FLASH_ATTN" else None),
        tensor_parallel_size=tp,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory,
        enforce_eager=args.enforce_eager,
        enable_chunked_prefill=True,
        trust_remote_code=True,
        seed=0,
    )
    print(f"[mixed_batch] LLM ready in {time.perf_counter()-t0:.1f}s", flush=True)

    engine = llm.llm_engine
    suffix = uuid.uuid4().hex[:6]

    # ===== Phase 1: pre-warm decode_requests =====
    decode_ids = []
    for i, dr in enumerate(dr_list):
        kv = int(dr["kv_cache"])
        # Arbitrary fresh tokens of length kv. Token IDs in [100, 100+kv).
        token_ids = list(range(100, 100 + kv))
        rid = f"warm_decode_{i}_{suffix}"
        decode_ids.append(rid)
        engine.add_request(
            request_id=rid,
            prompt=TokensPrompt(prompt_token_ids=token_ids),
            params=SamplingParams(max_tokens=64, ignore_eos=True, temperature=0.0),
        )
    print(f"\n[mixed_batch] Phase 1: pre-warming {len(decode_ids)} decode_targets...",
          flush=True)

    in_decode = set()
    n_steps = 0
    max_steps = 100
    while len(in_decode) < len(decode_ids) and n_steps < max_steps:
        outs = engine.step()
        n_steps += 1
        for o in outs:
            rid = o.request_id
            if rid in decode_ids and o.outputs and o.outputs[0].token_ids:
                in_decode.add(rid)
    if len(in_decode) != len(decode_ids):
        print(f"[mixed_batch] ERROR: pre-warm hit max_steps={max_steps}; "
              f"only {len(in_decode)}/{len(decode_ids)} reached decode phase")
        return 1
    print(f"[mixed_batch] Pre-warm done in {n_steps} steps; all decode_targets in decode phase",
          flush=True)

    # ===== Phase 2: inject prefill_requests =====
    prefill_ids = []
    for i, pr in enumerate(pr_list):
        tokens = int(pr["tokens"])
        token_ids = list(range(200, 200 + tokens))
        rid = f"target_prefill_{i}_{suffix}"
        prefill_ids.append(rid)
        engine.add_request(
            request_id=rid,
            prompt=TokensPrompt(prompt_token_ids=token_ids),
            params=SamplingParams(max_tokens=1, temperature=0.0),
        )
    print(f"\n[mixed_batch] Phase 2: injected {len(prefill_ids)} prefill_targets",
          flush=True)

    # ===== Phase 3: monkey-patch scheduler.schedule to capture composition =====
    # Same pattern as measure_vllm_streaming.py:230. Try the v1 path first.
    try:
        scheduler = engine.engine_core.engine_core.scheduler
    except AttributeError:
        scheduler = engine.engine_core.scheduler
    original_schedule = scheduler.schedule
    captured = [None]

    def capture_schedule():
        captured[0] = original_schedule()
        return captured[0]

    scheduler.schedule = capture_schedule

    # ===== Phase 4: profile exactly one step =====
    print("[mixed_batch] Phase 4: running 1 profiled engine.step()...", flush=True)
    trace_path = out_dir / f"mixed_batch_trace_{suffix}.json"

    torch.cuda.synchronize()
    t_step = time.perf_counter()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
        with_modules=True,
    ) as prof:
        _ = engine.step()
    torch.cuda.synchronize()
    step_dur_ms = (time.perf_counter() - t_step) * 1000.0

    scheduler.schedule = original_schedule
    prof.export_chrome_trace(str(trace_path))
    print(f"[mixed_batch] Trace exported to {trace_path}")
    print(f"[mixed_batch] Wall step duration: {step_dur_ms:.2f} ms")

    # ===== Phase 5: verify composition =====
    sched = captured[0]
    if sched is None:
        print("[mixed_batch] ERROR: scheduler.schedule was not invoked during step")
        return 1
    n_new = len(sched.scheduled_new_reqs)
    n_cached = sched.scheduled_cached_reqs.num_reqs
    total_tokens = sched.total_num_scheduled_tokens
    new_ids = [r.req_id for r in sched.scheduled_new_reqs]

    print(f"\n[mixed_batch] Captured step composition:")
    print(f"  scheduled_new_reqs : {n_new}  -> {new_ids}")
    print(f"  scheduled_cached   : {n_cached}")
    print(f"  total_tokens       : {total_tokens}")
    print(f"  num_tokens by req  :")
    for rid, nt in sched.num_scheduled_tokens.items():
        # vLLM may append a UUID suffix to our request_ids; match by prefix.
        kind = ("TARGET PREFILL" if any(rid.startswith(p) for p in prefill_ids) else
                "warmup decode"   if any(rid.startswith(d) for d in decode_ids)  else
                "other")
        print(f"    {rid:<48}: {nt} tokens   ({kind})")

    if n_new == 0 or n_cached == 0:
        print(f"\n[mixed_batch] ✗ Step is NOT mixed (need new>0 AND cached>0). "
              f"Aborting analysis.")
        return 1
    print(f"\n[mixed_batch] ✓ Step is MIXED ({n_new} new × {n_cached} cached, "
          f"{total_tokens} total tokens)")

    # ===== Phase 6: per-cpu_op (leaf) GPU kernel time, via External-id correlation =====
    leaf_op_times = aggregate_kernel_time_by_module(str(trace_path))
    no_ext_ms = leaf_op_times.get("__no_external_id__", 0.0)
    total_gpu_ms = sum(v for k, v in leaf_op_times.items() if not k.startswith("__"))

    print(f"\n[mixed_batch] Per-cpu_op GPU kernel time (External id attribution):")
    print(f"  {'leaf cpu_op':<55} {'gpu_ms':>10}")
    print(f"  {'-'*55} {'-'*10}")
    for name, ms in list(leaf_op_times.items())[:25]:
        print(f"  {name:<55} {ms:>10.4f}")
    print(f"  {'-'*55} {'-'*10}")
    print(f"  {'TOTAL (attributed)':<55} {total_gpu_ms:>10.4f}")
    if no_ext_ms > 0:
        print(f"  {'(kernels w/o External id)':<55} {no_ext_ms:>10.4f}")

    # ===== Phase 6b: nn.Module-class roll-up (requires with_stack=True) =====
    module_class_times = aggregate_kernel_time_by_module_class(str(trace_path))
    if module_class_times:
        outside_ms = module_class_times.get("__outside_any_module__", 0.0)
        in_module_ms = sum(v for k, v in module_class_times.items() if not k.startswith("__"))
        print(f"\n[mixed_batch] nn.Module-class roll-up (top 20):")
        print(f"  {'module class':<55} {'gpu_ms':>10}")
        print(f"  {'-'*55} {'-'*10}")
        for name, ms in list(module_class_times.items())[:20]:
            print(f"  {name:<55} {ms:>10.4f}")
        print(f"  {'-'*55} {'-'*10}")
        print(f"  {'TOTAL (in-module)':<55} {in_module_ms:>10.4f}")
        if outside_ms > 0:
            print(f"  {'(outside any nn.Module)':<55} {outside_ms:>10.4f}")

    # ===== Phase 7: save summary =====
    summary = {
        "config": cfg,
        "args": {
            "enforce_eager": args.enforce_eager,
            "moe_routing_sim": args.moe_routing_sim,
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
        },
        "captured_step": {
            "n_new_reqs": n_new,
            "n_cached_reqs": n_cached,
            "total_tokens": total_tokens,
            "new_req_ids": new_ids,
            "wall_step_ms": step_dur_ms,
            "num_tokens_by_req": dict(sched.num_scheduled_tokens),
        },
        "trace_path": str(trace_path),
        "leaf_cpu_op_gpu_ms": leaf_op_times,
        "module_class_gpu_ms": module_class_times,
        "total_gpu_ms_attributed": total_gpu_ms,
        "no_external_id_gpu_ms": no_ext_ms,
    }
    summary_path = out_dir / f"mixed_batch_summary_{suffix}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[mixed_batch] Summary saved to {summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
