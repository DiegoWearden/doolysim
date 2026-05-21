"""Direct-from-DB latency predictor.

Drop-in replacement for `simulator.benchmark.batch_predictor` that bypasses
the trained estimator entirely. For each operation profiled for the
(model, backend) pair, we look up the median per-call latency at the
matching shape directly in `results`, then sum count * per_call across all
ops to get a forward-pass total.

Usage:
    python db_predictor.py \\
        --batch-config /path/to/batch.json \\
        --db-path     /path/to/profile.db \\
        [--json-output /path/to/out.json]

Or for a sweep over multiple batch-configs:
    python db_predictor.py \\
        --batch-config-glob '/path/to/longctx_sweep/*.json' \\
        --db-path     /path/to/profile.db \\
        --output-dir  /path/to/results/

Lookup rules:
  - Context-split ops (Attention etc.): match (is_prefill, num_tokens, kv_cache_size)
  - Non-context ops:                    match num_tokens
  - If exact num_tokens not present: use the nearest profiled num_tokens
  - If exact kv_cache_size not present: use the nearest within the same is_prefill bucket
"""
import argparse
import glob
import json
import os
import sqlite3
import sys
from pathlib import Path


def get_ops(conn, model, backend, tp=1):
    return conn.execute("""
        SELECT s.operation_name, s.signature_hash, mo.count
        FROM model_operations mo
        JOIN signatures      s ON s.operation_id = mo.operation_id
        JOIN profiled_models pm ON pm.model_id   = mo.model_id
        WHERE pm.model_name=? AND pm.backend=? AND pm.tp_degree=?
    """, (model, backend, tp)).fetchall()


def is_context_op(conn, sig):
    return conn.execute(
        "SELECT 1 FROM results WHERE signature_hash=? AND is_context=1 LIMIT 1",
        (sig,)
    ).fetchone() is not None


def lookup_latency(conn, sig, is_ctx, num_tokens, kv_cache_size, is_prefill):
    """Return (latency_ms, used_num_tokens, used_kv) for the closest available match."""
    if is_ctx:
        rows = conn.execute("""
            SELECT num_tokens, kv_cache_size, AVG(median_latency_ms)
            FROM results
            WHERE signature_hash=? AND is_prefill=?
            GROUP BY num_tokens, kv_cache_size
        """, (sig, 1 if is_prefill else 0)).fetchall()
    else:
        rows = conn.execute("""
            SELECT num_tokens, AVG(median_latency_ms)
            FROM results
            WHERE signature_hash=?
            GROUP BY num_tokens
        """, (sig,)).fetchall()
        rows = [(nt, 0, lat) for nt, lat in rows]
    if not rows:
        return None, None, None
    # nearest match by (|Δnum_tokens|, |Δkv|)
    best = min(rows, key=lambda r: (abs(r[0] - num_tokens), abs(r[1] - kv_cache_size)))
    return best[2], best[0], best[1]


def predict_one(conn, batch_cfg):
    model   = batch_cfg["model"]
    backend = batch_cfg["backend"]
    tp      = batch_cfg.get("tensor_parallel_size", 1)
    prefill = batch_cfg.get("prefill_requests", [])
    decode  = batch_cfg.get("decode_requests", [])

    # For a prefill-only batch with one request, num_tokens == request size.
    # For mixed/multi-request, total is the union of per-request token counts.
    prefill_tokens = sum(r["tokens"] for r in prefill)
    decode_tokens  = len(decode)
    total_tokens   = prefill_tokens + decode_tokens
    is_prefill_phase = prefill_tokens > 0

    # KV-cache context for the attention call. For mixed batches we'd ideally
    # invoke the attention kernel per-request, but the DB stores per-step
    # measurements at the granularity of (num_tokens, kv_cache). Approximate
    # with the maximum kv_cache across requests, which dominates attention cost.
    if prefill:
        kv_cache_for_attn = max(r.get("kv_cache", 0) for r in prefill)
    elif decode:
        kv_cache_for_attn = max(decode) if isinstance(decode[0], int) else max(d.get("kv_cache", 0) for d in decode)
    else:
        kv_cache_for_attn = 0

    ops = get_ops(conn, model, backend, tp)
    if not ops:
        raise RuntimeError(f"No model_operations rows for ({model}, {backend}, tp={tp})")

    breakdown = []
    total_ms = 0.0
    for op_name, sig, count in ops:
        is_ctx = is_context_op(conn, sig)
        lat, used_n, used_kv = lookup_latency(
            conn, sig, is_ctx,
            num_tokens=total_tokens,
            kv_cache_size=kv_cache_for_attn,
            is_prefill=is_prefill_phase,
        )
        if lat is None:
            continue
        contrib = count * lat
        total_ms += contrib
        breakdown.append({
            "operation": f"{op_name}_{sig[:8]}",
            "type": "attention" if is_ctx else "computation",
            "pred_latency_ms": lat,
            "count": count,
            "total_latency_ms": contrib,
            "num_tokens": used_n,
            "kv_cache_size": used_kv,
            "exact_match": (used_n == total_tokens and used_kv == kv_cache_for_attn),
        })

    breakdown.sort(key=lambda r: -r["total_latency_ms"])
    return {
        "total_latency_ms": total_ms,
        "gpu_latency_ms":   total_ms,
        "comm_latency_ms":  0.0,
        "batch_size": len(prefill) + len(decode),
        "total_tokens": total_tokens,
        "num_prefill_requests": len(prefill),
        "num_decode_requests": len(decode),
        "operation_breakdown": breakdown,
    }


def run_one_config(db_path, batch_config_path, out_path=None, verbose=False):
    with open(batch_config_path) as f:
        batch_cfg = json.load(f)

    conn = sqlite3.connect(db_path)
    try:
        results = predict_one(conn, batch_cfg)
    finally:
        conn.close()

    output = {
        "model": batch_cfg["model"],
        "backend": batch_cfg["backend"],
        "tensor_parallel_size": batch_cfg.get("tensor_parallel_size", 1),
        "topology": batch_cfg.get("topology", "mew1"),
        "batch_config": {
            "prefill_requests": batch_cfg.get("prefill_requests", []),
            "decode_requests":  batch_cfg.get("decode_requests", []),
        },
        "results": results,
    }

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)

    if verbose:
        print(f"\n{'='*60}\n{Path(batch_config_path).name}\n{'='*60}")
        print(f"Total tokens: {results['total_tokens']}")
        print(f"PREDICTED LATENCY: {results['total_latency_ms']:.2f} ms")
        approx_count = sum(1 for op in results["operation_breakdown"] if not op["exact_match"])
        if approx_count:
            print(f"  ({approx_count}/{len(results['operation_breakdown'])} ops used nearest-neighbor lookup)")

    return output


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db-path", required=True)
    ap.add_argument("--batch-config", help="Single batch config JSON.")
    ap.add_argument("--batch-config-glob", help="Glob over multiple batch configs (e.g. './sweep/*.json').")
    ap.add_argument("--json-output", help="Output path for single config.")
    ap.add_argument("--output-dir",  help="Output directory for sweep mode.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.batch_config:
        out = args.json_output or None
        result = run_one_config(args.db_path, args.batch_config, out, verbose=True)
        if not out:
            print(json.dumps(result["results"], indent=2))
        return 0

    if args.batch_config_glob:
        if not args.output_dir:
            ap.error("--output-dir required with --batch-config-glob")
        paths = sorted(glob.glob(args.batch_config_glob))
        if not paths:
            print(f"No files matched: {args.batch_config_glob}", file=sys.stderr)
            return 1
        for p in paths:
            stem = Path(p).stem
            out = os.path.join(args.output_dir, f"dbpred_{stem}.json")
            run_one_config(args.db_path, p, out, verbose=args.verbose)
        print(f"\nDone — {len(paths)} configs written to {args.output_dir}")
        return 0

    ap.error("Provide --batch-config or --batch-config-glob")


if __name__ == "__main__":
    sys.exit(main())
