# DoolySim

A discrete-event simulator for vLLM LLM-inference serving.

`DoolySim` predicts end-to-end serving performance (latency, throughput, per-request TPOT/TTFT) by emulating vLLM's scheduler with learned latency models. 
It consumes a profiling database produced by [`DoolyProf`](https://github.com/dooly-project/doolyprof) and uses regression models to predict each step's latency without running the model on a GPU.

The scheduler emulation uses real vLLM scheduler code under the hood, so the batching decisions match production vLLM exactly.

## Installation

```bash
git clone https://github.com/dooly-project/doolysim.git
cd doolysim
pip install -r requirements.txt
```

## Configuration

`DoolySim` reads two paths repeatedly; set them once as environment variables to avoid passing them on every CLI call:

```bash
export DOOLYSIM_DB_PATH=/path/to/dooly_data.db        # profiling database
export DOOLYSIM_CACHE_DIR=/path/to/prediction_cache/  # regression cache
```

Every CLI also accepts `--db-path` and `--cache-dir` explicitly, which override the env vars.

## Quick start

### 1. Train models, generate prediction cache

```bash
python -m doolysim.estimator \
    --model meta-llama/Llama-3.1-8B \
    --backend FLASHINFER \
    --tp-size 1 \
    --topology DGX \
    --cache-dir ./cache/tp1/llama-3.1-8b \
    --train-comm
```

A profiling database may hold several deployments of one model (different TP degree, model dtype or KV-cache dtype). `--tp-size`, `--dtype` and `--kv-cache-dtype` select which one to train on; they default to all rows, bf16 and `auto` respectively, mirroring the profiler's defaults.

This trains Random Forest regression models from the profiled data, generates predictions across common input ranges, and writes both `.pkl` (for retraining) and `.csv` (for instant simulation loading) artifacts into `--cache-dir`.

### 2. Generate workload
If you want to run the simulator on a real conversational workload, regenerate from the openchat ShareGPT-4 dataset:

```bash
python -m simulator.tools.generate_openchat_sharegpt4_dataset \
    --output example_workloads/openchat-1rps.jsonl \
    --num-requests 100 \
    --arrival-rate 1.0 \
    --tokenizer meta-llama/Llama-3.1-8B \
    --max-input-len 2048 \
    --max-output-len 512
```

The script downloads the openchat-sharegpt4 dataset from Hugging Face, tokenizes each conversation, samples Poisson arrivals at `--arrival-rate` requests/sec, and writes a JSONL file with `{request_id, prompt, prompt_len, output_len, arrival_time}` per line.

A second generator, `simulator.tools.generate_sharegpt_dataset`, accepts a local ShareGPT JSON file via `--sharegpt-path` with the same output schema.


### 3. Run a simulation

```bash
python -m doolysim.cli \
    --model meta-llama/Llama-3.1-8B \
    --backend FLASHINFER \
    --load-dataset dataset.jsonl
```

The simulator prints per-request latency, throughput, and aggregate metrics.

Useful flags:
- `--dtype` / `--kv-cache-dtype`: model dtype and KV-cache dtype of the deployment (vLLM literals, default `auto`). They configure the emulated engine and select the matching profiled rows, so an fp8-KV-cache deployment uses the attention latencies profiled with an fp8 cache
- `--use-vllm-dataset`: use vLLM's RandomDataset for reproducible workloads
- `--load-dataset PATH`: replay a saved JSONL request dataset
- `--export-json PATH`: emit results in vLLM's benchmark format for comparison
- `--show-per-request`: show per-request latency breakdown

## ns-3 network backend (experimental)

`ns3_backend/` holds a standalone ns-3 scenario that simulates one collective
(ring AllReduce, ReduceScatter, AllGather, or a single transfer) over a
configurable accelerator fabric and reports its completion time, so that
collective-communication latency can come from a packet-level network model
instead of profiled measurements. It builds without GPUs:

```bash
ns3_backend/setup_ns3.sh      # minimal ns-3.49 into ~/ns-3-dev
ns3_backend/build.sh          # ns3_backend/build/dooly-collective
python -m pytest tests/test_ns3_analytical.py tests/test_ns3_ring.py -q
```

See [ns3_backend/README.md](./ns3_backend/README.md) for the topology and
result schemas and what the model does and does not capture.

## Repository layout

```
doolysim/
└── doolysim/             # python package (import as `doolysim.X`)
    ├── core/             # DoolySim main loop + predictor
    ├── emulator/         # fake vLLM components for CPU-only scheduling
    ├── benchmark/        # measurement + simulation runners
    ├── analysis/         # result summarization
    ├── tools/            # dataset generation + alt latency predictor
    ├── cli.py            # `python -m doolysim.cli`
    ├── estimator.py      # `python -m doolysim.estimator`
    ├── network/          # ns-3 runner + analytical ring model
    └── utils.py
├── ns3_backend/          # dooly-collective ns-3 scenario (C++), build scripts
├── examples/network/     # topology files for the ns-3 backend
├── tests/                # pytest suite
└── example_workload      # example sharegpt4 workload at 1rps
```

## Citation

```
@misc{kim2026doolyconfigurationagnosticredundancyawareprofiling,
      title={Dooly: Configuration-Agnostic, Redundancy-Aware Profiling for LLM Inference Simulation},
      author={Joon Ha Kim and Geon-Woo Kim and Anoop Rachakonda and Daehyeok Kim},
      year={2026},
      eprint={2605.07985},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2605.07985},
}
```

## License

MIT — see [LICENSE](./LICENSE).
