# DoolySim

A discrete-event simulator for vLLM LLM-inference serving.

`DoolySim` predicts end-to-end serving performance (latency, throughput, per-request TPOT/TTFT) by emulating vLLM's scheduler with learned latency models. 
It consumes a profiling database produced by [`DoolyProf`](https://github.com/dooly-project/doolyprof) and uses regression models to predict each step's latency without running the model on a GPU.

The scheduler emulation uses real vLLM scheduler code under the hood, so the batching decisions match production vLLM exactly.

## Installation

Note: if you had already installed the requirements from `DoolyProf`, you may skip the installation step. 

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
- `--use-vllm-dataset`: use vLLM's RandomDataset for reproducible workloads
- `--load-dataset PATH`: replay a saved JSONL request dataset
- `--export-json PATH`: emit results in vLLM's benchmark format for comparison
- `--show-per-request`: show per-request latency breakdown

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
    └── utils.py
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
