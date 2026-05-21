#!/usr/bin/env python
"""
Standalone batch latency predictor.

Predicts latency for a given batch configuration without running full simulation.
Uses pre-trained models and CSV prediction caches.
"""

import argparse
import json
import os
import sys
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from doolysim.core.simulation_predictor import SimulationPredictor


@dataclass
class SimpleBatchConfig:
    """Simplified batch configuration for prediction.

    This replaces the scheduler-dependent BatchConfig with a simple structure
    that can be manually constructed from CLI arguments.
    """
    prefill_requests: List[Tuple[int, int]]  # List of (tokens, kv_cache_size)
    decode_requests: List[int]  # List of kv_cache_sizes

    def __post_init__(self):
        """Calculate derived properties."""
        # Calculate total tokens
        prefill_tokens = sum(tokens for tokens, _ in self.prefill_requests)
        decode_tokens = len(self.decode_requests)  # Each decode request generates 1 token
        self.total_tokens = prefill_tokens + decode_tokens

        # Calculate batch size
        self.batch_size = len(self.prefill_requests) + len(self.decode_requests)

        # Number of requests (for collective operations)
        self.num_requests = self.batch_size

    def __repr__(self):
        return (f"SimpleBatchConfig(batch_size={self.batch_size}, "
                f"prefill={len(self.prefill_requests)} reqs, "
                f"decode={len(self.decode_requests)} reqs, "
                f"total_tokens={self.total_tokens})")


class MockSchedulerOutput:
    """Mock scheduler output for compatibility with SimulationPredictor."""

    def __init__(self, batch_config: SimpleBatchConfig):
        self.total_num_scheduled_tokens = batch_config.total_tokens

        # Mock new requests (prefills)
        self.scheduled_new_reqs = []
        for i, (tokens, kv_cache) in enumerate(batch_config.prefill_requests):
            mock_req = type('MockReq', (), {
                'req_id': f'prefill_{i}',
                'num_computed_tokens': kv_cache,
                'prompt_token_ids': [0] * tokens  # Dummy tokens
            })()
            self.scheduled_new_reqs.append(mock_req)

        # Mock cached requests (decode)
        self.scheduled_cached_reqs = type('MockCachedReqs', (), {
            'req_ids': [f'decode_{i}' for i in range(len(batch_config.decode_requests))],
            'num_reqs': len(batch_config.decode_requests),
            'num_computed_tokens': batch_config.decode_requests,
            'num_output_tokens': [1] * len(batch_config.decode_requests),  # All in decode phase
            'is_context_phase': lambda req_id: False  # All in decode
        })()

        # Mock token counts per request
        self.num_scheduled_tokens = {}
        for i, (tokens, _) in enumerate(batch_config.prefill_requests):
            self.num_scheduled_tokens[f'prefill_{i}'] = tokens
        for i in range(len(batch_config.decode_requests)):
            self.num_scheduled_tokens[f'decode_{i}'] = 1


class BatchPredictor:
    """Predicts latency for a given batch configuration."""

    def __init__(
        self,
        model_name: str,
        backend: str,
        cache_dir: str,
        db_path: str,
        tensor_parallel_size: int = 1,
        topology: str = "mew1",
        dtype: torch.dtype = torch.bfloat16,
        verbose: bool = False
    ):
        """Initialize the batch predictor.

        Args:
            model_name: Name of the model
            backend: Backend type (e.g., "FLASHINFER")
            cache_dir: Directory containing CSV prediction caches
            db_path: Path to profiling database
            tensor_parallel_size: Tensor parallel size
            topology: GPU interconnect topology
            dtype: Data type for tensors
            verbose: Enable verbose output
        """
        self.model_name = model_name
        self.backend = backend
        self.verbose = verbose

        # Create a mock vLLM config with minimal required fields
        self.mock_vllm_config = type('MockConfig', (), {
            'model_config': type('ModelConfig', (), {
                'max_model_len': 8192,
                'get_num_kv_heads': lambda parallel_cfg: 8,
                'get_head_size': lambda: 128,
            })(),
            'cache_config': type('CacheConfig', (), {
                'block_size': 16,
                'num_gpu_blocks': 1024,
            })(),
            'parallel_config': type('ParallelConfig', (), {
                'tensor_parallel_size': tensor_parallel_size,
            })(),
        })()

        print(f"Initializing BatchPredictor for {model_name} with backend {backend}")

        # Initialize the simulation predictor
        self.predictor = SimulationPredictor(
            model_name=model_name,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            backend=backend,
            vllm_config=self.mock_vllm_config,
            cache_dir=cache_dir,
            db_path=db_path,
            topology=topology,
            fallback_duration_ms=10.0,
            use_csv_cache=True
        )

    def predict(self, batch_config: SimpleBatchConfig) -> Dict[str, float]:
        """Predict latency for a batch configuration.

        Args:
            batch_config: Batch configuration to predict

        Returns:
            Dictionary with prediction results
        """
        import io
        import sys
        import re

        print(f"\nPredicting latency for batch: {batch_config}")

        # Create mock scheduler output
        mock_sched_output = MockSchedulerOutput(batch_config)

        # Capture console output to extract operation details
        original_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()

        try:
            # Get prediction from SimulationPredictor
            gpu_latency_ms, comm_latency_ms, per_comm_latency_ms = self.predictor.estimate_step_latency_log(
                sched_output=mock_sched_output,
                step_id=0
            )
        finally:
            sys.stdout = original_stdout

        # Parse captured output to extract operation breakdown
        output_lines = captured_output.getvalue().split('\n')
        operation_breakdown = []

        for line in output_lines:
            # Print the captured output back to console
            if line.strip():
                print(line)

            # Parse operation lines like: [VocabParallelEmbedding_81482ec3] pred=0.050070 ms × count=2 = 0.100139 ms (num_tokens=1024)
            if re.match(r'^\s*\[[^\]]+\]', line.strip()):
                parts = line.strip()

                # Extract operation name
                op_match = re.match(r'^\s*\[([^\]]+)\]', parts)
                if not op_match:
                    continue
                operation_name = op_match.group(1)

                # Parse computation operations
                if 'pred=' in parts and '×' in parts:
                    pred_match = re.search(r'pred=([0-9.]+)\s*ms.*count=([0-9]+).*=\s*([0-9.]+)\s*ms', parts)
                    token_match = re.search(r'num_tokens=([0-9]+)', parts)

                    if pred_match:
                        pred_latency = float(pred_match.group(1))
                        count = int(pred_match.group(2))
                        total_latency = float(pred_match.group(3))
                        num_tokens = int(token_match.group(1)) if token_match else batch_config.total_tokens

                        operation_breakdown.append({
                            'operation': operation_name,
                            'type': 'attention' if 'Attention' in operation_name else 'computation',
                            'pred_latency_ms': pred_latency,
                            'count': count,
                            'total_latency_ms': total_latency,
                            'num_tokens': num_tokens
                        })

                # Parse attention operations
                elif 'op_latency=' in parts and '×' in parts:
                    attn_match = re.search(r'op_latency=([0-9.]+)\s*ms.*count=([0-9]+).*=\s*([0-9.]+)\s*ms', parts)

                    if attn_match:
                        op_latency = float(attn_match.group(1))
                        count = int(attn_match.group(2))
                        total_latency = float(attn_match.group(3))

                        operation_breakdown.append({
                            'operation': operation_name,
                            'type': 'attention',
                            'pred_latency_ms': op_latency,
                            'count': count,
                            'total_latency_ms': total_latency,
                            'num_tokens': batch_config.total_tokens
                        })

                # Parse communication operations
                elif 'Comm' in operation_name and 'feature=' in parts:
                    comm_match = re.search(r'feature=([0-9]+).*latency=([0-9.]+)ms.*×\s*([0-9]+).*=\s*([0-9.]+)ms', parts)

                    if comm_match:
                        feature_size = int(comm_match.group(1))
                        op_latency = float(comm_match.group(2))
                        count = int(comm_match.group(3))
                        total_latency = float(comm_match.group(4))

                        operation_breakdown.append({
                            'operation': operation_name,
                            'type': 'communication',
                            'pred_latency_ms': op_latency,
                            'count': count,
                            'total_latency_ms': total_latency,
                            'feature_size': feature_size
                        })

        results = {
            'total_latency_ms': gpu_latency_ms + comm_latency_ms,
            'gpu_latency_ms': gpu_latency_ms,
            'comm_latency_ms': comm_latency_ms,
            'all_reduce_latency_ms': per_comm_latency_ms.get('all_reduce_latency_ms', 0.0),
            'all_reduce_count': per_comm_latency_ms.get('all_reduce_count', 0),
            'per_all_reduce_latency_ms': per_comm_latency_ms.get('per_all_reduce_latency_ms', 0.0),
            'all_gather_latency_ms': per_comm_latency_ms.get('all_gather_latency_ms', 0.0),
            'all_gather_count': per_comm_latency_ms.get('all_gather_count', 0),
            'per_all_gather_latency_ms': per_comm_latency_ms.get('per_all_gather_latency_ms', 0.0),
            'batch_size': batch_config.batch_size,
            'total_tokens': batch_config.total_tokens,
            'num_prefill_requests': len(batch_config.prefill_requests),
            'num_decode_requests': len(batch_config.decode_requests),
            'operation_breakdown': operation_breakdown,
        }

        # Add per-token metrics
        if batch_config.total_tokens > 0:
            results['ms_per_token'] = results['total_latency_ms'] / batch_config.total_tokens

        return results


def parse_prefill_requests(prefill_str: str) -> List[Tuple[int, int]]:
    """Parse prefill requests from string format.

    Format: "tokens1:kv1,tokens2:kv2,..."
    Example: "512:0,1024:256"
    """
    if not prefill_str:
        return []

    requests = []
    for req in prefill_str.split(','):
        parts = req.strip().split(':')
        if len(parts) == 2:
            tokens = int(parts[0])
            kv_cache = int(parts[1])
            requests.append((tokens, kv_cache))
        else:
            raise ValueError(f"Invalid prefill request format: {req}")

    return requests


def parse_decode_requests(decode_str: str) -> List[int]:
    """Parse decode requests from string format.

    Format: "kv1,kv2,kv3,..."
    Example: "1024,2048,512"
    """
    if not decode_str:
        return []

    return [int(kv.strip()) for kv in decode_str.split(',')]


def load_batch_from_json(json_path: str) -> Tuple[SimpleBatchConfig, Dict]:
    """Load batch configuration from JSON file.

    Returns:
        Tuple of (batch_config, metadata_dict)
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Extract batch configuration
    prefill_requests = []
    if 'prefill_requests' in data:
        for req in data['prefill_requests']:
            tokens = req.get('tokens', req.get('num_tokens', 0))
            kv_cache = req.get('kv_cache', req.get('kv_cache_size', 0))
            prefill_requests.append((tokens, kv_cache))

    decode_requests = []
    if 'decode_requests' in data:
        for req in data['decode_requests']:
            if isinstance(req, dict):
                kv_cache = req.get('kv_cache', req.get('kv_cache_size', 0))
            else:
                kv_cache = int(req)
            decode_requests.append(kv_cache)

    batch_config = SimpleBatchConfig(
        prefill_requests=prefill_requests,
        decode_requests=decode_requests
    )

    # Extract metadata
    metadata = {
        'model': data.get('model'),
        'backend': data.get('backend'),
        'tp_size': data.get('tensor_parallel_size', data.get('tp_size', 1)),
        'topology': data.get('topology', 'mew1'),
    }

    return batch_config, metadata


def main():
    parser = argparse.ArgumentParser(
        description='Predict latency for a batch configuration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple decode batch
  %(prog)s --model meta-llama/Llama-3.1-8B --backend FLASHINFER \\
           --decode-requests "1024,2048,512" \\
           --cache-dir ./cache/tp1/llama-3.1-8b

  # Prefill batch
  %(prog)s --model meta-llama/Llama-3.1-8B --backend FLASHINFER \\
           --prefill-requests "512:0,1024:0" \\
           --cache-dir ./cache/tp1/llama-3.1-8b

  # Mixed batch
  %(prog)s --model meta-llama/Llama-3.1-8B --backend FLASHINFER \\
           --prefill-requests "256:512" --decode-requests "1024,2048" \\
           --cache-dir ./cache/tp1/llama-3.1-8b

  # From JSON file
  %(prog)s --batch-config batch.json --cache-dir ./cache/tp1/llama-3.1-8b
"""
    )

    # Model configuration
    parser.add_argument('--model', type=str,
                        default='meta-llama/Llama-3.1-8B',
                        help='Model name')
    parser.add_argument('--backend', type=str,
                        default='FLASHINFER',
                        help='Backend type (FLASHINFER, FLASH_ATTN)')
    parser.add_argument('--tp-size', type=int, default=1,
                        help='Tensor parallel size')
    parser.add_argument('--topology', type=str, default='mew1',
                        help='GPU interconnect topology')

    # Batch specification
    parser.add_argument('--prefill-requests', type=str,
                        help='Prefill requests as "tokens1:kv1,tokens2:kv2,..."')
    parser.add_argument('--decode-requests', type=str,
                        help='Decode requests as "kv1,kv2,kv3,..."')
    parser.add_argument('--batch-config', type=str,
                        help='Load batch configuration from JSON file')

    # Paths
    parser.add_argument('--cache-dir', type=str, required=True,
                        help='Directory containing CSV prediction caches')
    parser.add_argument('--db-path', type=str,
                        default=os.environ.get('DOOLYSIM_DB_PATH'),
                        help='Path to profiling database (or set DOOLYSIM_DB_PATH env var)')

    # Output options
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('--json-output', type=str,
                        help='Save results to JSON file')

    args = parser.parse_args()

    # Load batch configuration
    if args.batch_config:
        # Load from JSON file
        batch_config, metadata = load_batch_from_json(args.batch_config)

        # Override with metadata from JSON (JSON takes priority)
        if metadata['model']:
            args.model = metadata['model']
        if metadata['backend']:
            args.backend = metadata['backend']
        if metadata['tp_size']:
            args.tp_size = metadata['tp_size']
        if metadata['topology']:
            args.topology = metadata['topology']
    else:
        # Parse from CLI arguments
        prefill_requests = parse_prefill_requests(args.prefill_requests or '')
        decode_requests = parse_decode_requests(args.decode_requests or '')

        if not prefill_requests and not decode_requests:
            parser.error('Must specify either --prefill-requests, --decode-requests, or --batch-config')

        batch_config = SimpleBatchConfig(
            prefill_requests=prefill_requests,
            decode_requests=decode_requests
        )

    # Initialize predictor
    predictor = BatchPredictor(
        model_name=args.model,
        backend=args.backend,
        cache_dir=args.cache_dir,
        db_path=args.db_path,
        tensor_parallel_size=args.tp_size,
        topology=args.topology,
        verbose=args.verbose
    )

    # Get prediction
    results = predictor.predict(batch_config)

    # Print results
    print("\n" + "=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Backend: {args.backend}")
    print(f"Tensor Parallel Size: {args.tp_size}")
    print(f"Topology: {args.topology}")
    print("-" * 60)
    print(f"Batch Size: {results['batch_size']} requests")
    print(f"  - Prefill: {results['num_prefill_requests']} requests")
    print(f"  - Decode: {results['num_decode_requests']} requests")
    print(f"Total Tokens: {results['total_tokens']}")
    print("-" * 60)
    print(f"PREDICTED LATENCY: {results['total_latency_ms']:.2f} ms")
    print(f"   COMPUTATION LATENCY: {results['gpu_latency_ms']:.2f} ms")
    print(f"   COMMUNICATION LATENCY: {results['comm_latency_ms']:.2f} ms")
    print("=" * 60)

    # Save to JSON if requested
    if args.json_output:
        output_data = {
            'model': args.model,
            'backend': args.backend,
            'tensor_parallel_size': args.tp_size,
            'topology': args.topology,
            'batch_config': {
                'prefill_requests': [
                    {'tokens': t, 'kv_cache': kv}
                    for t, kv in batch_config.prefill_requests
                ],
                'decode_requests': [
                    {'kv_cache': kv}
                    for kv in batch_config.decode_requests
                ],
            },
            'results': results
        }

        with open(args.json_output, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\nResults saved to {args.json_output}")

    return 0


if __name__ == '__main__':
    sys.exit(main())