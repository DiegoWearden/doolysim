"""Simulation predictor for per-step latency estimation.

This module provides the SimulationPredictor class, which loads pre-generated
CSV prediction caches and estimates latency for each simulation step based on
batch composition and operation counts.

This is a runtime-only component - it does NOT train models. Training must be
done separately using simulator.estimator.
"""

from typing import Dict, Tuple, Optional, List
import sqlite3
import os
import sys
import numpy as np
from pathlib import Path

from ..request import BatchConfig

import torch
# Dtype to bytes mapping
DTYPE_TO_BYTES = {
    torch.float32: 4,
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.int64: 8,
    torch.int32: 4,
    torch.int8: 1,
}

def is_verbose() -> bool:
    """Check if verbose mode is enabled."""
    try:
        from .. import cli
        return cli.VERBOSE
    except (ImportError, AttributeError):
        return False


class SimulationPredictor:
    """Runtime predictor for simulation step latencies.

    This class loads pre-generated CSV prediction caches and estimates
    per-step latency based on:
    - Batch composition (prefill vs decode)
    - Operation counts from profiling
    - Collective communication configurations

    NOTE: This class does NOT train models. Use simulator.estimator for training.
    """

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int,
        dtype: torch.dtype,
        backend: str,
        vllm_config,
        cache_dir: str,
        db_path: str,
        topology: str = "mew1",
        fallback_duration_ms: float = 10.0,
        use_csv_cache: bool = True,
    ):
        """Initialize latency estimator.

        Args:
            model_name: Name of the model being simulated
            backend: Backend type (e.g., "FLASHINFER")
            vllm_config: vLLM configuration object
            cache_dir: Directory containing cached prediction CSV files
            db_path: Path to profiling database
            topology: GPU interconnect topology for collective operations
            fallback_duration_ms: Fallback step duration if models fail to load
            use_csv_cache: If True, try to load from CSV cache first (faster)
        """
        self.model_name = model_name
        self.backend = backend
        self.vllm_config = vllm_config
        self.fallback_duration_ms = fallback_duration_ms
        self.use_csv_cache = use_csv_cache
        self.db_path = db_path

        # Extract TP size and topology for collective operations
        self.tp_size = tensor_parallel_size
        self.topology = topology

        # Extract model config for size calculations
        self.dtype_bytes = DTYPE_TO_BYTES.get(dtype)

        # Initialize predictor and load operations
        self.latency_predictor = self._init_predictor(cache_dir, db_path)
        self.model_operations = self._load_model_operations(db_path)
        self.model_collectives = self._load_model_collectives(db_path)
        self.collective_configs = self._load_model_collective_configs(db_path)
        # (op_name, sig_hash) pairs profiled with is_context=1; estimator trains
        # them with separate prefill/decode models. Drives the split at lookup time.
        self.context_split_ops = self._load_context_split_ops(db_path)

    def _init_predictor(self, cache_dir: str, db_path: str):
        """Initialize latency predictor from CSV cache ONLY.

        This method ONLY loads pre-generated CSV predictions.
        If CSV cache is missing, it fails with a helpful error message.
        Training must be done separately using estimator.py.

        Args:
            cache_dir: Directory containing CSV prediction files
            db_path: Path to profiling database (unused, kept for compatibility)

        Returns:
            OperationTimePredictor instance

        Raises:
            RuntimeError: If CSV cache is not found or empty
        """
        try:
            # Add simulator directory to path if not already there
            simulator_dir = Path(__file__).parent.parent
            if str(simulator_dir) not in sys.path:
                sys.path.insert(0, str(simulator_dir))

            from doolysim.core.prediction_cache import load_all_predictions
            from doolysim.operation_time_predictor import OperationTimePredictor

            print(f"Loading CSV prediction cache from {cache_dir}...")
            cache = load_all_predictions(cache_dir)

            if len(cache.predictions) == 0:
                raise RuntimeError(
                    f"\n{'='*80}\n"
                    f"ERROR: No CSV predictions found in {cache_dir}\n"
                    f"\nPlease run training first:\n"
                    f"  python -m doolysim.estimator \\\n"
                    f"    --model {self.model_name} \\\n"
                    f"    --backend {self.backend} \\\n"
                    f"    --tp-size {self.tp_size} \\\n"
                    f"    --topology {self.topology} \\\n"
                    f"    --cache-dir {cache_dir} \\\n"
                    f"    --db-path {db_path}\n"
                    f"{'='*80}"
                )

            # Infer operation_is_context from cache key naming: any op with a
            # _prefill or _decode variant in the cache is context-split (the
            # estimator only produces those when is_context=1 in the DB).
            # This covers Attention, FusedMoE, and any future context-sensitive op.
            import re
            operation_is_context = {}
            for op_name in cache.predictions.keys():
                base_name = op_name.replace('_prefill', '').replace('_decode', '')
                base_name = re.sub(r'_[0-9a-f]{8}$', '', base_name)
                if '_prefill' in op_name or '_decode' in op_name:
                    operation_is_context[base_name] = True
                else:
                    operation_is_context.setdefault(base_name, False)

            predictor = OperationTimePredictor(
                models={},  # No sklearn models needed for CSV cache
                predictions=cache.predictions,
                operation_is_context=operation_is_context,
                kv_cache_granularity=64,
            )
            print(f"Successfully loaded {len(cache.predictions)} operations from CSV cache")
            return predictor

        except Exception as e:
            print(f"\nERROR: Failed to initialize simulation predictor: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _load_model_operations(self, db_path: str) -> Dict[Tuple[str, str], int]:
        """Load operation counts from model_operations table.

        Returns:
            Dict mapping (operation_name, signature_hash) -> count
        """
        if not os.path.exists(db_path):
            print(f"Warning: Database not found at {db_path}")
            return {}

        try:
            conn = sqlite3.connect(db_path)
            query = """
                SELECT s.operation_name, s.signature_hash, mo.count
                FROM model_operations mo
                JOIN signatures s ON mo.operation_id = s.operation_id
                JOIN profiled_models pm ON mo.model_id = pm.model_id
                WHERE pm.model_name = ? AND pm.backend = ?
            """

            cursor = conn.execute(query, (self.model_name, self.backend))
            operations = {}
            for row in cursor:
                op_name, sig_hash, count = row
                operations[(op_name, sig_hash)] = count

            conn.close()
            print(f"Loaded {len(operations)} operation signatures from database (backend={self.backend})")
            return operations

        except Exception as e:
            print(f"Warning: Failed to load model operations: {e}")
            return {}

    def _load_context_split_ops(self, db_path: str) -> set:
        """Return {(operation_name, signature_hash)} for ops whose measurements
        include is_context=1 rows in the DB.

        These ops need the estimator's prefill/decode split handling (same as
        Attention). Derived from the DB so new context-sensitive ops (e.g.
        FusedMoE) don't require hardcoding in the predictor.
        """
        if not os.path.exists(db_path):
            return set()
        try:
            conn = sqlite3.connect(db_path)
            rows = conn.execute(
                """
                SELECT DISTINCT s.operation_name, s.signature_hash
                FROM signatures s
                JOIN results r ON r.signature_hash = s.signature_hash
                JOIN model_operations mo ON mo.operation_id = s.operation_id
                JOIN profiled_models pm ON pm.model_id = mo.model_id
                WHERE pm.model_name = ? AND pm.backend = ?
                  AND r.is_context = 1
                """,
                (self.model_name, self.backend),
            ).fetchall()
            conn.close()
            result = {(op, sig) for op, sig in rows}
            if result:
                print(f"Context-split ops (prefill/decode): "
                      f"{sorted({op for op, _ in result})}")
            return result
        except Exception as e:
            print(f"Warning: Failed to load context_split_ops: {e}")
            return set()

    def _load_model_collectives(self, db_path: str):
        """Load collective operations for this model configuration.

        Returns:
            List of (operation_name, dim_param, num_dims, count) tuples
        """
        if not os.path.exists(db_path):
            print(f"Warning: Database not found at {db_path}")
            return []

        try:
            conn = sqlite3.connect(db_path)
            query = """
                SELECT co.operation_name, co.dim_param, co.num_dims, mc.count
                FROM model_collectives mc
                JOIN coll_ops co ON mc.collective_id = co.collective_id
                JOIN profiled_models pm ON mc.model_id = pm.model_id
                WHERE pm.model_name = ?
                  AND pm.backend = ?
                  AND co.topology = ?
                  AND co.tp_degree = ?
            """

            cursor = conn.execute(query, (self.model_name, self.backend, self.topology, self.tp_size))
            collectives = cursor.fetchall()
            conn.close()

            if collectives:
                print(f"Loaded {len(collectives)} collective operations (topology={self.topology}, tp={self.tp_size})")
                for op_name, dim_param, num_dims, count in collectives:
                    print(f"  - {op_name} (dim={dim_param}, ndims={num_dims}) × {count}")
            else:
                print(f"No collective operations found for {self.model_name} (topology={self.topology}, tp={self.tp_size})")

            
            # print(collectives)
            # exit()
            return collectives

        except Exception as e:
            print(f"Warning: Failed to load collective operations: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _load_model_collective_configs(self, db_path: str) -> Dict[Tuple, Dict]:
        """Load collective operation configurations for this model.

        Returns:
            Dict mapping (operation_name, dim_param, num_dims) -> config dict
            where config dict contains 'batch_param' and 'model_config_dim'
        """
        if not os.path.exists(db_path):
            print(f"Warning: Database not found at {db_path}")
            return {}

        try:
            conn = sqlite3.connect(db_path)
            query = """
                SELECT co.operation_name, co.dim_param, co.num_dims,
                       mcc.batch_param, mcc.model_config_dim
                FROM model_collective_configs mcc
                JOIN coll_ops co ON mcc.collective_id = co.collective_id
                JOIN profiled_models pm ON mcc.model_id = pm.model_id
                WHERE pm.model_name = ?
                  AND pm.backend = ?
                  AND pm.tp_degree = ?
                  AND co.topology = ?
            """

            cursor = conn.execute(query, (self.model_name, self.backend, self.tp_size, self.topology))
            configs = {}
            for row in cursor:
                op_name, dim_param, num_dims, batch_param, model_config_dim = row
                key = (op_name, dim_param, num_dims)
                configs[key] = {
                    'batch_param': batch_param,
                    'model_config_dim': model_config_dim
                }
            conn.close()

            if configs:
                print(f"Loaded {len(configs)} collective operation configs")
                for key, config in configs.items():
                    op_name, dim_param, num_dims = key
                    print(f"  - {op_name} (dim={dim_param}, ndims={num_dims}): "
                          f"{config['batch_param']} × {config['model_config_dim']}")
            else:
                print(f"No collective configs found for {self.model_name} (tp={self.tp_size}, topology={self.topology})")

            return configs

        except Exception as e:
            print(f"Warning: Failed to load collective configs: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def _calculate_collective_size(self, op_name: str, dim_param: int, num_dims: int, batch) -> int:
        # Look up configuration
        key = (op_name, dim_param, num_dims)
        config = self.collective_configs.get(key)

        # Use stored configuration
        batch_param = config['batch_param']

        if batch_param == "NUM_TOKS":
            num_elements = batch.total_tokens
        elif batch_param == "NUM_REQS":
            num_elements = batch.num_requests  # Number of requests in batch
        else:
            raise ValueError(f"Unknown batch parameter {batch_param} for collective config {key}")
        
        return num_elements

    def _estimate_collective_latency(
        self,
        op_name: str,
        dim_param: int,
        num_dims: int,
        num_elements: int
    ) -> float:
        dim_str = "None" if dim_param is None or (isinstance(dim_param, float) and np.isnan(dim_param)) else str(int(dim_param))
        model_key = f"{op_name}_dim{dim_str}_ndims{int(num_dims)}"

        # Use regression model from predictor
        if self.latency_predictor is None:
            print(f"  Warning: No latency predictor available for {model_key}")
            return 0.0

        try:
            # Check if model exists in either models or predictions (CSV cache)
            if model_key not in self.latency_predictor.models and model_key not in self.latency_predictor.predictions:
                print(f"  Warning: No model found for {model_key}")
                return 0.0

            # Use predict_simple with num_tokens
            pred = self.latency_predictor.predict_simple(
                model_key,
                num_tokens=num_elements
            )
            return float(pred)

        except Exception as e:
            print(f"  Warning: Model prediction failed for {model_key}: {e}")
            return 0.0

    def estimate_step_latency(self, sched_output, step_id: int = -1) -> float:
        """Estimate step latency using regression models.

        Args:
            sched_output: SchedulerOutput from scheduler.schedule()
            step_id: Current step ID for debug output
        Returns:
            Estimated step latency in milliseconds
        """
        if self.latency_predictor is None or not self.model_operations:
            print(f"[WARNING] Latency predictor or model operations not available, using fallback duration")
            return self.fallback_duration_ms

        # Debug: Show available models on first call
        if is_verbose() and step_id == 0:
            print(f"[DEBUG] Total models loaded: {len(self.latency_predictor.models)}")
            attention_models = sorted([k for k in self.latency_predictor.models.keys() if 'Attention' in k])
            print(f"[DEBUG] Attention models available: {attention_models}")

        try:
            # Extract batch information
            batch = BatchConfig(sched_output)
            print(f"\n[Step {step_id}] Estimating latency for batch: total_tokens={batch.total_tokens}, {batch.batch_size} requests (prefill={len(batch.prefill_requests)}, decode={len(batch.decode_requests)})")

            # DETECT MIXED BATCHES (prefill + decode together)
            if len(batch.prefill_requests) > 0 and len(batch.decode_requests) > 0:
                print(f"\n{'='*80}")
                print(f"MIXED BATCH DETECTED at Step {step_id}!")
                print(f"  Prefill requests: {len(batch.prefill_requests)}")
                print(f"  Decode requests: {len(batch.decode_requests)}")
                print(f"  This requires special handling for latency prediction!")
                print(f"{'='*80}\n")

            # If no tokens scheduled, return 0
            if batch.total_tokens == 0:
                return 0.0

            total_latency_ms = 0.0

            # Iterate through all operations for this model
            for (op_name, sig_hash), count in self.model_operations.items():
                sig_short = sig_hash[:8]
                model_key = f"{op_name}_{sig_short}"

                # Ops whose profile has is_context=1 rows get a prefill/decode split
                # (estimator trains separate models with the same feature schema).
                is_split = (op_name, sig_hash) in self.context_split_ops

                if is_split:
                    prefill_key = f"{op_name}_prefill_{sig_short}"
                    decode_key = f"{op_name}_decode_{sig_short}"
                    # Check both models (for sklearn) and predictions (for CSV cache)
                    has_model = (prefill_key in self.latency_predictor.models or
                                decode_key in self.latency_predictor.models or
                                prefill_key in self.latency_predictor.predictions or
                                decode_key in self.latency_predictor.predictions)
                    if not has_model:
                        if is_verbose():
                            print(f"[DEBUG Step {step_id}] Skipping {op_name}_{sig_short} - no prefill/decode models found")
                        continue
                else:
                    # Standard check for non-split operations
                    # Check both models (for sklearn) and predictions (for CSV cache)
                    if model_key not in self.latency_predictor.models and model_key not in self.latency_predictor.predictions:
                        if is_verbose():
                            print(f"[DEBUG Step {step_id}] Skipping {model_key} - no model found")
                        continue

                # Predict latency based on operation type
                if is_split:
                    op_latency = self._estimate_attention_latency(batch, sig_short, op_name, step_id)
                    contribution = op_latency * count
                    total_latency_ms += contribution
                    print(f"  [{op_name}_{sig_short}] op_latency={op_latency:.6f} ms × count={count} = {contribution:.6f} ms")
                else:
                    # Non-attention operations (linear, etc)
                    try:
                        pred = self.latency_predictor.predict_simple(
                            model_key,
                            num_tokens=batch.total_tokens
                        )
                        contribution = pred * count
                        total_latency_ms += contribution

                    # DEBUG: Show predictions for decode steps
                        print(f"  [{model_key}] pred={pred:.6f} ms × count={count} = {contribution:.6f} ms (num_tokens={batch.total_tokens})")
                    except Exception as e:
                        print(f"  [{model_key}] FAILED: {e}")
                        pass
            
            gpu_latency_ms = total_latency_ms
            
            # Add collective communication overhead
            comm_latency_ms = 0.0

            if self.model_collectives:
                for op_name, dim_param, num_dims, count in self.model_collectives:
                    # Calculate tensor size for this batch using model-specific config
                    num_elements = self._calculate_collective_size(op_name, dim_param, num_dims, batch)

                    # Estimate latency for one occurrence
                    op_latency = self._estimate_collective_latency(op_name, dim_param, num_dims, num_elements)

                    # Multiply by count
                    contribution = op_latency * count
                    comm_latency_ms += contribution

                    print(f"  [Comm {op_name}] feature={num_elements}, latency={op_latency:.3f}ms × {count} = {contribution:.3f}ms")

            total_latency_ms += comm_latency_ms

            if is_verbose() and comm_latency_ms > 0 and step_id == 0:
                print(f"  Total comm overhead: {comm_latency_ms:.3f}ms")

            # Return latency in milliseconds
            # DEBUG: Show total for decode steps
            if len(batch.decode_requests) > 0 and step_id >= 0:
                print(f"\n[Step {step_id}] TOTAL PREDICTED LATENCY: {total_latency_ms:.6f} ms")
                print(f"  (This is for {len(batch.decode_requests)} decode request(s) generating {batch.total_tokens} token(s))")

            return total_latency_ms

        except Exception as e:
            # Fallback to fixed duration on any error
            print(f"Warning: Latency estimation failed: {e}, using fixed duration")
            return self.fallback_duration_ms
        
    def estimate_step_latency_with_breakdown(self, sched_output, step_id: int = -1) -> Tuple[float, float, Dict[str, float], List[Dict]]:
        """Estimate step latency with detailed operation breakdown for JSON output.

        Returns:
            Tuple of (gpu_latency_ms, comm_latency_ms, per_comm_latency_dict, operation_breakdown)
        """
        # Store original breakdown data
        self._operation_breakdown = []

        # Call the original function
        gpu_latency_ms, comm_latency_ms, per_comm_latency_dict = self.estimate_step_latency_log(sched_output, step_id)

        # Return with breakdown
        return gpu_latency_ms, comm_latency_ms, per_comm_latency_dict, self._operation_breakdown


    def estimate_step_latency_log(self, sched_output, step_id: int = -1) -> Tuple[float, float, Dict[str, float]]:
        """Estimate step latency using regression models.

        Args:
            sched_output: SchedulerOutput from scheduler.schedule()
            step_id: Current step ID for debug output
        Returns:
            Tuple of (gpu_latency_ms, comm_latency_ms, per_comm_latency_dict)
        """
        per_communication_latency = {
            'all_reduce_latency_ms': 0.0,
            'all_reduce_count': 0,
            'per_all_reduce_latency_ms': 0.0,
            'all_gather_latency_ms': 0.0,
            'all_gather_count': 0,
            'per_all_gather_latency_ms': 0.0,
        }

        # Initialize breakdown storage if not already done
        if not hasattr(self, '_operation_breakdown'):
            self._operation_breakdown = []

        if self.latency_predictor is None or not self.model_operations:
            print(f"[WARNING] Latency predictor or model operations not available, using fallback duration")
            return self.fallback_duration_ms, 0.0, per_communication_latency

        # Debug: Show available models on first call
        if is_verbose() and step_id == 0:
            print(f"[DEBUG] Total models loaded: {len(self.latency_predictor.models)}")
            attention_models = sorted([k for k in self.latency_predictor.models.keys() if 'Attention' in k])
            print(f"[DEBUG] Attention models available: {attention_models}")

        try:
            # Extract batch information
            batch = BatchConfig(sched_output)
            print(f"\n[Step {step_id}] Estimating latency for batch: total_tokens={batch.total_tokens}, {batch.batch_size} requests (prefill={len(batch.prefill_requests)}, decode={len(batch.decode_requests)})")

            # DETECT MIXED BATCHES (prefill + decode together)
            if len(batch.prefill_requests) > 0 and len(batch.decode_requests) > 0:
                print(f"\n{'='*80}")
                print(f"MIXED BATCH DETECTED at Step {step_id}!")
                print(f"  Prefill requests: {len(batch.prefill_requests)}")
                print(f"  Decode requests: {len(batch.decode_requests)}")
                print(f"  This requires special handling for latency prediction!")
                print(f"{'='*80}\n")

            # If no tokens scheduled, return 0
            if batch.total_tokens == 0:
                return 0.0

            total_latency_ms = 0.0

            # Iterate through all operations for this model
            for (op_name, sig_hash), count in self.model_operations.items():
                sig_short = sig_hash[:8]
                model_key = f"{op_name}_{sig_short}"

                # Ops whose profile has is_context=1 rows get a prefill/decode split
                # (estimator trains separate models with the same feature schema).
                is_split = (op_name, sig_hash) in self.context_split_ops

                if is_split:
                    prefill_key = f"{op_name}_prefill_{sig_short}"
                    decode_key = f"{op_name}_decode_{sig_short}"
                    # Check both models (for sklearn) and predictions (for CSV cache)
                    has_model = (prefill_key in self.latency_predictor.models or
                                decode_key in self.latency_predictor.models or
                                prefill_key in self.latency_predictor.predictions or
                                decode_key in self.latency_predictor.predictions)
                    if not has_model:
                        if is_verbose():
                            print(f"[DEBUG Step {step_id}] Skipping {op_name}_{sig_short} - no prefill/decode models found")
                        continue
                else:
                    # Standard check for non-split operations
                    # Check both models (for sklearn) and predictions (for CSV cache)
                    if model_key not in self.latency_predictor.models and model_key not in self.latency_predictor.predictions:
                        if is_verbose():
                            print(f"[DEBUG Step {step_id}] Skipping {model_key} - no model found")
                        continue

                # Predict latency based on operation type
                if is_split:
                    op_latency = self._estimate_attention_latency(batch, sig_short, op_name, step_id)
                    contribution = op_latency * count
                    total_latency_ms += contribution
                    print(f"  [{op_name}_{sig_short}] op_latency={op_latency:.6f} ms × count={count} = {contribution:.6f} ms")
                else:
                    # Non-attention operations (linear, etc)
                    try:
                        pred = self.latency_predictor.predict_simple(
                            model_key,
                            num_tokens=batch.total_tokens
                        )
                        contribution = pred * count
                        total_latency_ms += contribution

                    # DEBUG: Show predictions for decode steps
                        print(f"  [{model_key}] pred={pred:.6f} ms × count={count} = {contribution:.6f} ms (num_tokens={batch.total_tokens})")
                    except Exception as e:
                        print(f"  [{model_key}] FAILED: {e}")
                        pass
            
            gpu_latency_ms = total_latency_ms
            
            # Add collective communication overhead
            comm_latency_ms = 0.0

            if self.model_collectives:
                for op_name, dim_param, num_dims, count in self.model_collectives:
                    # Calculate tensor size for this batch using model-specific config
                    num_elements = self._calculate_collective_size(op_name, dim_param, num_dims, batch)

                    # Estimate latency for one occurrence
                    op_latency = self._estimate_collective_latency(op_name, dim_param, num_dims, num_elements)

                    # Multiply by count
                    contribution = op_latency * count
                    comm_latency_ms += contribution
                    
                    # Track detailed communication metrics
                    if 'all_reduce' in op_name.lower():
                        per_communication_latency['all_reduce_latency_ms'] += contribution
                        per_communication_latency['all_reduce_count'] += count
                        per_communication_latency['per_all_reduce_latency_ms'] = op_latency
                    elif 'all_gather' in op_name.lower():
                        per_communication_latency['all_gather_latency_ms'] += contribution
                        per_communication_latency['all_gather_count'] += count
                        per_communication_latency['per_all_gather_latency_ms'] = op_latency

                    print(f"  [Comm {op_name}] feature={num_elements}, latency={op_latency:.3f}ms × {count} = {contribution:.3f}ms")

            total_latency_ms += comm_latency_ms

            if is_verbose() and comm_latency_ms > 0 and step_id == 0:
                print(f"  Total comm overhead: {comm_latency_ms:.3f}ms")

            # Return latency in milliseconds
            # DEBUG: Show total for decode steps
            if len(batch.decode_requests) > 0 and step_id >= 0:
                print(f"\n[Step {step_id}] TOTAL PREDICTED LATENCY: {total_latency_ms:.6f} ms")
                print(f"  (This is for {len(batch.decode_requests)} decode request(s) generating {batch.total_tokens} token(s))")

            return gpu_latency_ms, comm_latency_ms, per_communication_latency

        except Exception as e:
            # Fallback to fixed duration on any error
            print(f"Warning: Latency estimation failed: {e}, using fixed duration")
            return self.fallback_duration_ms

    def _estimate_attention_latency(self, batch, sig_short: str, op_name: str, step_id: int = -1) -> float:
        """Estimate attention operation latency for prefill and decode phases.

        Args:
            batch: BatchConfig containing prefill and decode request info
            sig_short: Short signature hash
            op_name: Operation name ("Attention")
            step_id: Current step ID for debug output

        Returns:
            Estimated latency in milliseconds
        """
        op_latency = 0.0

        # Prefill phase - aggregate using L2 norm
        if batch.prefill_requests:
            # Extract kv_cache_sizes and prefill_chunk_sizes
            # Note: tuple format is (num_tokens, kv_cache_size)
            kv_cache_sizes = [kv_cache_size for num_tokens, kv_cache_size in batch.prefill_requests]
            prefill_chunk_sizes = [num_tokens for num_tokens, kv_cache_size in batch.prefill_requests]

            # Aggregate using sum for kv_cache and L2 norm for prefill_chunk_size
            agg_kv_cache_size = sum(kv_cache_sizes)
            agg_prefill_chunk_size = sum([x**2 for x in prefill_chunk_sizes]) ** 0.5  # L2 norm

            # if is_verbose():
                # print(f"      [Attention Prefill] num_reqs={len(batch.prefill_requests)}, "
                #         f"agg_kv_cache={agg_kv_cache_size}, "
                #         f"agg_prefill_chunk={agg_prefill_chunk_size:.1f} (L2 norm of {prefill_chunk_sizes})")

            # Use Attention_prefill_{sig_short} model key
            prefill_model_key = f"{op_name}_prefill_{sig_short}"

            try:
                # Make ONE prediction for the entire prefill batch
                pred = self.latency_predictor.predict_simple(
                    prefill_model_key,
                    kv_cache_size=agg_kv_cache_size,
                    prefill_chunk_size=int(round(agg_prefill_chunk_size))
                )

                # Apply batching overhead (Vidur's approach for GQA models)
                # Llama-3.1-8B uses GQA: num_q_heads=32, num_kv_heads=8
                ATTENTION_PREFILL_BATCHING_OVERHEAD = 0.1
                num_prefill_reqs = len(batch.prefill_requests)
                overhead_multiplier = 1.0 + ATTENTION_PREFILL_BATCHING_OVERHEAD * int(num_prefill_reqs > 1)

                pred_with_overhead = pred * overhead_multiplier
                op_latency += pred_with_overhead

                if num_prefill_reqs > 1:
                    print(f"  [Attention Prefill] base={pred:.6f}ms, overhead={ATTENTION_PREFILL_BATCHING_OVERHEAD*100:.0f}%, final={pred_with_overhead:.6f}ms ({num_prefill_reqs} reqs)")

            except Exception as e:
                    print(f"    WARNING: Prefill prediction failed - agg_kv={agg_kv_cache_size}, "
                          f"agg_chunk={int(round(agg_prefill_chunk_size))}, error: {e}")
                    op_latency = 0.0

        # Decode phase - use decode batch size only
        if batch.decode_requests:
            # Decode batch size = number of decode requests ONLY
            decode_batch_size = len(batch.decode_requests)
            # Average KV cache size across decode requests
            avg_kv_cache = sum(batch.decode_requests) / decode_batch_size

            # DEBUGGING: Check if batch_size exceeds profiled range
            MAX_PROFILED_BATCH_SIZE = 32
            MAX_PROFILED_KV_CACHE = 16384

            print(f"\n[DECODE DEBUG] Step {step_id}:")
            print(f"  decode_batch_size: {decode_batch_size}")
            print(f"  decode_kv_values (first 5): {batch.decode_requests[:5]}")
            print(f"  avg_kv_cache: {int(avg_kv_cache)}")

            # if decode_batch_size > MAX_PROFILED_BATCH_SIZE:
            #     print(f"\n{'='*80}")
            #     print(f"ERROR: Decode batch_size={decode_batch_size} EXCEEDS profiled range!")
            #     print(f"  Max profiled batch_size: {MAX_PROFILED_BATCH_SIZE}")
            #     print(f"  Current batch_size: {decode_batch_size}")
            #     print(f"  This will cause EXTRAPOLATION errors in regression model!")
            #     print(f"{'='*80}\n")
            #     import sys
            #     sys.exit(1)

            if avg_kv_cache > MAX_PROFILED_KV_CACHE:
                print(f"\n{'='*80}")
                print(f"WARNING: avg_kv_cache={int(avg_kv_cache)} exceeds profiled range!")
                print(f"  Max profiled kv_cache: {MAX_PROFILED_KV_CACHE}")
                print(f"  Current avg_kv_cache: {int(avg_kv_cache)}")
                print(f"{'='*80}\n")

            # Use Attention_decode_{sig_short} model key
            decode_model_key = f"{op_name}_decode_{sig_short}"

            try:
                pred = self.latency_predictor.predict_simple(
                    decode_model_key,
                    batch_size=decode_batch_size,
                    kv_cache_size=int(avg_kv_cache)
                )

                # Apply batching overhead (Vidur's approach for GQA models)
                ATTENTION_DECODE_BATCHING_OVERHEAD = 0.1
                overhead_multiplier = 1.0 + ATTENTION_DECODE_BATCHING_OVERHEAD * int(decode_batch_size > 1)

                pred_with_overhead = pred * overhead_multiplier
                op_latency += pred_with_overhead

                print(f"  Base prediction: {pred:.6f} ms (for ONE layer)")
                if decode_batch_size > 1:
                    print(f"  Batching overhead: {ATTENTION_DECODE_BATCHING_OVERHEAD*100:.0f}% (batch_size={decode_batch_size})")
                print(f"  Final prediction: {pred_with_overhead:.6f} ms (for ONE layer)")
                print(f"  Total for 32 layers: {pred_with_overhead * 32:.6f} ms")
            except Exception as e:
                    print(f"    WARNING: Decode prediction failed - batch={decode_batch_size}, "
                          f"avg_kv={int(avg_kv_cache)}, error: {e}")
        else:
            print(f"      [Attention] NO DECODE REQUESTS - prefill_reqs={len(batch.prefill_requests)}, "
                f"decode_reqs={len(batch.decode_requests)}")

        return op_latency
