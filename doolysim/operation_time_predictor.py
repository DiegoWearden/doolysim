"""
Operation Time Predictor - Vidur-style prediction interface.

Simple two-path prediction based on is_context flag:
- is_context=False: Use num_tokens_rounded (like Vidur's MLP/Linear/Norm/etc)
- is_context=True (Attention): Use Vidur's batch aggregation methods
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np


@dataclass
class Request:
    """Represents a single request in a batch."""
    num_processed_tokens: int  # How many tokens already processed (context length)
    is_prefill_complete: bool  # Whether prefill is done


@dataclass
class Batch:
    """Represents a batch of requests for prediction."""
    requests: List[Request]
    num_tokens: List[int]  # Tokens to process for each request in this iteration
    size: int  # Batch size

    def __post_init__(self):
        """Calculate derived properties."""
        # Total tokens across all requests
        self._total_num_tokens = sum(self.num_tokens)
        # Rounded up to nearest multiple of 8 (like Vidur)
        self._total_num_tokens_rounded = ((self._total_num_tokens + 7) // 8) * 8


class OperationTimePredictor:
    """
    Prediction interface matching Vidur's approach.

    Two prediction paths:
    1. Non-context operations (is_context=False): Use num_tokens_rounded
    2. Attention operations (is_context=True): Use Vidur's batch aggregation
    """

    def __init__(
        self,
        models: Dict[str, any],  # Trained sklearn models
        predictions: Dict[str, Dict[Tuple, float]],  # Pre-computed predictions
        operation_is_context: Dict[str, bool],  # Map operation -> is_context flag
        kv_cache_granularity: int = 64,
        attention_prefill_batching_overhead_fraction: float = 0.0,
        attention_decode_batching_overhead_fraction: float = 0.0,
    ):
        """
        Initialize predictor with trained models.

        Args:
            models: Dictionary of trained sklearn models per operation
            predictions: Pre-computed prediction lookup tables
            operation_is_context: Map from operation_name to is_context flag
            kv_cache_granularity: Granularity for rounding KV cache sizes
            attention_prefill_batching_overhead_fraction: Overhead when batching prefill
            attention_decode_batching_overhead_fraction: Overhead when batching decode
        """
        self.models = models
        self.predictions = predictions
        self.operation_is_context = operation_is_context
        self.kv_cache_granularity = kv_cache_granularity
        self.prefill_overhead = attention_prefill_batching_overhead_fraction
        self.decode_overhead = attention_decode_batching_overhead_fraction

    @classmethod
    def from_estimator(cls, estimator, operation_is_context_map: Dict[str, bool], **kwargs):
        """
        Create predictor from a trained estimator.

        Args:
            estimator: Trained SimplifiedExecutionTimeEstimator
            operation_is_context_map: Dictionary mapping operation names to is_context flag
            **kwargs: Additional arguments for overhead fractions, etc.
        """
        return cls(
            models=estimator.models,
            predictions=estimator.predictions,
            operation_is_context=operation_is_context_map,
            **kwargs
        )

    def _get_batch_decode_attention_params(self, batch: Batch) -> Tuple[int, int]:
        """
        Extract decode attention parameters from batch (Vidur's method).

        Returns:
            (decode_batch_size, average_kv_cache_size_rounded)
        """
        if hasattr(batch, "_decode_params"):
            return batch._decode_params

        decode_kv_cache_sizes = []

        for request in batch.requests:
            if request.is_prefill_complete:
                decode_kv_cache_sizes.append(request.num_processed_tokens)

        if not decode_kv_cache_sizes:
            batch._decode_params = (0, 0)
            return batch._decode_params

        decode_batch_size = len(decode_kv_cache_sizes)
        decode_avg_kv_cache_size = int(np.mean(decode_kv_cache_sizes))

        # Round to granularity
        decode_avg_kv_cache_size = (
            (decode_avg_kv_cache_size + self.kv_cache_granularity - 1)
            // self.kv_cache_granularity
        ) * self.kv_cache_granularity

        batch._decode_params = (decode_batch_size, decode_avg_kv_cache_size)
        return batch._decode_params

    def _get_batch_prefill_attention_params(self, batch: Batch) -> List[Tuple[int, int]]:
        """
        Extract prefill attention parameters from batch (Vidur's method).

        Returns:
            List of (kv_cache_size_rounded, prefill_chunk_size) for each prefill request
        """
        if hasattr(batch, "_prefill_params"):
            return batch._prefill_params

        prefill_params = []

        for request, num_tokens_to_process in zip(batch.requests, batch.num_tokens):
            if request.is_prefill_complete:
                continue

            prefill_chunk_size = num_tokens_to_process

            # Round KV cache size to granularity
            kv_cache_size = (
                (request.num_processed_tokens + self.kv_cache_granularity - 1)
                // self.kv_cache_granularity
            ) * self.kv_cache_granularity

            prefill_params.append((kv_cache_size, prefill_chunk_size))

        batch._prefill_params = prefill_params
        return prefill_params

    def predict(self, operation_name: str, batch: Batch) -> float:
        """
        Predict execution time for an operation given a batch.

        Two paths based on is_context flag:
        - is_context=False: Use num_tokens_rounded
        - is_context=True: Use Vidur's batch aggregation for attention

        Args:
            operation_name: Name of the operation (may include _prefill/_decode suffix)
            batch: Batch object containing requests

        Returns:
            Predicted execution time in milliseconds
        """
        if operation_name not in self.predictions:
            raise ValueError(f"No model trained for operation: {operation_name}")

        # Check if this is an attention operation
        base_op_name = operation_name.replace('_prefill', '').replace('_decode', '')
        is_context = self.operation_is_context.get(base_op_name, False)

        if not is_context:
            # Non-context operation: Use num_tokens_rounded (Vidur's approach for MLP/Linear/Norm/etc)
            return self.predictions[operation_name][(batch._total_num_tokens_rounded,)]

        # Context operation (Attention): Check prefill vs decode
        if '_prefill' in operation_name:
            return self._predict_attention_prefill(operation_name, batch)
        elif '_decode' in operation_name:
            return self._predict_attention_decode(operation_name, batch)
        else:
            raise ValueError(f"Attention operation {operation_name} must have _prefill or _decode suffix")

    def _predict_attention_decode(self, operation_name: str, batch: Batch) -> float:
        """
        Predict decode attention time using Vidur's aggregation.

        Uses average KV cache size across decode requests and applies
        batching overhead if batch_size > 1.
        """
        decode_batch_size, decode_avg_kv_cache_size = self._get_batch_decode_attention_params(batch)

        if decode_batch_size == 0:
            return 0.0

        # Lookup prediction
        base_time = self.predictions[operation_name][
            (decode_batch_size, decode_avg_kv_cache_size)
        ]

        # Apply batching overhead (Vidur's approach)
        overhead_multiplier = 1 + self.decode_overhead * int(decode_batch_size > 1)

        return base_time * overhead_multiplier

    def _predict_attention_prefill(self, operation_name: str, batch: Batch) -> float:
        """
        Predict prefill attention time using Vidur's aggregation.

        Uses sum of kv_cache_sizes and L2 norm of prefill_chunk_sizes.
        Applies batching overhead if batch has multiple prefill requests.
        """
        prefill_params = self._get_batch_prefill_attention_params(batch)

        if len(prefill_params) == 0:
            return 0.0

        # Vidur's aggregation: sum of kv_cache, L2 norm of prefill chunks
        kv_cache_sizes, prefill_chunk_sizes = zip(*prefill_params)

        agg_kv_cache_size = sum(kv_cache_sizes)
        agg_prefill_chunk_size = sum([x**2 for x in prefill_chunk_sizes]) ** 0.5

        # Lookup prediction (using squared chunk size as feature)
        base_time = self.predictions[operation_name][
            (agg_kv_cache_size, round(agg_prefill_chunk_size) ** 2)
        ]

        # Apply batching overhead (Vidur's approach)
        overhead_multiplier = 1 + self.prefill_overhead * int(len(prefill_params) > 1)

        return base_time * overhead_multiplier

    def predict_simple(
        self,
        operation_name: str,
        num_tokens: int = None,
        batch_size: int = None,
        kv_cache_size: int = None,
        prefill_chunk_size: int = None,
    ) -> float:
        """
        Simple prediction interface without creating Batch objects.

        Args:
            operation_name: Name of operation
            num_tokens: Total number of tokens (for non-context ops and collective ops)
            batch_size: Batch size (for decode attention)
            kv_cache_size: KV cache size (for attention) - will be rounded to granularity
            prefill_chunk_size: Prefill chunk size (for prefill attention)

        Returns:
            Predicted execution time in milliseconds
        """
        
        print(f"Predicting time for {operation_name} with num_tokens={num_tokens}, batch_size={batch_size}, kv_cache_size={kv_cache_size}, prefill_chunk_size={prefill_chunk_size}")
        
        if operation_name not in self.predictions:
            raise ValueError(f"No model trained for operation: {operation_name}")

        # Check if this is a collective communication operation
        if operation_name.startswith(('all_reduce_', 'all_gather_', 'reduce_scatter_')):
            # Collective operations now use num_tokens
            if num_tokens is None:
                # If size_bytes is provided but not num_tokens, this is an error
                # The caller should provide num_tokens directly
                raise ValueError(f"Need num_tokens for collective operation {operation_name}")
            key = (num_tokens,)
            if key in self.predictions[operation_name]:
                return self.predictions[operation_name][key]
            else:
                raise KeyError(f"Prediction not available for {operation_name} with num_tokens={num_tokens}")

        # Check if context operation
        # Remove _prefill, _decode, and signature hash to get base operation name
        base_op_name = operation_name.replace('_prefill', '').replace('_decode', '')
        # Also remove signature hash pattern (e.g., "_902fb90b")
        import re
        base_op_name = re.sub(r'_[0-9a-f]{8}$', '', base_op_name)
        is_context = self.operation_is_context.get(base_op_name, False)

        if not is_context:
            # Non-context: use num_tokens
            if num_tokens is None:
                raise ValueError(f"Need num_tokens for non-context operation {operation_name}")
            num_tokens_rounded = ((num_tokens + 7) // 8) * 8
            key = (num_tokens_rounded,)

        elif '_prefill' in operation_name:
            # Prefill attention
            if kv_cache_size is None or prefill_chunk_size is None:
                raise ValueError(f"Need kv_cache_size and prefill_chunk_size for {operation_name}")

            # Round KV cache to granularity (matching Vidur's behavior)
            kv_cache_rounded = (
                (kv_cache_size + self.kv_cache_granularity - 1)
                // self.kv_cache_granularity
            ) * self.kv_cache_granularity

            key = (kv_cache_rounded, prefill_chunk_size ** 2)

        elif '_decode' in operation_name:
            # Decode attention
            if batch_size is None or kv_cache_size is None:
                raise ValueError(f"Need batch_size and kv_cache_size for {operation_name}")

            # Round KV cache to granularity (matching Vidur's behavior)
            kv_cache_rounded = (
                (kv_cache_size + self.kv_cache_granularity - 1)
                // self.kv_cache_granularity
            ) * self.kv_cache_granularity

            key = (batch_size, kv_cache_rounded)

        else:
            raise ValueError(f"Unknown operation type: {operation_name}")

        # Lookup prediction
        if key in self.predictions[operation_name]:
            return self.predictions[operation_name][key]
        else:
            raise KeyError(f"Prediction not available for {operation_name} with key {key}")


# ========== Example usage ==========

def example_usage():
    """Demonstrate how to use the predictor."""
    print("This is a demonstration of the OperationTimePredictor class.")
    print("See estimator_v2.py for actual training and prediction examples.")


if __name__ == "__main__":
    example_usage()
