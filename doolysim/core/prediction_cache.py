"""
CSV-based prediction cache for fast latency lookups.

This module generates and loads prediction caches from CSV files instead of
requiring model loading and inference on every simulator run.
"""

import os
import csv
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional


class PredictionCache:
    """Manages CSV-based prediction caches for latency estimation."""

    def __init__(self, cache_dir: str):
        """Initialize prediction cache.

        Args:
            cache_dir: Directory to store/load CSV prediction files
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        # Dictionary of operation_name -> feature_tuple -> prediction
        self.predictions: Dict[str, Dict[Tuple, float]] = {}

    def get_cache_path(self, operation_name: str) -> str:
        """Get path to cache file for an operation.

        Args:
            operation_name: Name of the operation

        Returns:
            Path to CSV cache file
        """
        # Sanitize operation name for filename
        safe_name = operation_name.replace('/', '_').replace(':', '_')
        return os.path.join(self.cache_dir, f"{safe_name}_predictions.csv")

    def has_cache(self, operation_name: str) -> bool:
        """Check if cache exists for an operation.

        Args:
            operation_name: Name of the operation

        Returns:
            True if cache file exists
        """
        return os.path.exists(self.get_cache_path(operation_name))

    def save_predictions(
        self,
        operation_name: str,
        predictions: Dict[Tuple, float],
        feature_names: list
    ):
        """Save predictions to CSV file.

        Args:
            operation_name: Name of the operation
            predictions: Dictionary of feature_tuple -> prediction
            feature_names: List of feature column names
        """
        cache_path = self.get_cache_path(operation_name)

        with open(cache_path, 'w', newline='') as f:
            writer = csv.writer(f)

            # Write header
            header = feature_names + ['prediction_ms']
            writer.writerow(header)

            # Write predictions
            for feature_tuple, prediction in predictions.items():
                # Handle both tuple keys (regular ops) and int keys (collective ops)
                if isinstance(feature_tuple, tuple):
                    row = list(feature_tuple) + [prediction]
                else:
                    # Single feature (e.g., size_bytes for collective ops)
                    row = [feature_tuple, prediction]
                writer.writerow(row)

        print(f"  Saved {len(predictions)} predictions to {cache_path}")

    def load_predictions(self, operation_name: str) -> Optional[Dict[Tuple, float]]:
        """Load predictions from CSV file.

        Args:
            operation_name: Name of the operation

        Returns:
            Dictionary of feature_tuple -> prediction, or None if cache doesn't exist
        """
        cache_path = self.get_cache_path(operation_name)

        if not os.path.exists(cache_path):
            return None

        predictions = {}

        with open(cache_path, 'r', newline='') as f:
            reader = csv.reader(f)

            # Read header
            header = next(reader)
            num_features = len(header) - 1  # Exclude 'prediction_ms' column

            # Read predictions
            for row in reader:
                # Parse feature values (convert to int or float as appropriate)
                features = []
                for i in range(num_features):
                    val = row[i]
                    # Try to convert to int first, then float
                    try:
                        features.append(int(val))
                    except ValueError:
                        features.append(float(val))

                feature_tuple = tuple(features)
                prediction = float(row[-1])
                predictions[feature_tuple] = prediction

        print(f"  Loaded {len(predictions)} predictions from {cache_path}")
        return predictions

    def get_or_generate(
        self,
        operation_name: str,
        generator_func,
        force_regenerate: bool = False
    ) -> Dict[Tuple, float]:
        """Get predictions from cache, or generate and cache them.

        Args:
            operation_name: Name of the operation
            generator_func: Function that returns (predictions_dict, feature_names)
            force_regenerate: Force regeneration even if cache exists

        Returns:
            Dictionary of feature_tuple -> prediction
        """
        # Check if we already loaded this operation
        if operation_name in self.predictions and not force_regenerate:
            return self.predictions[operation_name]

        # Try to load from cache
        if not force_regenerate:
            cached_predictions = self.load_predictions(operation_name)
            if cached_predictions is not None:
                self.predictions[operation_name] = cached_predictions
                return cached_predictions

        # Generate predictions
        print(f"  Generating predictions for {operation_name}...")
        predictions, feature_names = generator_func()

        # Save to cache
        self.save_predictions(operation_name, predictions, feature_names)

        # Store in memory
        self.predictions[operation_name] = predictions

        return predictions


def generate_prediction_cache_from_estimator(
    estimator,
    cache_dir: str,
    force_regenerate: bool = False
):
    """Generate CSV prediction caches from a trained estimator.

    Args:
        estimator: Trained SimplifiedExecutionTimeEstimator
        cache_dir: Directory to save CSV files
        force_regenerate: Force regeneration even if cache exists
    """
    print("\n" + "=" * 80)
    print("GENERATING PREDICTION CACHE FILES")
    print("=" * 80)

    cache = PredictionCache(cache_dir)

    for operation_name, model in estimator.models.items():
        print(f"\nProcessing {operation_name}...")

        # Skip if cache exists and not forcing regeneration
        if not force_regenerate and cache.has_cache(operation_name):
            print(f"  Cache already exists, skipping")
            continue

        # Check if predictions were generated during training
        if operation_name not in estimator.predictions:
            print(f"  No predictions available, skipping")
            continue

        predictions = estimator.predictions[operation_name]
        if len(predictions) == 0:
            print(f"  Empty predictions, skipping")
            continue

        # Infer feature names from operation type
        sample_key = next(iter(predictions.keys()))

        # Check for collective operations first
        if operation_name.startswith(('all_reduce_', 'all_gather_', 'reduce_scatter_')):
            # Collective communication operations now use num_tokens
            feature_names = ['num_tokens']
        elif '_prefill' in operation_name:
            feature_names = ['kv_cache_size', 'prefill_chunk_size_squared']
        elif '_decode' in operation_name:
            feature_names = ['batch_size', 'kv_cache_size']
        else:
            feature_names = ['num_tokens']

        # Save to cache
        cache.save_predictions(operation_name, predictions, feature_names)

    print("\n" + "=" * 80)
    print(f"CACHE GENERATION COMPLETE: {len(estimator.models)} operations")
    print(f"Cache directory: {cache_dir}")
    print("=" * 80)


def load_all_predictions(cache_dir: str) -> PredictionCache:
    """Load all prediction caches from a directory.

    Args:
        cache_dir: Directory containing CSV prediction files

    Returns:
        PredictionCache with all predictions loaded
    """
    print(f"\nLoading prediction caches from {cache_dir}...")

    cache = PredictionCache(cache_dir)

    if not os.path.exists(cache_dir):
        print(f"Warning: Cache directory does not exist: {cache_dir}")
        return cache

    # Find all CSV files
    csv_files = list(Path(cache_dir).glob("*_predictions.csv"))

    if len(csv_files) == 0:
        print(f"Warning: No prediction cache files found in {cache_dir}")
        return cache

    for csv_file in csv_files:
        # Extract operation name from filename - keep underscores!
        # Format: operation_name_hash_predictions.csv
        operation_name = csv_file.stem.replace('_predictions', '')

        # Restore original operation name format (convert __ back to ::)
        # e.g., "aten__linear_169ab818" -> "aten::linear_169ab818"
        # But preserve single underscores (e.g., "_C__fused_add_rms_norm" -> "_C::fused_add_rms_norm")
        # Strategy: Replace double underscore with ::, but only if it's likely a namespace separator
        # Look for pattern: word__word -> word::word
        import re
        # Match double underscores that are preceded and followed by alphanumeric characters
        operation_name = re.sub(r'([a-zA-Z0-9])__([a-zA-Z0-9])', r'\1::\2', operation_name)

        # Load directly from file (don't use load_predictions which modifies the name)
        predictions = {}

        with open(str(csv_file), 'r', newline='') as f:
            reader = csv.reader(f)

            # Read header
            header = next(reader)
            num_features = len(header) - 1  # Exclude 'prediction_ms' column

            # Read predictions
            for row in reader:
                # Parse feature values (convert to int or float as appropriate)
                features = []
                for i in range(num_features):
                    val = row[i]
                    # Try to convert to int first, then float
                    try:
                        features.append(int(val))
                    except ValueError:
                        features.append(float(val))

                feature_tuple = tuple(features)
                prediction = float(row[-1])
                predictions[feature_tuple] = prediction

        if predictions:
            cache.predictions[operation_name] = predictions
            print(f"  Loaded {len(predictions)} predictions from {csv_file} as {operation_name}")

    print(f"Loaded {len(cache.predictions)} operation caches")

    return cache
