"""
Simplified Execution Time Estimator using is_context flag.
Uses is_context to determine whether to use attention features or just sequence length.
"""

import hashlib
import os
import pickle
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV


@dataclass
class EstimatorConfig:
    """Configuration for the execution time estimator."""

    # Database paths. Read from env vars at instantiation time so tests can
    # override; raise at runtime if both unset and not passed explicitly.
    db_path: str = ""
    cache_dir: str = ""

    # Model/Backend filtering
    model_name: Optional[str] = None  # e.g., "meta-llama/Llama-3.1-8B"
    backend: Optional[str] = None  # e.g., "FLASHINFER" or "FLASH_ATTN"

    # Model training parameters (matching Vidur)
    k_fold_cv_splits: int = 10  # Vidur default: 10
    num_training_job_threads: int = 4
    no_cache: bool = False

    # Random Forest parameters for grid search (matching Vidur)
    num_estimators: List[int] = None
    max_depth: List[int] = None
    min_samples_split: List[int] = None

    # Prediction parameters
    prediction_max_tokens: int = 16384
    prediction_max_batch_size: int = 32
    prediction_max_prefill_chunk_size: int = prediction_max_tokens
    kv_cache_granularity: int = 64  # Granularity for KV cache size predictions

    # Collective communication parameters
    train_comm_models: bool = False  # Whether to train collective communication models
    train_comm_only: bool = False  # Train ONLY comm models, skip operation models
    tp_size: Optional[int] = None  # Tensor parallel size filter
    dtype: Optional[str] = None  # Model dtype filter, as stored by the profiler (e.g. "bfloat16")
    kv_cache_dtype: Optional[str] = None  # KV-cache dtype filter (vLLM literal, e.g. "auto", "fp8")
    topology: Optional[str] = None  # Network topology filter
    collective_backend: Optional[str] = None  # Collective backend filter (e.g., "NCCL")
    embedding_dim: int = 4096  # Model embedding dimension (for calculating max size_bytes)

    def __post_init__(self):
        """Set default values for mutable parameters (matching Vidur)."""
        if self.num_estimators is None:
            self.num_estimators = [250, 500, 750]  # Vidur: [250, 500, 750]
        if self.max_depth is None:
            self.max_depth = [8, 16, 32]  # Vidur: [8, 16, 32]
        if self.min_samples_split is None:
            self.min_samples_split = [2, 5, 10]  # Vidur: [2, 5, 10]


class SimplifiedExecutionTimeEstimator:
    """
    Simplified estimator using is_context flag for feature selection.

    Feature selection logic:
    - is_context=False: Use only num_tokens (for MLP, linear, norm, etc.)
    - is_context=True: Use batch_size, kv_cache_size, prefill_chunk_size (for attention)
    """

    def __init__(self, config: EstimatorConfig = None):
        """Initialize the estimator with configuration."""
        self.config = config or EstimatorConfig()
        os.makedirs(self.config.cache_dir, exist_ok=True)

        # Will store trained models per operation signature
        self.models: Dict[str, BaseEstimator] = {}

        # Will store predictions for quick lookup
        self.predictions: Dict[str, Dict[Tuple, float]] = {}

        # Map operation names to is_context flag
        self.operation_is_context: Dict[str, bool] = {}

        # Load and prepare data
        self._load_data()

    def _load_data(self):
        """Load data from the results database."""
        conn = sqlite3.connect(self.config.db_path)

        # First, get the list of operation signatures that should be trained for this model
        model_ops_query = """
            SELECT DISTINCT s.operation_name, s.signature_hash
            FROM model_operations mo
            JOIN signatures s ON s.operation_id = mo.operation_id
            JOIN profiled_models pm ON mo.model_id = pm.model_id
        """

        model_where_conditions = []
        model_params = []

        if self.config.model_name:
            model_where_conditions.append("pm.model_name = ?")
            model_params.append(self.config.model_name)

        if self.config.backend:
            model_where_conditions.append("pm.backend = ?")
            model_params.append(self.config.backend)

        if self.config.tp_size is not None:
            model_where_conditions.append("pm.tp_degree = ?")
            model_params.append(self.config.tp_size)
        if self.config.dtype:
            model_where_conditions.append("pm.dtype = ?")
            model_params.append(self.config.dtype)
        if self.config.kv_cache_dtype:
            model_where_conditions.append("pm.kv_cache_dtype = ?")
            model_params.append(self.config.kv_cache_dtype)

        if model_where_conditions:
            model_ops_query += " WHERE " + " AND ".join(model_where_conditions)

        model_operations = pd.read_sql_query(model_ops_query, conn, params=model_params)
        signature_list = model_operations['signature_hash'].tolist()
        operation_counts = model_operations['operation_name'].value_counts()

        print(f"Found {len(signature_list)} operation signatures for model:")
        for op, count in operation_counts.items():
            print(f"  {op}: {count} signatures")

        # Now load ALL profiling data for these signatures (regardless of model/backend/tp)
        # This gives us maximum training data for each operation signature
        if not signature_list:
            print("No operation signatures found for the specified model configuration")
            self.results_df = pd.DataFrame()
        else:
            query = """
                SELECT
                    r.id, r.mean_latency_ms, r.median_latency_ms, r.max_latency_ms,
                    r.min_latency_ms, r.std_latency_ms, r.num_tokens, r.num_requests,
                    r.is_context, r.is_prefill, r.kv_cache_size, r.prefill_chunk_size,
                    r.num_tensor_parallel_workers, r.input_shapes, r.created_at,
                    s.operation_name, s.signature_hash
                FROM results r
                JOIN signatures s ON r.signature_hash = s.signature_hash
                WHERE s.signature_hash IN ({})
            """.format(','.join('?' * len(signature_list)))

            # Load results data for these signatures from ALL models/backends/tp
            self.results_df = pd.read_sql_query(query, conn, params=signature_list)

            print(f"Loaded training data:")
            for (op, sig), group_df in self.results_df.groupby(['operation_name', 'signature_hash']):
                print(f"  {op} ({sig[:8]}): {len(group_df)} samples")

        # Load collective communication data if enabled
        if self.config.train_comm_models:
            self._load_comm_data(conn)
        else:
            self.comm_df = pd.DataFrame()

        conn.close()

        # Print filtering information
        filter_info = []
        if self.config.model_name:
            filter_info.append(f"model={self.config.model_name}")
        if self.config.backend:
            filter_info.append(f"backend={self.config.backend}")
        if self.config.tp_size is not None:
            filter_info.append(f"tp={self.config.tp_size}")
        if self.config.dtype:
            filter_info.append(f"dtype={self.config.dtype}")
        if self.config.kv_cache_dtype:
            filter_info.append(f"kv_cache_dtype={self.config.kv_cache_dtype}")

        filter_str = f" (filtered by {', '.join(filter_info)})" if filter_info else ""
        print(f"Loaded {len(self.results_df)} results{filter_str}")

        if self.results_df.empty:
            raise SystemExit(
                "\nERROR: no profiled results match the requested filter.\n"
                f"  model={self.config.model}\n"
                f"  backend={self.config.backend}\n"
                f"  tp_size={self.config.tp_size}\n"
                f"  dtype={self.config.dtype}\n"
                f"  kv_cache_dtype={self.config.kv_cache_dtype}\n"
                f"  db_path={self.config.db_path}\n"
                "List what's actually in the DB with:\n"
                f"  sqlite3 {self.config.db_path} "
                "'SELECT model_name, backend, tp_degree, dtype, kv_cache_dtype FROM profiled_models;'"
            )

        # Check if new columns exist, otherwise parse from workload_params
        self._ensure_columns()

        # Group by operation AND signature to train separate models per signature
        # This is important for operations like aten::linear which have multiple
        # signatures with different matrix dimensions
        self.operation_groups = self.results_df.groupby(['operation_name', 'signature_hash'])

        # Build operation_is_context mapping from the data
        self._build_operation_is_context_mapping()

    def _load_comm_data(self, conn: sqlite3.Connection):
        """Load collective communication data from coll_op_results table."""
        try:
            # Check if table exists
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='coll_op_results'")
            if not cursor.fetchone():
                print("Note: coll_op_results table not found, skipping collective communication data")
                self.comm_df = pd.DataFrame()
                return

            # Build query with optional filtering
            # Join with model_collective_configs to get model_config_dim for num_tokens calculation
            query = """
                SELECT DISTINCT cor.*,
                       mcc.model_config_dim,
                       mcc.batch_param
                FROM coll_op_results cor
                LEFT JOIN coll_ops co ON (
                    cor.operation = co.operation_name
                    AND cor.topology = co.topology
                    AND cor.tp_size = co.tp_degree
                    AND (cor.dim_param = co.dim_param OR (cor.dim_param IS NULL AND co.dim_param IS NULL))
                    AND cor.num_dims = co.num_dims
                )
                LEFT JOIN model_collective_configs mcc ON co.collective_id = mcc.collective_id
                WHERE 1=1
            """
            params = []

            if self.config.tp_size is not None:
                query += " AND cor.tp_size = ?"
                params.append(self.config.tp_size)

            if self.config.topology is not None:
                query += " AND cor.topology = ?"
                params.append(self.config.topology)

            if self.config.collective_backend is not None:
                query += " AND cor.backend = ?"
                params.append(self.config.collective_backend)

            self.comm_df = pd.read_sql_query(query, conn, params=params)

            if len(self.comm_df) > 0:
                # Calculate num_tokens from size_bytes and model_config_dim
                # Map dtype to bytes
                dtype_to_bytes = {
                    'float32': 4,
                    'float16': 2,
                    'bfloat16': 2,
                    'int64': 8,
                    'int32': 4,
                    'int8': 1,
                }

                def calculate_num_tokens(row):
                    if pd.isna(row.get('model_config_dim')) or row.get('model_config_dim', 0) == 0:
                        # Fallback: use a default model dimension if not found
                        # This might happen for operations not in model_collective_configs
                        model_config_dim = 4096  # Default for Llama models
                    else:
                        model_config_dim = row['model_config_dim']

                    dtype_bytes = dtype_to_bytes.get(row.get('dtype', 'bfloat16'), 2)  # Default to 2 bytes
                    num_tokens = row['size_bytes'] / (model_config_dim * dtype_bytes)
                    return int(num_tokens)

                self.comm_df['num_tokens'] = self.comm_df.apply(calculate_num_tokens, axis=1)

                # New schema uses dim_param and num_dims
                # Check for legacy columns and handle accordingly
                if 'num_dims' not in self.comm_df.columns:
                    if 'ndim' in self.comm_df.columns:
                        print("  Note: Renaming 'ndim' to 'num_dims' (legacy schema)")
                        self.comm_df['num_dims'] = self.comm_df['ndim']
                    else:
                        print("  Note: Adding num_dims=2 to comm data (legacy profiling)")
                        self.comm_df['num_dims'] = 2

                # Ensure dim_param exists
                if 'dim_param' not in self.comm_df.columns:
                    print("  Note: Adding dim_param=-1 to comm data (legacy profiling)")
                    self.comm_df['dim_param'] = -1

                filter_info = []
                if self.config.tp_size is not None:
                    filter_info.append(f"tp_size={self.config.tp_size}")
                if self.config.topology is not None:
                    filter_info.append(f"topology={self.config.topology}")
                if self.config.collective_backend is not None:
                    filter_info.append(f"backend={self.config.collective_backend}")

                filter_str = f" (filtered by {', '.join(filter_info)})" if filter_info else ""
                print(f"Loaded {len(self.comm_df)} collective communication samples{filter_str}")
            else:
                print("Note: No collective communication data found with current filters")

        except Exception as e:
            print(f"Warning: Could not load comm_latencies: {e}")
            self.comm_df = pd.DataFrame()

    def _ensure_columns(self):
        """Ensure required columns exist, parsing from workload_params if needed."""

        # Check if new column structure exists
        if 'is_context' not in self.results_df.columns:
            print("Note: is_context column not found. Parsing from workload_params...")

            # Parse workload_params to create columns
            import json

            for idx, row in self.results_df.iterrows():
                if pd.notna(row.get('workload_params')):
                    try:
                        params = json.loads(row['workload_params'])

                        # Set is_context based on operation name (fallback method)
                        op_lower = row['operation_name'].lower() if 'operation_name' in row else ''
                        self.results_df.loc[idx, 'is_context'] = 'attention' in op_lower or 'attn' in op_lower

                        # Extract other parameters
                        self.results_df.loc[idx, 'num_tokens'] = params.get('num_tokens', 0)
                        self.results_df.loc[idx, 'batch_size'] = params.get('batch_size', params.get('num_requests', 1))
                        self.results_df.loc[idx, 'is_prefill'] = params.get('is_prefill', False)
                        self.results_df.loc[idx, 'kv_cache_size'] = params.get('kv_cache_size', 0)
                        self.results_df.loc[idx, 'prefill_chunk_size'] = params.get('prefill_chunk_size', 0)

                    except Exception as e:
                        print(f"Warning: Could not parse workload_params at index {idx}: {e}")

    def _build_operation_is_context_mapping(self):
        """Build a mapping of operation_name -> is_context flag."""
        if 'is_context' not in self.results_df.columns:
            print("Warning: is_context column not found, mapping will be empty")
            return

        # For each operation, determine if it's a context operation
        # Note: operation_groups now returns (operation_name, signature_hash), group_df
        for (op_name, sig_hash), group_df in self.operation_groups:
            # Take the first value (should be consistent across the operation)
            if len(group_df) > 0 and 'is_context' in group_df.columns:
                is_context_value = group_df['is_context'].iloc[0]
                # Handle different representations of boolean
                if pd.notna(is_context_value):
                    self.operation_is_context[op_name] = bool(is_context_value)
                else:
                    self.operation_is_context[op_name] = False
            else:
                self.operation_is_context[op_name] = False

        print(f"Built operation_is_context mapping for {len(self.operation_is_context)} unique operations")
        # Show which operations are context operations
        context_ops = [op for op, is_ctx in self.operation_is_context.items() if is_ctx]
        if context_ops:
            print(f"  Context operations: {context_ops}")

    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Determine feature columns based on is_context flag.

        Simple rule:
        - is_context=False: Use only num_tokens
        - is_context=True: Use attention-specific features
        """

        if 'is_context' not in df.columns:
            # Fallback: assume non-context if column doesn't exist
            return ['num_tokens'] if 'num_tokens' in df.columns else []

        # Get the is_context value (should be same for all rows in this operation)
        is_context = df['is_context'].iloc[0] if len(df) > 0 else False

        if is_context:
            # Context-dependent operation (attention)
            # Check if prefill or decode
            is_prefill = df['is_prefill'].iloc[0] if 'is_prefill' in df.columns and len(df) > 0 else None

            if is_prefill == 1 or is_prefill == 1.0 or is_prefill is True:
                # Prefill attention: use kv_cache_size and prefill_chunk_size^2
                feature_cols = []
                if 'kv_cache_size' in df.columns:
                    feature_cols.append('kv_cache_size')
                if 'prefill_chunk_size' in df.columns:
                    df['prefill_chunk_size_squared'] = df['prefill_chunk_size'] ** 2
                    feature_cols.append('prefill_chunk_size_squared')
                return feature_cols if feature_cols else ['num_tokens']

            elif is_prefill == 0 or is_prefill == 0.0 or is_prefill is False:
                # Decode attention: use batch_size and kv_cache_size
                feature_cols = []

                # Use batch_size if available, otherwise num_requests as proxy
                if 'batch_size' in df.columns:
                    feature_cols.append('batch_size')
                elif 'num_requests' in df.columns:
                    # Use num_requests as batch_size for decode
                    df['batch_size'] = df['num_requests']
                    feature_cols.append('batch_size')

                if 'kv_cache_size' in df.columns:
                    feature_cols.append('kv_cache_size')

                return feature_cols if feature_cols else ['num_tokens']

            else:
                # Mixed or unknown: use all available context features
                feature_cols = []
                if 'batch_size' in df.columns:
                    feature_cols.append('batch_size')
                if 'num_tokens' in df.columns:
                    feature_cols.append('num_tokens')
                return feature_cols
        else:
            # Non-context operation (MLP, linear, norm, etc.)
            # Use only sequence length
            return ['num_tokens'] if 'num_tokens' in df.columns else []

    def train_all_models(self):
        """Train models for all operation signatures in the database."""

        # If train_comm_only is set, skip operation models and only train comm
        if self.config.train_comm_only:
            print("\n=== Training ONLY Collective Communication Models ===")
            if self.config.train_comm_models:
                self._train_comm_models()
            else:
                print("Warning: train_comm_only is set but train_comm_models is False")
                return
        else:
            print("\n=== Training or Loading Models for All Operation Signatures ===")

            # Note: operation_groups now returns (operation_name, signature_hash), group_df
            for (operation_name, sig_hash), group_df in self.operation_groups:
                # Create a unique model name that includes signature hash (first 8 chars)
                sig_short = sig_hash[:8]
                model_key = f"{operation_name}_{sig_short}"

                print(f"\nProcessing: {operation_name} (sig: {sig_short}, {len(group_df)} samples)")

                # Check if this is a context operation
                is_context = group_df['is_context'].iloc[0] if 'is_context' in group_df else False

                # Special handling for context operations - split prefill and decode
                if is_context and 'is_prefill' in group_df.columns:
                    # Get unique values of is_prefill
                    prefill_values = group_df['is_prefill'].unique()

                    if len(prefill_values) > 1:
                        # We have both prefill and decode data
                        prefill_df = group_df[group_df['is_prefill'] == 1.0].copy()
                        decode_df = group_df[group_df['is_prefill'] == 0.0].copy()

                        # Train separate models with signature hash included
                        if len(prefill_df) >= 5:
                            prefill_model_name = f"{operation_name}_prefill_{sig_short}"
                            self._train_operation_model(prefill_model_name, prefill_df)
                            # Mark as context operation
                            self.operation_is_context[prefill_model_name] = True

                        if len(decode_df) >= 5:
                            decode_model_name = f"{operation_name}_decode_{sig_short}"
                            self._train_operation_model(decode_model_name, decode_df)
                            # Mark as context operation
                            self.operation_is_context[decode_model_name] = True

                        continue  # Skip the unified model

                # Train unified model for this operation signature
                self._train_operation_model(model_key, group_df)

            # Train collective communication models if enabled
            if self.config.train_comm_models:
                self._train_comm_models()

        print(f"\n=== Training Complete: {len(self.models)} models trained ===")

        # Automatically save predictions to CSV cache
        print(f"\n=== Saving Predictions to CSV Cache ===")
        from doolysim.core.prediction_cache import generate_prediction_cache_from_estimator
        generate_prediction_cache_from_estimator(
            estimator=self,
            cache_dir=self.config.cache_dir,
            force_regenerate=False  # Don't overwrite existing CSV files
        )
        print(f"CSV cache saved to {self.config.cache_dir}")

    def _get_model_hash(self, model_name: str, df: pd.DataFrame) -> str:
        """
        Generate a hash for model caching based on config and data.

        Args:
            model_name: Name of the operation/model
            df: Training dataframe

        Returns:
            8-character hash string
        """
        # Create a config string with relevant parameters
        config_dict = {
            'model_name': self.config.model_name,
            'backend': self.config.backend,
            'num_estimators': self.config.num_estimators,
            'max_depth': self.config.max_depth,
            'min_samples_split': self.config.min_samples_split,
            'k_fold_cv_splits': self.config.k_fold_cv_splits,
        }
        config_str = str(config_dict)

        # Hash the data
        df_hash_str = hashlib.md5(df.to_json().encode("utf-8")).hexdigest()
        combined_str = f"{config_str}_{model_name}_{df_hash_str}"

        return hashlib.md5(combined_str.encode("utf-8")).hexdigest()[:8]

    def _load_model_from_cache(self, model_name: str, model_hash: str) -> Optional[BaseEstimator]:
        """
        Load a trained model from cache.

        Args:
            model_name: Name of the operation/model
            model_hash: Hash of the model configuration and data

        Returns:
            Loaded model or None if not found or caching disabled
        """
        if self.config.no_cache:
            return None

        cache_file = os.path.join(self.config.cache_dir, f"{model_name}_{model_hash}.pkl")
        print(cache_file)

        if not os.path.exists(cache_file):
            return None

        try:
            with open(cache_file, "rb") as f:
                model = pickle.load(f)
            print(f"    Loaded {model_name} from cache")
            return model
        except Exception as e:
            print(f"    Warning: Could not load {model_name} from cache: {e}")
            return None

    def _store_model_in_cache(self, model_name: str, model_hash: str, model: BaseEstimator) -> None:
        """
        Store a trained model in cache.

        Args:
            model_name: Name of the operation/model
            model_hash: Hash of the model configuration and data
            model: Trained model to cache
        """
        if self.config.no_cache:
            return

        cache_file = os.path.join(self.config.cache_dir, f"{model_name}_{model_hash}.pkl")

        try:
            with open(cache_file, "wb") as f:
                pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"    Warning: Could not cache {model_name}: {e}")

    def _train_comm_models(self):
        """Train models for collective communication operations.

        Uses operation-specific features:
        - all_reduce: num_tokens (derived from size_bytes / model_config_dim / dtype_bytes)
        - all_gather: num_requests (derived from size_bytes / model_config_dim / dtype_bytes)

        Groups by (operation, dim_param, num_dims) to capture different memory access patterns.
        """
        if not hasattr(self, 'comm_df') or len(self.comm_df) == 0:
            print("\n=== No collective communication data to train ===")
            return

        print("\n=== Training Collective Communication Models ===")

        # Group by (operation, dim_param, num_dims) to capture different memory patterns
        # Note: Use dropna=False to include NULL dim_param values (for all_reduce)
        grouping_cols = ['operation', 'dim_param', 'num_dims']

        for group_key, op_df in self.comm_df.groupby(grouping_cols, dropna=False):
            operation, dim_param, num_dims = group_key

            # Handle NULL dim_param (e.g., all_reduce has no dimension parameter)
            dim_str = "None" if pd.isna(dim_param) else str(int(dim_param))
            model_key = f"{operation}_dim{dim_str}_ndims{int(num_dims)}"

            if len(op_df) < 5:
                print(f"  Skipping {model_key}: insufficient data ({len(op_df)} samples)")
                continue

            print(f"\nProcessing: {model_key} ({len(op_df)} samples)")

            # Determine feature based on operation type
            # Use num_tokens as the feature instead of size_bytes
            feature_cols = ['num_tokens']
            target_col = 'latency_ms'

            # Store metadata about what this operation uses
            # all_reduce typically uses NUM_TOKS, all_gather uses NUM_REQS
            batch_param = "num_tokens" if "reduce" in operation.lower() else "num_requests"

            # Prepare training data
            train_df = op_df[feature_cols + [target_col]].copy()
            train_df = train_df.dropna()

            if len(train_df) < 5:
                print(f"  Skipping {model_key}: insufficient data after cleanup ({len(train_df)} samples)")
                continue

            # Generate hash for caching
            model_hash = self._get_model_hash(model_key, train_df)

            # Try to load from cache
            cached_model = self._load_model_from_cache(model_key, model_hash)
            if cached_model is not None:
                self.models[model_key] = cached_model
                # Still generate predictions for the cached model
                # Convert NaN to None for consistency
                dim_param_val = None if pd.isna(dim_param) else int(dim_param)
                self._generate_comm_predictions(model_key, cached_model, feature_cols, dim_param_val, int(num_dims))
                continue

            print(f"  Training {model_key} with features: {feature_cols}")

            try:
                model = self._train_model(
                    operation_name=model_key,
                    df=train_df,
                    feature_cols=feature_cols,
                    target_col=target_col
                )
                self.models[model_key] = model

                # Store in cache
                self._store_model_in_cache(model_key, model_hash, model)

                # Generate predictions for common input ranges
                # Convert NaN to None for consistency
                dim_param_val = None if pd.isna(dim_param) else int(dim_param)
                self._generate_comm_predictions(model_key, model, feature_cols, dim_param_val, int(num_dims))

            except Exception as e:
                print(f"  Error training model for {model_key}: {e}")
                import traceback
                traceback.print_exc()

    def _generate_comm_predictions(
        self,
        operation_name: str,
        model: BaseEstimator,
        feature_cols: List[str],
        dim_param: Optional[int],
        num_dims: int
    ):
        """
        Generate predictions for collective communication operations.

        Uses features: num_tokens
        Predictions are indexed by (size_bytes,) for lookup.
        """
        # Generate num_tokens range for prediction
        min_tokens = 1
        max_tokens = self.config.prediction_max_tokens

        # Create a range of num_tokens values with increment of 1
        tokens_range = np.arange(min_tokens, max_tokens + 1, 1)

        # Create prediction inputs using num_tokens
        predictions_list = [{'num_tokens': num_tokens} for num_tokens in tokens_range]

        # Create DataFrame and generate predictions
        X = pd.DataFrame(predictions_list)
        predictions = model.predict(X[feature_cols])

        # Store predictions with num_tokens as key
        pred_dict = {}

        for idx, row in X.iterrows():
            num_tokens = int(row['num_tokens'])
            # Store by num_tokens directly
            pred_dict[num_tokens] = predictions[idx]

        self.predictions[operation_name] = pred_dict
        print(f"    Generated {len(pred_dict)} predictions for {operation_name}")
        print(f"      Token range: {min_tokens} to {max_tokens} tokens")

    def _train_operation_model(self, operation_name: str, df: pd.DataFrame):
        """Train a model for a specific operation."""

        # Get features based on is_context flag
        feature_cols = self._get_feature_columns(df)
        target_col = 'median_latency_ms'

        if not feature_cols:
            print(f"  Skipping {operation_name}: no valid features found")
            return

        # Prepare training data
        available_cols = [c for c in feature_cols + [target_col] if c in df.columns]
        train_df = df[available_cols].copy()
        train_df = train_df.dropna()

        # Update feature_cols to only include available columns
        feature_cols = [c for c in feature_cols if c in train_df.columns and c != target_col]

        if len(train_df) < 5:
            print(f"  Skipping {operation_name}: insufficient data ({len(train_df)} samples)")
            return

        # Generate hash for caching
        model_hash = self._get_model_hash(operation_name, train_df)

        # Try to load from cache
        cached_model = self._load_model_from_cache(operation_name, model_hash)
        if cached_model is not None:
            self.models[operation_name] = cached_model
            # Still generate predictions for the cached model
            self._generate_predictions(operation_name, cached_model, feature_cols,
                                      is_context='is_context' in df and df['is_context'].iloc[0])
            return

        print(f"  Training {operation_name} with features: {feature_cols}")

        try:
            model = self._train_model(
                operation_name=operation_name,
                df=train_df,
                feature_cols=feature_cols,
                target_col=target_col
            )
            self.models[operation_name] = model

            # Store in cache
            self._store_model_in_cache(operation_name, model_hash, model)

            # Generate predictions for common input ranges
            self._generate_predictions(operation_name, model, feature_cols,
                                      is_context='is_context' in df and df['is_context'].iloc[0])

        except Exception as e:
            print(f"  Error training model for {operation_name}: {e}")
            import traceback
            traceback.print_exc()

    def _train_model(
        self,
        operation_name: str,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str
    ) -> BaseEstimator:
        """Train a Random Forest model for specific operation and features."""

        # Prepare model and grid search
        model = RandomForestRegressor(random_state=42)

        grid_params = {
            'n_estimators': self.config.num_estimators,
            'max_depth': self.config.max_depth,
            'min_samples_split': self.config.min_samples_split,
        }

        scorer = make_scorer(
            self.mean_absolute_percentage_error,
            greater_is_better=False
        )

        cv_splits = min(self.config.k_fold_cv_splits, len(df))
        if cv_splits < 2:
            cv_splits = 2

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=grid_params,
            scoring=scorer,
            cv=cv_splits,
            n_jobs=self.config.num_training_job_threads,
            verbose=0
        )

        # Train
        X = df[feature_cols]
        y = df[target_col]

        grid_search.fit(X, y)
        best_model = grid_search.best_estimator_
        score = -grid_search.best_score_

        print(f"    Best params: {grid_search.best_params_}")
        print(f"    MAPE: {score:.2f}%")

        return best_model

    @staticmethod
    def mean_absolute_percentage_error(y_true: np.array, y_pred: np.array) -> float:
        """Calculate MAPE for model evaluation."""
        y_true, y_pred = np.array(y_true), np.array(y_pred)

        # Handle zero values
        zero_mask = y_true == 0
        non_zero_mask = ~zero_mask

        error = np.zeros_like(y_true, dtype=float)
        error[non_zero_mask] = np.abs(
            (y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask]
        ) * 100

        error[zero_mask] = np.where(y_pred[zero_mask] == 0, 0, 100)

        return np.mean(error)

    def _generate_predictions(
        self,
        operation_name: str,
        model: BaseEstimator,
        feature_cols: List[str],
        is_context: bool = False
    ):
        """
        Generate predictions for common input ranges.

        Matches Vidur's approach: generate predictions for EVERY value in range
        (not with gaps).
        """

        if not is_context:
            # Non-context: generate predictions for EVERY num_tokens (like Vidur)
            if feature_cols == ['num_tokens']:
                num_tokens_range = np.arange(1, self.config.prediction_max_tokens + 1)
                X = pd.DataFrame({'num_tokens': num_tokens_range})

                predictions = model.predict(X)
                pred_dict = {(tokens,): pred for tokens, pred in zip(num_tokens_range, predictions)}
                self.predictions[operation_name] = pred_dict
                print(f"    Generated {len(pred_dict)} predictions")

        else:
            # Context operation: generate appropriate prediction grids
            if 'prefill_chunk_size_squared' in feature_cols:
                # Prefill predictions (like Vidur)
                # kv_cache: with granularity, prefill_chunk: EVERY value from 1 to max
                kv_cache_range = np.arange(0, self.config.prediction_max_tokens + 1,
                                          self.config.kv_cache_granularity)
                prefill_chunk_range = np.arange(1, self.config.prediction_max_prefill_chunk_size + 1)

                kv_cache, prefill_chunk = np.meshgrid(kv_cache_range, prefill_chunk_range)
                X = pd.DataFrame({
                    'kv_cache_size': kv_cache.flatten(),
                    'prefill_chunk_size_squared': (prefill_chunk.flatten()) ** 2
                })

            elif set(feature_cols) == {'batch_size', 'kv_cache_size'}:
                # Decode predictions (like Vidur)
                # batch_size: EVERY value from 1 to max, kv_cache: with granularity
                batch_range = np.arange(1, self.config.prediction_max_batch_size + 1)
                kv_cache_range = np.arange(0, self.config.prediction_max_tokens + 1,
                                          self.config.kv_cache_granularity)

                batch, kv_cache = np.meshgrid(batch_range, kv_cache_range)
                X = pd.DataFrame({
                    'batch_size': batch.flatten(),
                    'kv_cache_size': kv_cache.flatten()
                })
            else:
                print(f"    Skipping prediction generation for {operation_name}: unusual features {feature_cols}")
                return

            predictions = model.predict(X)
            pred_dict = {}
            for idx, row in X.iterrows():
                key = tuple(row[col] for col in feature_cols)
                pred_dict[key] = predictions[idx]

            self.predictions[operation_name] = pred_dict
            print(f"    Generated {len(pred_dict)} predictions")

    def predict(
        self,
        operation_name: str,
        **kwargs
    ) -> float:
        """
        Predict execution time for an operation.

        Args:
            operation_name: Name of the operation (possibly with _prefill/_decode suffix)
            **kwargs: Feature values (e.g., num_tokens=128, batch_size=4, kv_cache_size=512)
                     For comm operations: size_bytes and ndim (or num_tokens which gets converted)

        Returns:
            Predicted execution time in milliseconds
        """

        if operation_name not in self.models:
            raise ValueError(f"No model trained for operation: {operation_name}")

        model = self.models[operation_name]

        # Check if this is a collective communication operation
        comm_ops = ['all_reduce', 'all_gather', 'reduce_scatter']
        is_comm_op = operation_name in comm_ops

        if is_comm_op:
            # Collective communication operations use size_bytes + dim_param + dim0 + dim1
            # Convert num_tokens to size_bytes if needed
            if 'size_bytes' not in kwargs and 'num_tokens' in kwargs:
                # size_bytes = num_tokens * embedding_dim * 2 (for float16)
                kwargs['size_bytes'] = kwargs['num_tokens'] * self.config.embedding_dim * 2

            # Default values for tensor shape
            if 'dim_param' not in kwargs:
                kwargs['dim_param'] = -1  # Default to last dimension
            if 'dim0' not in kwargs:
                # For 1D tensor: dim0 = num_elements
                num_elements = kwargs['size_bytes'] // 2
                kwargs['dim0'] = num_elements
            if 'dim1' not in kwargs:
                kwargs['dim1'] = -1  # -1 indicates 1D tensor

            feature_names = ['size_bytes', 'dim_param', 'dim0', 'dim1']
        else:
            # Regular operations - determine features from predictions or operation type
            if operation_name in self.predictions and len(self.predictions[operation_name]) > 0:
                # Get feature names from a sample prediction key
                sample_key = next(iter(self.predictions[operation_name].keys()))
                num_features = len(sample_key)

                # Map common features based on number and operation name
                if '_prefill' in operation_name:
                    feature_names = ['kv_cache_size', 'prefill_chunk_size_squared']
                    # Handle prefill_chunk_size -> prefill_chunk_size_squared
                    if 'prefill_chunk_size' in kwargs and 'prefill_chunk_size_squared' not in kwargs:
                        kwargs['prefill_chunk_size_squared'] = kwargs['prefill_chunk_size'] ** 2
                elif '_decode' in operation_name:
                    feature_names = ['batch_size', 'kv_cache_size']
                else:
                    feature_names = ['num_tokens']
            else:
                # Fallback
                feature_names = ['num_tokens']

        # Prepare input
        features = []
        for col in feature_names:
            if col not in kwargs:
                raise ValueError(f"Missing required feature '{col}' for operation {operation_name}")
            features.append(kwargs[col])

        # Check if we have pre-computed prediction
        feature_tuple = tuple(int(f) for f in features)  # Convert to int for consistent keys
        if operation_name in self.predictions and feature_tuple in self.predictions[operation_name]:
            return self.predictions[operation_name][feature_tuple]

        # Otherwise, predict on the fly
        X = pd.DataFrame([{col: kwargs[col] for col in feature_names}])
        prediction = model.predict(X)[0]

        return prediction

    def get_operation_stats(self) -> pd.DataFrame:
        """Get statistics about trained operations."""

        stats = []
        for (op_name, sig_hash), group_df in self.operation_groups:
            is_context = group_df['is_context'].iloc[0] if 'is_context' in group_df else False
            sig_short = sig_hash[:8]
            model_key = f"{op_name}_{sig_short}"

            stats.append({
                'operation': op_name,
                'signature': sig_short,
                'is_context': is_context,
                'num_samples': len(group_df),
                'min_tokens': group_df['num_tokens'].min() if 'num_tokens' in group_df else None,
                'max_tokens': group_df['num_tokens'].max() if 'num_tokens' in group_df else None,
                'mean_latency': group_df['median_latency_ms'].mean(),
                'std_latency': group_df['median_latency_ms'].std(),
                'has_model': model_key in self.models or
                           f"{op_name}_prefill" in self.models or
                           f"{op_name}_decode" in self.models
            })

        return pd.DataFrame(stats)

def main(model_name=None, backend=None, db_path=None, cache_dir=None,
         max_tokens=16384, max_batch_size=32,
         train_comm=False, train_comm_only=False, tp_size=None, embedding_dim=4096,
         topology="DGX", collective_backend="NCCL", dtype=None, kv_cache_dtype=None):

    # Initialize estimator with optional filtering. Resolution order:
    # explicit arg > DOOLYSIM_CACHE_DIR/DB_PATH env var > error.
    resolved_cache = cache_dir or os.environ.get("DOOLYSIM_CACHE_DIR")
    resolved_db = db_path or os.environ.get("DOOLYSIM_DB_PATH")
    if not resolved_cache:
        raise ValueError("cache_dir must be provided (or set DOOLYSIM_CACHE_DIR env var)")
    if not resolved_db:
        raise ValueError("db_path must be provided (or set DOOLYSIM_DB_PATH env var)")
    config = EstimatorConfig(
        cache_dir=resolved_cache,
        model_name=model_name,
        backend=backend,
        db_path=resolved_db,
        no_cache=False,
        train_comm_models=train_comm or train_comm_only,  # Enable if either flag is set
        train_comm_only=train_comm_only,
        tp_size=tp_size,
        dtype=dtype,
        kv_cache_dtype=kv_cache_dtype,
        embedding_dim=embedding_dim,
        topology=topology,
        collective_backend=collective_backend,
        prediction_max_tokens=max_tokens,
        prediction_max_batch_size=max_batch_size
    )

    print(f"\n=== Initializing Simplified Estimator ===")
    print(f"Using is_context flag for feature selection")
    if model_name:
        print(f"Model filter: {model_name}")
    if backend:
        print(f"Backend filter: {backend}")
    if dtype:
        print(f"dtype filter: {dtype}")
    if kv_cache_dtype:
        print(f"KV-cache dtype filter: {kv_cache_dtype}")

    estimator = SimplifiedExecutionTimeEstimator(config)

    # Train all models
    estimator.train_all_models()

    # Show statistics
    print("\n=== Operation Statistics ===")
    stats = estimator.get_operation_stats()
    print(stats.to_string())

    return estimator


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train simplified execution time estimator")
    parser.add_argument("--model", type=str, default=None,
                       help="Model name to filter (e.g., 'meta-llama/Llama-3.1-8B')")
    parser.add_argument("--backend", type=str, default=None,
                       help="Backend to filter (e.g., 'FLASHINFER' or 'FLASH_ATTN')")
    parser.add_argument("--db-path", type=str, default=os.environ.get("DOOLYSIM_DB_PATH"),
                       help="Path to SQLite database for results (or set DOOLYSIM_DB_PATH env var)")
    parser.add_argument("--cache-dir", type=str, default=os.environ.get("DOOLYSIM_CACHE_DIR"),
                       help="Directory for caching trained models (or set DOOLYSIM_CACHE_DIR env var)")
    parser.add_argument("--max-batch-size", type=int, default=32)
    parser.add_argument("--max-token-count", type=int, default=16384)
    parser.add_argument("--train-comm", action="store_true", help="Train collective communication models")
    parser.add_argument("--train-comm-only", action="store_true", help="Train ONLY comm models, skip operation models")
    parser.add_argument("--tp-size", type=int, default=None, help="Filter comm data by tensor parallel size")
    parser.add_argument("--dtype", type=str, default=None,
                        help="Filter by model dtype as stored by the profiler (e.g. bfloat16, float16)")
    parser.add_argument("--kv-cache-dtype", type=str, default=None,
                        help="Filter by KV-cache dtype (vLLM literal: auto, fp8, fp8_e4m3, fp8_e5m2)")
    parser.add_argument("--embedding-dim", type=int, default=4096, help="Model embedding dimension (for max size_bytes calculation)")
    parser.add_argument("--topology", type=str, default="mew1", help="Filter comm data by topology (e.g., 'DGX', 'NVLINK')")
    parser.add_argument("--collective-backend", type=str, default="NCCL", help="Collective communication backend (e.g., 'NCCL')")

    args = parser.parse_args()
    
    estimator = main(
        model_name=args.model,
        backend=args.backend,
        db_path=args.db_path,
        cache_dir=args.cache_dir,
        train_comm=args.train_comm,
        train_comm_only=args.train_comm_only,
        tp_size=args.tp_size,
        dtype=args.dtype,
        kv_cache_dtype=args.kv_cache_dtype,
        embedding_dim=args.embedding_dim,
        topology=args.topology,
        collective_backend=args.collective_backend,
        max_tokens=args.max_token_count,
        max_batch_size=args.max_batch_size
    )