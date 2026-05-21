from dataclasses import dataclass, field
from typing import List, Dict, Optional
import csv
import os
import json 

@dataclass
class TokenEvent:
    """Record of a single token generation event."""
    token_index: int  # 0 = first token, 1 = second token, etc.
    timestamp_ms: float  # Simulation time when token was generated (milliseconds)
    step_duration_ms: float  # Duration of the step that generated this token (milliseconds)
    batch_size: int  # Number of requests in the batch
    total_tokens_in_batch: int  # Total tokens generated in this step

@dataclass
class RequestTrace:
    """Detailed trace of a single request with token-level tracking."""
    request_id: str
    arrival_time_ms: float  # Milliseconds
    prompt_len: int
    max_output_len: int

    # Token generation events
    token_events: List[TokenEvent] = field(default_factory=list)

    # Scheduling info
    scheduled_time_ms: Optional[float] = None  # When request was first scheduled (started processing)

    # Completion info
    finish_time_ms: Optional[float] = None  # Milliseconds
    finish_reason: Optional[str] = None

    @property
    def actual_output_len(self) -> int:
        """Number of tokens actually generated."""
        return len(self.token_events)

    @property
    def queue_time_ms(self) -> Optional[float]:
        """Time spent waiting in queue before processing starts (milliseconds)."""
        if self.scheduled_time_ms is None:
            return None
        return self.scheduled_time_ms - self.arrival_time_ms

    @property
    def ttft_ms(self) -> Optional[float]:
        """Time to first token (milliseconds)."""
        if not self.token_events:
            return None
        return self.token_events[0].timestamp_ms - self.arrival_time_ms

    @property
    def total_latency_ms(self) -> Optional[float]:
        """Total end-to-end latency (milliseconds)."""
        if self.finish_time_ms is None:
            return None
        return self.finish_time_ms - self.arrival_time_ms

    @property
    def inter_token_latencies_ms(self) -> List[float]:
        """List of inter-token latencies in milliseconds (time between consecutive tokens)."""
        if len(self.token_events) < 2:
            return []
        return [
            self.token_events[i].timestamp_ms - self.token_events[i-1].timestamp_ms
            for i in range(1, len(self.token_events))
        ]

    @property
    def mean_inter_token_latency_ms(self) -> Optional[float]:
        """Average inter-token latency in milliseconds (decode phase)."""
        itls = self.inter_token_latencies_ms
        return sum(itls) / len(itls) if itls else None

    @property
    def throughput_tokens_per_sec(self) -> Optional[float]:
        """Overall throughput in tokens/second."""
        if self.total_latency_ms is None or self.total_latency_ms == 0:
            return None
        return self.actual_output_len / (self.total_latency_ms / 1000.0)

@dataclass
class SimulationStep:
    """Record of a single simulation step."""
    step_id: int
    sim_time_ms: float  # Simulation time in milliseconds
    elapsed_time_ms: float  # Step duration in milliseconds (for comparison with vLLM)

    # Scheduling info
    num_scheduled_new: int
    num_scheduled_cached: int
    total_tokens_scheduled: int
    scheduled_req_ids: List[str]

    # Block usage
    free_blocks: int

    # Finished/preempted
    finished_req_ids: List[str]
    preempted_req_ids: List[str]

    # processed requests
    processed_reqs: List[str]

    # Request states
    waiting_count: int
    running_count: int


class SimulationResult:
    """Results from a simulation run with export capabilities."""

    def __init__(self):
        self.steps: List[SimulationStep] = []
        self.total_requests: int = 0
        self.total_steps: int = 0
        self.total_tokens_generated: int = 0
        self.preemption_count: int = 0

        # Per-request stats (backward compatibility)
        self.request_latencies: Dict[str, float] = {}  # req_id -> total latency
        self.request_ttft: Dict[str, float] = {}  # req_id -> time to first token

        # Detailed per-request traces
        self.request_traces: Dict[str, RequestTrace] = {}  # req_id -> RequestTrace

    def export_aggregate_csv(self, output_path: str) -> None:
        """Export per-request aggregate statistics to CSV.

        Columns: request_id, prompt_len, output_tokens, arrival_time_ms, queue_time_ms,
                 ttft_ms, total_latency_ms, mean_itl_ms, throughput_tokens_per_sec
        """
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)

            # Write header
            writer.writerow([
                'request_id',
                'prompt_len',
                'output_tokens',
                'arrival_time_ms',
                'queue_time_ms',
                'ttft_ms',
                'total_latency_ms',
                'mean_itl_ms',
                'throughput_tokens_per_sec'
            ])

            # Write data rows
            for req_id, trace in self.request_traces.items():
                writer.writerow([
                    req_id,
                    trace.prompt_len,
                    trace.actual_output_len,
                    f"{trace.arrival_time_ms:.2f}",
                    f"{trace.queue_time_ms:.2f}" if trace.queue_time_ms is not None else "",
                    f"{trace.ttft_ms:.2f}" if trace.ttft_ms is not None else "",
                    f"{trace.total_latency_ms:.2f}" if trace.total_latency_ms is not None else "",
                    f"{trace.mean_inter_token_latency_ms:.2f}" if trace.mean_inter_token_latency_ms is not None else "",
                    f"{trace.throughput_tokens_per_sec:.2f}" if trace.throughput_tokens_per_sec is not None else ""
                ])

        print(f"Exported aggregate statistics to {output_path}")

    def export_cdf_csv(self, output_path: str) -> None:
        """Export TTFT and ITL data for CDF plotting.

        Two CSV files will be created:
        - {output_path}_ttft.csv: One TTFT value per request
        - {output_path}_itl.csv: All inter-token latencies across all requests
        """
        # Remove .csv extension if present
        base_path = output_path.replace('.csv', '')

        # Export TTFT data
        ttft_path = f"{base_path}_ttft.csv"
        with open(ttft_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['request_id', 'ttft_ms'])

            for req_id, trace in self.request_traces.items():
                if trace.ttft_ms is not None:
                    writer.writerow([req_id, f"{trace.ttft_ms:.2f}"])

        print(f"Exported TTFT CDF data to {ttft_path}")

        # Export ITL data
        itl_path = f"{base_path}_itl.csv"
        with open(itl_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['request_id', 'token_index', 'itl_ms'])

            for req_id, trace in self.request_traces.items():
                itls = trace.inter_token_latencies_ms
                for i, itl in enumerate(itls):
                    # token_index is the index of the token whose ITL this is (starts at 1)
                    writer.writerow([req_id, i + 1, f"{itl:.2f}"])

        print(f"Exported ITL CDF data to {itl_path}")

    def export_throughput_vs_time_csv(self, output_path: str, window_ms: float = 100.0) -> None:
        """Export throughput over time for plotting.

        Calculates throughput in a sliding window and exports timestamped data.

        Columns: time_ms, tokens_per_sec, num_active_requests
        """
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time_ms', 'tokens_per_sec', 'num_active_requests'])

            # Collect all token generation events with timestamps
            token_events = []
            for req_id, trace in self.request_traces.items():
                for event in trace.token_events:
                    token_events.append((event.timestamp_ms, req_id))

            # Sort by timestamp
            token_events.sort(key=lambda x: x[0])

            if not token_events:
                print(f"Warning: No token events found for throughput calculation")
                return

            # Calculate throughput at regular intervals
            start_time = token_events[0][0]
            end_time = token_events[-1][0]

            current_time = start_time
            interval_ms = window_ms / 2  # Sample every half window for smoother curve

            while current_time <= end_time:
                # Count tokens in [current_time - window_ms, current_time]
                window_start = current_time - window_ms
                tokens_in_window = sum(1 for t, _ in token_events if window_start <= t <= current_time)

                # Count unique active requests in window
                active_reqs = set(req_id for t, req_id in token_events if window_start <= t <= current_time)

                # Calculate throughput (tokens per second)
                throughput = (tokens_in_window / window_ms) * 1000.0 if window_ms > 0 else 0.0

                writer.writerow([
                    f"{current_time:.2f}",
                    f"{throughput:.2f}",
                    len(active_reqs)
                ])

                current_time += interval_ms

        print(f"Exported throughput vs time to {output_path}")

    def export_vllm_json(self, output_path: str, model_id: str, request_rate: float) -> None:
        """Export results in vLLM benchmark JSON format for easy comparison.

        This matches the output format from `vllm bench serve --save-result`.

        Args:
            output_path: Path to output JSON file
            model_id: Model identifier
            request_rate: Request arrival rate used
        """
        import json
        from datetime import datetime

        # Calculate aggregate metrics
        # Sort traces by arrival_time to match input request order
        traces = sorted(self.request_traces.values(), key=lambda t: t.arrival_time_ms)
        if not traces:
            print("No traces to export")
            return

        # Get simulation duration (from first arrival to last completion)
        start_time = min(t.arrival_time_ms for t in traces)
        end_time = max(t.finish_time_ms for t in traces if t.finish_time_ms is not None)
        duration_sec = (end_time - start_time) / 1000.0

        completed = sum(1 for t in traces if t.finish_time_ms is not None)
        total_input_tokens = sum(t.prompt_len for t in traces)
        total_output_tokens = sum(t.actual_output_len for t in traces)

        import numpy as np
        
        
        all_ttfts = [t.ttft_ms for t in traces if t.ttft_ms is not None]
        all_itls = [itl for t in traces if t.mean_inter_token_latency_ms is not None for itl in t.inter_token_latencies_ms] 
        
         
        # Build result dict matching vLLM format
        result = {
            "endpoint_type": "dooly_sim",
            "backend": "dooly_sim",
            "model": model_id,
            "num_prompts": len(traces),
            "arrival_rate": request_rate,
            "duration": duration_sec,
            "request_throughput": completed / duration_sec if duration_sec > 0 else 0,
            "output_throughput": total_output_tokens / duration_sec if duration_sec > 0 else 0,
            "total_throughput": (total_input_tokens + total_output_tokens) / duration_sec if duration_sec > 0 else 0,
            # "completed": completed,
            # "failed": len(traces) - completed,
            "mean_ttft_ms": sum(t.ttft_ms for t in traces if t.ttft_ms is not None) / completed if completed > 0 else 0,
            "median_ttft_ms": np.median(all_ttfts) if completed > 0 else 0,
            "p95_ttft_ms": np.percentile(all_ttfts, 95) if completed > 0 else 0,
            "p99_ttft_ms": np.percentile(all_ttfts, 99) if completed > 0 else 0,
            "mean_itl_ms": float(np.mean(all_itls)) if all_itls else 0,  # Flat mean across all tokens (matches vLLM's calculate_metrics)
            "median_itl_ms": np.median(all_itls) if all_itls else 0,
            "p95_itl_ms": np.percentile(all_itls, 95) if all_itls else 0,
            "p99_itl_ms": np.percentile(all_itls, 99) if all_itls else 0,

            # Per-request arrays
            "ttfts": [t.ttft_ms / 1000.0 if t.ttft_ms is not None else 0 for t in traces],  # Convert to seconds
            "itls": [[itl / 1000.0 for itl in t.inter_token_latencies_ms] for t in traces],  # Convert to seconds
            "latencies": [t.total_latency_ms / 1000.0 if t.total_latency_ms is not None else 0 for t in traces],  # End-to-end latency in seconds
            "input_lens": [t.prompt_len for t in traces],
            "output_lens": [t.actual_output_len for t in traces],
        }

        # Write JSON
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"Exported vLLM-format JSON to {output_path}")

    def export_all_csvs(self, output_dir: str, prefix: str = "simulation") -> None:
        """Export all CSV files for analysis.

        Args:
            output_dir: Directory to save CSV files
            prefix: Prefix for CSV filenames
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Export all three types of CSVs
        aggregate_path = os.path.join(output_dir, f"{prefix}_aggregate.csv")
        self.export_aggregate_csv(aggregate_path)

        cdf_base_path = os.path.join(output_dir, f"{prefix}_cdf")
        self.export_cdf_csv(cdf_base_path)

        throughput_path = os.path.join(output_dir, f"{prefix}_throughput_vs_time.csv")
        self.export_throughput_vs_time_csv(throughput_path)

        print(f"\nAll CSV files exported to {output_dir}/")

    def export_batch_composition_json(self, output_path: str) -> None:
        """Export detailed batch composition for each step.

        Outputs a JSON file with per-step batch information including:
        - Step metadata (time, step_id)
        - Requests scheduled (new vs cached)
        - Per-request token counts
        - Queue states
        - Preemptions
        """
        batch_data = []

        for step in self.steps:
            step_info = {
                "step_id": step.step_id,
                "sim_time_ms": round(step.sim_time_ms, 2),
                "elapsed_time_ms": round(step.elapsed_time_ms, 2),  # **Add elapsed_time_ms for comparison with vLLM**
                "batch_size": len(step.scheduled_req_ids),
                "scheduled_req_ids": step.scheduled_req_ids,
                "num_scheduled_new": step.num_scheduled_new,
                "num_scheduled_cached": step.num_scheduled_cached,
                "total_tokens_scheduled": step.total_tokens_scheduled,
                "free_blocks": step.free_blocks,
                "waiting_count": step.waiting_count,
                "running_count": step.running_count,
                "finished_req_ids": step.finished_req_ids,
                "preempted_req_ids": step.preempted_req_ids,
            }
            batch_data.append(step_info)

        # Write JSON
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(batch_data, f, indent=2)

        print(f"Exported batch composition to {output_path}")


def load_requests_from_jsonl(jsonl_path: str, tokenizer):

    from .request import RequestConfig 
    
    print(f"Loading requests from {jsonl_path}...")

    requests = []
    with open(jsonl_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            entry = json.loads(line)

            # Tokenize prompt
            token_ids = tokenizer.encode(entry['prompt'])

            # Create RequestConfig
            req = RequestConfig(
                request_id=entry.get('request_id', f"req_{line_num-1}"),
                prompt_token_ids=token_ids,
                max_tokens=entry['output_len'],
                arrival_time=entry['arrival_time'],
            )
            requests.append(req)

    print(f"✓ Loaded {len(requests)} requests from dataset")
    if len(requests) > 0:
        print(f"  Sample prompt lengths: {[len(r.prompt_token_ids) for r in requests[:5]]}")
        print(f"  Sample output lengths: {[r.max_tokens for r in requests[:5]]}")
        print(f"  Sample arrival times: {[f'{r.arrival_time:.3f}s' for r in requests[:5]]}")

    return requests 
