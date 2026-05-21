"""Analysis and reporting functions for simulation results."""

from ..utils import SimulationResult


def print_per_request_stats(result: SimulationResult, max_requests: int = 10):
    """Print detailed per-request statistics."""
    print("\n" + "=" * 60)
    print(f"PER-REQUEST STATISTICS (showing first {max_requests})")
    print("=" * 60)

    for idx, (req_id, trace) in enumerate(list(result.request_traces.items())[:max_requests]):
        print(f"\n{req_id}:")
        print(f"  Prompt length: {trace.prompt_len}")
        print(f"  Output tokens: {trace.actual_output_len} (requested: {trace.max_output_len})")
        print(f"  Arrival time: {trace.arrival_time_ms:.1f}ms")

        if trace.queue_time_ms is not None:
            print(f"  Queue time: {trace.queue_time_ms:.1f}ms")

        if trace.ttft_ms is not None:
            print(f"  Time to First Token: {trace.ttft_ms:.1f}ms")

        if trace.total_latency_ms is not None:
            print(f"  Total latency: {trace.total_latency_ms:.1f}ms")

        if trace.throughput_tokens_per_sec is not None:
            print(f"  Throughput: {trace.throughput_tokens_per_sec:.2f} tokens/s")

        if trace.mean_inter_token_latency_ms is not None:
            print(f"  Mean inter-token latency: {trace.mean_inter_token_latency_ms:.1f}ms")

        # Show first few token events
        if trace.token_events:
            print(f"  Token events (first 10):")
            for i, event in enumerate(trace.token_events[:10]):
                print(f"    Token {event.token_index}: t={event.timestamp_ms:.1f}ms, "
                      f"step_dur={event.step_duration_ms:.1f}ms, batch={event.batch_size}")

        if trace.finish_reason:
            print(f"  Finish reason: {trace.finish_reason}")


def print_summary(result: SimulationResult, show_per_request: bool = False):
    """Print simulation summary."""
    print("\n" + "=" * 60)
    print("SIMULATION SUMMARY")
    print("=" * 60)

    print(f"\nRequests: {result.total_requests}")
    print(f"Steps: {result.total_steps}")
    print(f"Total tokens generated: {result.total_tokens_generated}")
    print(f"Preemptions: {result.preemption_count}")

    if result.request_latencies:
        latencies = list(result.request_latencies.values())
        print(f"\nLatency (ms):")
        print(f"  Mean: {sum(latencies)/len(latencies):.1f}")
        print(f"  Min:  {min(latencies):.1f}")
        print(f"  Max:  {max(latencies):.1f}")

    if result.request_ttft:
        ttfts = list(result.request_ttft.values())
        print(f"\nTime to First Token (ms):")
        print(f"  Mean: {sum(ttfts)/len(ttfts):.1f}")
        print(f"  Min:  {min(ttfts):.1f}")
        print(f"  Max:  {max(ttfts):.1f}")

    # Block usage over time
    if result.steps:
        block_usage = [s.free_blocks for s in result.steps if s.total_tokens_scheduled > 0]
        if block_usage:
            print(f"\nFree blocks:")
            print(f"  Mean: {sum(block_usage)/len(block_usage):.1f}")
            print(f"  Min:  {min(block_usage)}")
            print(f"  Max:  {max(block_usage)}")

    # Per-request statistics from traces
    if result.request_traces and show_per_request:
        print_per_request_stats(result)
