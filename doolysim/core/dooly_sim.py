import os
import torch
import time

from ..utils import SimulationResult, SimulationStep, RequestTrace, TokenEvent
from ..request import RequestConfig, make_request, create_mock_output
from ..emulator import emulate_init
from .simulation_predictor import SimulationPredictor

from typing import Dict, Iterator, Optional

# Import vLLM components for process_outputs
try:
    from vllm.v1.engine import EngineCoreOutput
    from vllm.v1.engine.output_processor import RequestState
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("Warning: vLLM components not available for process_outputs")

class DoolySim:
    def __init__(
        self,
        model: str = "gpt2",
        args: Dict = {"max_model_len": 16384, "gpu_memory_utilization": 0.9, "max_num_seqs": 10, "max_num_batched_tokens": 512},
        tensor_parallel_size: int = 1,
        step_duration_ms: float = 10.0,  # Simulated time per forward pass (fallback)
        cache_dir: Optional[str] = None,
        db_path: Optional[str] = None,
        backend: str = "FLASHINFER",
        topology: str = "mew1",  # GPU interconnect topology for collective operations
        cpu_overhead: float = 10.0,
        use_actual_process_outputs: bool = True  # Use actual vLLM process_outputs for CPU overhead
    ):
        cache_dir = cache_dir or os.environ.get("DOOLYSIM_CACHE_DIR")
        db_path = db_path or os.environ.get("DOOLYSIM_DB_PATH")
        if cache_dir is None:
            raise ValueError("cache_dir must be provided (or set DOOLYSIM_CACHE_DIR env var)")
        if db_path is None:
            raise ValueError("db_path must be provided (or set DOOLYSIM_DB_PATH env var)")
        self.cpu_overhead = cpu_overhead
        self.use_actual_process_outputs = use_actual_process_outputs
        self.llm= emulate_init(model=model, args=args)
        self.tokenizer = self.llm.get_tokenizer()
        self.vllm_config = self.llm.llm_engine.vllm_config

        
        cache_cfg = self.vllm_config.cache_config
        model_cfg = self.vllm_config.model_config
        parallel_cfg = self.vllm_config.parallel_config

        self.max_model_len: int = model_cfg.max_model_len
        self.block_size: int = cache_cfg.block_size
        self.num_blocks: int = cache_cfg.num_gpu_blocks
        self.num_kv_heads: int = model_cfg.get_num_kv_heads(parallel_cfg)
        self.head_size: int = model_cfg.get_head_size()

        self.step_duration_ms = step_duration_ms

        # Create scheduler
        self.scheduler= self.llm.llm_engine.engine_core.engine_core.scheduler
        
        # Access the output processor for CPU overhead estimation
        self.output_processor = self.llm.llm_engine.output_processor if use_actual_process_outputs else None

        self.max_num_seqs: int = self.scheduler.max_num_running_reqs
        self.max_num_batched_tokens: int = self.scheduler.max_num_scheduled_tokens

        # Initialize simulation predictor
        self.model_name = model
        self.backend = backend
        # dtype and KV-cache dtype come from the emulated engine's resolved
        # config (the same values vLLM would run with), so the predictor picks
        # the profiled_models row of exactly this deployment.
        self.latency_estimator = SimulationPredictor(
            model_name=model,
            backend=backend,
            tensor_parallel_size=tensor_parallel_size,
            dtype=model_cfg.dtype,
            vllm_config=self.vllm_config,
            cache_dir=cache_dir,
            db_path=db_path,
            topology=topology,  # Pass topology for collective operations
            fallback_duration_ms=step_duration_ms,
            kv_cache_dtype=str(cache_cfg.cache_dtype),
        )

        # Tracking (all in milliseconds)
        self.request_arrival_times_ms: Dict[str, float] = {}
        self.request_first_token_times_ms: Dict[str, float] = {}
        self.request_finish_times_ms: Dict[str, float] = {}

    def dump_config(self):
        """Print all configuration fields extracted from the LLM object."""
        cfg = self.vllm_config
        print("=" * 60)
        print("SchedulerSimulator Configuration")
        print("=" * 60)
        print(f"  max_model_len          : {self.max_model_len}")
        print(f"  block_size             : {self.block_size}")
        print(f"  num_blocks             : {self.num_blocks}")
        print(f"  num_kv_heads           : {self.num_kv_heads}")
        print(f"  head_size              : {self.head_size}")
        print(f"  max_num_seqs           : {self.max_num_seqs}")
        print(f"  max_num_batched_tokens : {self.max_num_batched_tokens}")
        print("=" * 60)

    def _create_mock_output(self, scheduler_output, token_id: int = 1000):
        return create_mock_output(self, scheduler_output, token_id)

    def _estimate_process_outputs(self, engine_outputs, has_prefill: bool) -> float:
        if not VLLM_AVAILABLE or not self.output_processor:
            # Fallback to fixed overhead
            return self.cpu_overhead if has_prefill else 0.0

        try:
            # Time the actual process_outputs call
            
            start_time = time.perf_counter()
            _ = self.output_processor.process_outputs(
                engine_outputs.get(0).outputs,
                engine_core_timestamp=None,
                iteration_stats=None
            )
            end_time = time.perf_counter()

            process_output_ms = (end_time - start_time) * 1000.0

            print(f"Estimated process_outputs overhead: {process_output_ms:.1f}ms (has_prefill={has_prefill})")
            return process_output_ms

        except Exception as e:
            import traceback
            print(f"Error estimating process_outputs overhead: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            # Fallback to fixed overhead
            return self.cpu_overhead if has_prefill else 0.0

    def _add_request(self, config: RequestConfig, result: SimulationResult, actual_arrival_time_ms: float):
        req = make_request(self, config)
        print(f"Adding Request {req.request_id} of prompt length {len(req.prompt_token_ids)}")
        # req_id = config.request_id
        
        process_input_start = time.perf_counter()
        # self.scheduler.add_request(req)
        req_id = self.llm.llm_engine.add_request(req.request_id, req.prompt_token_ids, req.sampling_params)
        process_input_end = time.perf_counter()
        process_input_ms = (process_input_end - process_input_start) * 1000.0
        
        self.request_arrival_times_ms[req_id] = actual_arrival_time_ms


        # Create RequestTrace for detailed tracking
        result.request_traces[req_id] = RequestTrace(
            request_id=config.request_id,
            arrival_time_ms=actual_arrival_time_ms,
            prompt_len=len(config.prompt_token_ids),
            max_output_len=config.max_tokens,
        )
        
        return process_input_ms

    def run(
            self,
            request_stream: Iterator[RequestConfig],
            max_steps: int = 10000,
            verbose: bool = False,
        ) -> SimulationResult:
            result = SimulationResult()
            all_requests = list(request_stream)
            all_requests.sort(key=lambda r: r.arrival_time)
            result.total_requests = len(all_requests)

            request_idx = 0
            sim_time_ms = 0.0
            step_id = 1

            if verbose:
                print(f"Starting simulation with {len(all_requests)} requests")
                print(f"Block capacity: {self.num_blocks} blocks x {self.block_size} = {self.num_blocks * self.block_size} tokens")

            while step_id < max_steps:
                
                # 1. Add any requests that have arrived by this time
                process_input_times = []
                current_time_ms = sim_time_ms
                while request_idx < len(all_requests) and all_requests[request_idx].arrival_time * 1000.0 <= current_time_ms:
                    req_config = all_requests[request_idx]
                    # Use the request's scheduled arrival time, not current sim_time.
                    # This matches vLLM which records metrics.arrival_time as the
                    # scheduled time. Without this, requests that arrive during a
                    # previous step's execution have their wait time excluded from TTFT.
                    scheduled_arrival_ms = req_config.arrival_time * 1000.0
                    process_input_ms = self._add_request(req_config, result, scheduled_arrival_ms)
                    process_input_times.append(process_input_ms)
                    if verbose:
                        print(f"[t={sim_time_ms:.1f}ms] Request {req_config.request_id} arrived "
                            f"(prompt={len(req_config.prompt_token_ids)}, max_out={req_config.max_tokens})")
                    request_idx += 1

                # 2. Check if there's work to do
                if not self.scheduler.has_requests() and request_idx >= len(all_requests):
                    if verbose:
                        print(f"\n[t={sim_time_ms:.1f}ms] All requests completed!")
                    break

                if not self.scheduler.has_requests():
                    # No requests yet, advance time to next arrival
                    if request_idx < len(all_requests):
                        sim_time_ms = all_requests[request_idx].arrival_time * 1000.0
                    continue

                # 3. Schedule next batch (Moved up to follow vLLM's step() flow)
                sched_output = self.scheduler.schedule()

                # Track when requests are first scheduled
                for new_req in sched_output.scheduled_new_reqs:
                    if new_req.req_id in result.request_traces:
                        trace = result.request_traces[new_req.req_id]
                        if trace.scheduled_time_ms is None:
                            trace.scheduled_time_ms = sim_time_ms
                            if verbose:
                                queue_time = sim_time_ms - trace.arrival_time_ms
                                print(f"[t={sim_time_ms:.1f}ms] Request {new_req.req_id} first scheduled "
                                    f"(queue_time={queue_time:.1f}ms)")

                # If nothing scheduled, advance time slightly
                if sched_output.total_num_scheduled_tokens == 0:
                    if request_idx < len(all_requests):
                        sim_time_ms = all_requests[request_idx].arrival_time * 1000.0
                    else:
                        sim_time_ms += 1.0
                    continue
                
                # 4. Execute and Update Scheduler (Replaces prev_output logic)
                mock_output = self._create_mock_output(sched_output, token_id=1000 + step_id)
                
                # 5. Estimate the step duration 
                has_prefill = len(sched_output.scheduled_new_reqs) > 0 or any(sched_output.scheduled_cached_reqs.is_context_phase(req_id) for req_id in sched_output.scheduled_cached_reqs.req_ids)

                gpu_latency_ms = self.latency_estimator.estimate_step_latency(
                    sched_output=sched_output,
                    step_id=step_id
                )
                
                # update scheduler
                engine_outputs = self.scheduler.update_from_output(sched_output, mock_output)
                
                # process output
                if self.use_actual_process_outputs and self.output_processor:
                    process_output_ms = self._estimate_process_outputs(engine_outputs, True)
                else:
                    process_output_ms = self.cpu_overhead if has_prefill else 0.0

                self.step_duration_ms = gpu_latency_ms + process_output_ms + (sum(process_input_times) if process_input_times else 0.0)

                if verbose:
                    print(f"  └─ TIMING: GPU={gpu_latency_ms:.1f}ms, CPU={process_output_ms:.1f}ms, Total={self.step_duration_ms:.1f}ms")

                # 6. Advance simulation time 
                sim_time_ms += self.step_duration_ms
                result.total_tokens_generated += sched_output.total_num_scheduled_tokens

                # 7. Process Events based on actual engine_outputs
                batch_size = len(sched_output.num_scheduled_tokens)
                
                for req_id, outputs in engine_outputs.items():
                    for out in outputs.outputs:
                        # Record token events
                        if out.new_token_ids and len(out.new_token_ids) > 0:
                            if out.request_id not in self.request_first_token_times_ms:
                                self.request_first_token_times_ms[out.request_id] = sim_time_ms

                            if out.request_id in result.request_traces:
                                trace = result.request_traces[out.request_id]

                                for token_idx, _ in enumerate(out.new_token_ids):
                                    token_event = TokenEvent(
                                        token_index=len(trace.token_events),
                                        timestamp_ms=sim_time_ms, # Now correctly reflects time AFTER execution
                                        step_duration_ms=self.step_duration_ms,
                                        batch_size=batch_size,
                                        total_tokens_in_batch=sched_output.total_num_scheduled_tokens,
                                    )
                                    trace.token_events.append(token_event)

                        # Record finishes
                        if out.finish_reason:
                            self.request_finish_times_ms[out.request_id] = sim_time_ms
                            if out.request_id in result.request_traces:
                                trace = result.request_traces[out.request_id]
                                trace.finish_time_ms = sim_time_ms 
                                trace.finish_reason = str(out.finish_reason)

                            if verbose:
                                latency_ms = sim_time_ms - self.request_arrival_times_ms[out.request_id]
                                print(f"[t={sim_time_ms:.1f}ms] Request {out.request_id} FINISHED "
                                    f"(reason={out.finish_reason}, latency={latency_ms:.1f}ms)")

                # 8. Record step metrics
                free_blocks = self.scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
                preempted = list(sched_output.preempted_req_ids) if sched_output.preempted_req_ids else []
                if preempted:
                    result.preemption_count += len(preempted)

                step_record = SimulationStep(
                    step_id=step_id,
                    sim_time_ms=sim_time_ms,
                    elapsed_time_ms=self.step_duration_ms,
                    processed_reqs=engine_outputs.keys(),
                    num_scheduled_new=len(sched_output.scheduled_new_reqs),
                    num_scheduled_cached=sched_output.scheduled_cached_reqs.num_reqs,
                    total_tokens_scheduled=sched_output.total_num_scheduled_tokens,
                    scheduled_req_ids=list(sched_output.num_scheduled_tokens.keys()),
                    free_blocks=free_blocks,
                    finished_req_ids=list(sched_output.finished_req_ids),
                    preempted_req_ids=preempted,
                    waiting_count=len(self.scheduler.waiting),
                    running_count=len(self.scheduler.running),
                )
                result.steps.append(step_record)
                
                step_id += 1

            result.total_steps = step_id

            # Calculate final latencies
            for req_id, arrival_ms in self.request_arrival_times_ms.items():
                if req_id in self.request_finish_times_ms:
                    result.request_latencies[req_id] = self.request_finish_times_ms[req_id] - arrival_ms
                if req_id in self.request_first_token_times_ms:
                    result.request_ttft[req_id] = self.request_first_token_times_ms[req_id] - arrival_ms

            return result