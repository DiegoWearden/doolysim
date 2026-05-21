from attr import dataclass
from typing import Iterator, List
import random

from vllm.v1.request import Request
from vllm.sampling_params import SamplingParams
from vllm.v1.outputs import ModelRunnerOutput

class BatchConfig():
    def __init__(self, sched_out):
        # from sched_out
        self.sched_out = sched_out
        self.cached_reqs = sched_out.scheduled_cached_reqs
        
        # things to fill out
        self.prefill_requests: List[tuple[int, int]] = []
        self.decode_requests: List[int] = []
        self.batch_size = 0
        self.total_tokens = sched_out.total_num_scheduled_tokens
        # initialize variables
        self.extract_new_reqs()
        self.extract_processed_reqs()
        self.compute_batch_size()
        # Calculate num_requests AFTER extracting the requests
        self.num_requests = len(self.prefill_requests) + len(self.decode_requests)
    
    def __repr__(self):
        return (f"BatchConfig(batch_size={self.batch_size}, "
                f"prefill_requests={self.prefill_requests}, "
                f"decode_requests={self.decode_requests}, "
                f"total_tokens={self.total_tokens})")
        
    def extract_new_reqs(self):
         # Process new requests (prefill)
        for new_req in self.sched_out.scheduled_new_reqs:
            num_tokens = self.sched_out.num_scheduled_tokens.get(new_req.req_id, 0)
            kv_cache_size = new_req.num_computed_tokens
            self.prefill_requests.append((num_tokens, kv_cache_size))
            print(f"DEBUG: NEW req {new_req.req_id}: scheduling {num_tokens} tokens, kv_cache={kv_cache_size}")
        
    def extract_processed_reqs(self):

        print(f"DEBUG _extract_batch_info: cached_reqs.num_computed_tokens={self.cached_reqs.num_computed_tokens}, num_output_tokens={self.cached_reqs.num_output_tokens}")

        for idx, req_id in enumerate(self.cached_reqs.req_ids):
            if idx >= len(self.cached_reqs.num_computed_tokens):
                continue

            kv_cache_size = self.cached_reqs.num_computed_tokens[idx]
            num_output_tokens = self.cached_reqs.num_output_tokens[idx]
            num_tokens = self.sched_out.num_scheduled_tokens.get(req_id, 0)

            # Check if this is chunked prefill or decode
            if num_output_tokens == 0:
                # Chunked prefill - request hasn't finished prefill yet
                self.prefill_requests.append((num_tokens, kv_cache_size))
                print(f"DEBUG: CACHED req {req_id} (chunked prefill): scheduling {num_tokens} tokens, kv_cache={kv_cache_size}, num_output={num_output_tokens}")
            else:
                # Decode phase
                self.decode_requests.append(kv_cache_size)
                print(f"DEBUG: CACHED req {req_id} (decode): scheduling {num_tokens} tokens, kv_cache={kv_cache_size}, num_output={num_output_tokens}")
    
    def compute_batch_size(self):
        self.batch_size = len(self.sched_out.scheduled_new_reqs) + self.cached_reqs.num_reqs
    
@dataclass
class RequestConfig:
    """Configuration for a single request."""
    request_id: str
    prompt_token_ids: List[int]
    max_tokens: int
    arrival_time: float  
    
class RequestStream:
    def __init__(
        self,
        num_requests: int,
        arrival_rate: float,  # requests per second
        prompt_len_range: tuple[int, int] = (256, 1024),
        output_len_range: tuple[int, int] = (50, 256),
        seed: int = 42,
    ):
        self.num_requests = num_requests
        self.arrival_rate = arrival_rate
        self.prompt_len_range = prompt_len_range
        self.output_len_range = output_len_range
        self.rng = random.Random(seed)

    def generate(self) -> Iterator[RequestConfig]:
        """Generate requests with Poisson arrival times."""
        current_time = 0.0

        for i in range(self.num_requests):
            if self.arrival_rate > 0:
                inter_arrival = self.rng.expovariate(self.arrival_rate)
            else:
                inter_arrival = 0.0
            current_time += inter_arrival

            prompt_len = self.rng.randint(*self.prompt_len_range)
            output_len = self.rng.randint(*self.output_len_range)

            yield RequestConfig(
                request_id=f"req_{i}",
                prompt_token_ids=list(range(100, 100 + prompt_len)),
                max_tokens=output_len,
                arrival_time=current_time,
            )

def make_request(self, config: RequestConfig):
    """Create a request object."""""
    request_block_hasher = self.llm.llm_engine.engine_core.engine_core.request_block_hasher
    sampling_params = SamplingParams(max_tokens=config.max_tokens)
    req = Request(
        request_id=config.request_id,
        prompt_token_ids=config.prompt_token_ids,
        sampling_params=sampling_params,
        pooling_params=None,
        block_hasher=request_block_hasher,
    )

    print(f"DEBUG make_request: {config.request_id}, prompt_len={len(config.prompt_token_ids)}, "
          f"config.max_tokens={config.max_tokens}, req.max_tokens={req.max_tokens}, "
          f"sampling_params.max_tokens={sampling_params.max_tokens}")

    return req


def create_mock_output(self, scheduler_output, token_id: int = 1000):
    """Create mock ModelRunnerOutput simulating one token generated per request.

    Token generation rules:
    - Decode requests (num_output_tokens > 0): return 1 token
    - Prefill completing this step: return 1 token (the first output token)
    - Chunked prefill (incomplete): return empty list []

    Returns:
        tuple: (ModelRunnerOutput, set of request IDs that completed prefill this step)
    """
    # Get all scheduled request IDs
    all_req_ids = list(scheduler_output.num_scheduled_tokens.keys())

    # Track which requests should generate tokens
    generate_token = {}
    # Track which requests are completing prefill (for TTFT tracking)
    prefill_completing_req_ids = set()

    # Check new requests - they are in prefill
    for new_req in scheduler_output.scheduled_new_reqs:
        req_id = new_req.req_id
        num_scheduled = scheduler_output.num_scheduled_tokens.get(req_id, 0)
        prompt_len = len(new_req.prompt_token_ids) if new_req.prompt_token_ids else 0
        num_computed_before = new_req.num_computed_tokens

        # Check if prefill will complete after this step
        if num_computed_before + num_scheduled >= prompt_len:
            # Prefill completing - generate first token
            generate_token[req_id] = True
        else:
            # Chunked prefill, not done yet
            generate_token[req_id] = False

    # Check cached requests
    cached_reqs = scheduler_output.scheduled_cached_reqs
    for idx, req_id in enumerate(cached_reqs.req_ids):
        num_output_tokens = cached_reqs.num_output_tokens[idx]
        num_computed_before = cached_reqs.num_computed_tokens[idx]
        num_scheduled = scheduler_output.num_scheduled_tokens.get(req_id, 0)

        if num_output_tokens > 0:
            # Already in decode phase - generate token
            generate_token[req_id] = True
        else:
            # In prefill phase - need to check if completing
            # Get the actual Request object from scheduler to check prompt length
            request = self.scheduler.requests.get(req_id)
            if request:
                prompt_len = request.num_prompt_tokens
                will_complete = (num_computed_before + num_scheduled >= prompt_len)
                # Check if prefill will complete after this step
                if will_complete:
                    # Prefill completing - generate first token
                    print(f"DEBUG create_mock_output: {req_id} prefill COMPLETING: "
                          f"computed_before={num_computed_before}, scheduled={num_scheduled}, "
                          f"prompt_len={prompt_len} → generating token")
                    generate_token[req_id] = True
                else:
                    # Chunked prefill, not done yet
                    print(f"DEBUG create_mock_output: {req_id} chunked prefill CONTINUING: "
                          f"computed_before={num_computed_before}, scheduled={num_scheduled}, "
                          f"prompt_len={prompt_len} → NO token")
                    generate_token[req_id] = False
            else:
                # Fallback: assume prefill completes (shouldn't happen)
                print(f"DEBUG create_mock_output: WARNING - {req_id} not found in scheduler.requests, assuming completion")
                generate_token[req_id] = True

    # Create output
    sampled_token_ids = []
    for req_id in all_req_ids:
        if generate_token.get(req_id, True):
            sampled_token_ids.append([token_id])
        else:
            sampled_token_ids.append([])

    return ModelRunnerOutput(
        req_ids=all_req_ids,
        req_id_to_index={req_id: i for i, req_id in enumerate(all_req_ids)},
        sampled_token_ids=sampled_token_ids,
    )