# ns-3 network backend

`dooly-collective` is a standalone ns-3 program that simulates one collective
operation over a configurable accelerator fabric and reports its completion
time. DoolySim's ns-3 network backend calls it once per collective instead of
looking up a profiled latency.

## Build

```bash
ns3_backend/setup_ns3.sh     # clone + build + install a minimal ns-3.49 into ~/ns-3-dev
ns3_backend/build.sh         # build ns3_backend/build/dooly-collective against it
python -m pytest tests/test_ns3_ring.py -q
```

Both scripts honour `NS3_DIR` and `NS3_PREFIX` if you keep ns-3 elsewhere.
The Python side finds the binary at `ns3_backend/build/dooly-collective` or
via `DOOLYSIM_NS3_BIN`.

## Run

```bash
ns3_backend/build/dooly-collective \
    --topology=examples/network/ring-8.json \
    --collective=all_reduce --message-bytes=16777216
```

or, from Python, `doolysim.network.ns3_runner.run_collective(...)`. Passing
`--analytical` to `python -m doolysim.network.ns3_runner` prints the ideal
ring-model time next to the simulated one.

| Flag | Meaning |
| --- | --- |
| `--topology` | topology JSON file (below) |
| `--collective` | `all_reduce`, `reduce_scatter`, `all_gather`, `p2p` |
| `--algorithm` | `ring` (only option today) |
| `--ranks` | participating ranks `r0..r(N-1)`; `0` means all ranks in the topology |
| `--ring-order` | participating ranks in ring order, e.g. `0,4,1,5,2,6,3,7`; overrides `--ranks` and lets you study rank placement |
| `--message-bytes` | per-rank buffer handed to the collective |
| `--seed` | RNG seed; the model is deterministic but the seed is recorded for reproducibility |
| `--max-sim-seconds` | give up after this much simulated time |

## Result schema (version 1)

One JSON object on stdout. Exit code 0 on completion, 3 if the collective did
not complete (dropped packets or `--max-sim-seconds`), 2 on a usage or
topology error (`{"schema_version": 1, "error": "..."}`).

```json
{
  "schema_version": 1,
  "topology": "ring-8",
  "collective": "all_reduce",
  "algorithm": "ring",
  "ranks": 8,
  "ring_order": [0, 1, 2, 3, 4, 5, 6, 7],
  "message_bytes": 16777216,
  "chunk_bytes": 2097152,
  "steps": 14,
  "packet_bytes": 4096,
  "completed": true,
  "completion_time_ns": 596783,
  "bytes_transferred": 234881024,
  "packets_sent": 57344,
  "max_queue_bytes": 1044956,
  "max_buffer_bytes": 0,
  "dropped_packets": 0,
  "wall_time_ms": 112.1,
  "peak_rss_kb": 14624
}
```

`bytes_transferred` counts payload delivered to destination ranks.
`max_queue_bytes` is the deepest any link transmit queue got and
`max_buffer_bytes` the most any forwarding buffer held; both stay small
unless flows share a link. `dropped_packets` is always 0 in the lossless
model and is reported as a sanity check. `wall_time_ms` and `peak_rss_kb`
are the simulator's own cost, for the overhead evaluation.

## Topology schema (version 1)

```json
{
  "schema_version": 1,
  "name": "ring-8",
  "ranks": 8,
  "switches": 0,
  "packet_bytes": 4096,
  "generator": "ring",
  "defaults": { "rate": "400Gbps", "delay": "500ns", "queue_bytes": 1048576,
                "buffer_bytes": 4194304 },
  "links": [ { "a": "r2", "b": "r3", "rate": "100Gbps" } ]
}
```

* Nodes are named `r<i>` (ranks) and `s<i>` (switches).
* `generator` is optional: `ring` links `r<i>`–`r<i+1>` around the ring (a
  2-rank ring is one link); `star` links every rank to `s0`.
* `links` adds explicit full-duplex links; each may override `rate`,
  `delay`, `queue_bytes` and `buffer_bytes` from `defaults`. Rates and
  delays use ns-3 spellings such as `400Gbps`, `50GB/s`, `500ns`, `1us`.
* `queue_bytes` is the link's transmit queue; `buffer_bytes` (default 4 MiB)
  is the forwarding buffer a node keeps in front of that queue for traffic
  passing through it.
* `packet_bytes` is the payload per packet; each packet also carries an
  18-byte header on the wire.

Examples live in `examples/network/`:

| File | Shape |
| --- | --- |
| `ring-2/4/8.json` | ranks in a ring, 400 Gbps links |
| `ring-4-slow-link.json` | ring with one 100 Gbps link |
| `star-8.json` | eight ranks on one switch |
| `two-switch-8.json` | four ranks per switch, one inter-switch link; with `--ring-order=0,4,1,5,2,6,3,7` every ring edge crosses that link |

## What is modelled

* Every rank and switch is an ns-3 node. Every link is a full-duplex
  `PointToPointChannel` with its rate, propagation delay and a per-direction
  `DropTailQueue` of `queue_bytes`.
* Packets carry a 16-byte header (source, destination, step, sequence,
  payload) and are forwarded hop by hop along hop-count shortest paths, with
  ties broken by link order so routes are reproducible. There is no IP or
  TCP stack.
* The fabric is lossless, like the credit-based flow control of NVLink and
  InfiniBand class interconnects. Every port of a forwarding node has a
  `buffer_bytes` buffer in front of its transmit queue, and a node puts a
  packet on a link only after the next hop has reserved room for it in the
  buffer of the port it will continue through. Credits are granted
  instantly (no pause-frame latency), so nothing is ever dropped and
  contention shows up purely as queueing: flows sharing a port interleave
  packet by packet and the buffer fills until upstream senders stall.
* Ring collectives: `all_reduce` runs `2(N-1)` steps, `reduce_scatter` and
  `all_gather` run `N-1`. Each step every rank sends one chunk to the next
  rank in ring order and starts step `k+1` only once it has received all of
  step `k`, which is the data dependency of a ring reduction. Chunk size is
  `ceil(S/N)` for `all_reduce` and `reduce_scatter` and `S` for
  `all_gather`, where `S` is `--message-bytes`. `p2p` is a single transfer
  from the first to the second rank in the order, for calibration.
* Reduction arithmetic is free; only network time is simulated.

## Analytical reference

With an idle link of rate `B`, delay `d`, and a chunk of `c` bytes split into
`p` packets, one step takes `(c + 18p) * 8 / B + d`. Under ideal conditions
a uniform ring finishes in `steps` times that. Multi-hop edges are
pipelined store-and-forward: each extra hop adds one packet's serialization
plus its propagation delay. With heterogeneous links the model is a
recurrence: a rank starts step `k+1` once it has received step `k` *and*
its own link has finished sending step `k`, so a slow link serialises every
step that crosses it back to back rather than merely adding its extra delay
once per step.

`doolysim.network.ns3_runner.analytical_time_ns` routes every ring edge the
way the simulator does and applies that recurrence. The tests hold the
simulation to it within 1 ns on every contention-free scenario (rings,
star, two switches, sub-rings forwarded through idle ranks). When ring
edges share a link direction the model is only a lower bound; the
contention test checks that four flows on one port cost close to four
serializations per step.
