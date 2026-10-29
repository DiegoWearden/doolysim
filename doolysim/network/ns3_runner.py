"""Run the ``dooly-collective`` ns-3 scenario and model it analytically.

The ns-3 program lives in ``ns3_backend/`` and is built with
``ns3_backend/setup_ns3.sh`` followed by ``ns3_backend/build.sh``.  This
module is deliberately free of DoolySim imports so it can be exercised on its
own and from the test-suite.

Two things live here:

* :func:`run_collective` -- spawn the binary for one collective and return its
  JSON result (schema version 1, documented in ``ns3_backend/README.md``).
* :func:`analytical_time_ns` -- the closed-form completion time of a ring
  collective on a given topology under ideal (store-and-forward, no
  contention) conditions, used by the tests to check the packet simulation.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BINARY = REPO_ROOT / "ns3_backend" / "build" / "dooly-collective"

SCHEMA_VERSION = 1

#: Bytes added to every packet's payload on the wire: the 16-byte CollHeader
#: plus the 2-byte PPP header that ns-3's point-to-point device prepends.
HEADER_BYTES = 16 + 2

COLLECTIVES = ("all_reduce", "reduce_scatter", "all_gather", "p2p")


class Ns3RunnerError(RuntimeError):
    """Raised when the ns-3 scenario cannot be run or did not complete."""

    def __init__(self, message: str, result: dict[str, Any] | None = None):
        super().__init__(message)
        self.result = result


# --------------------------------------------------------------------------
# Running the binary
# --------------------------------------------------------------------------
def find_binary() -> Path | None:
    """Locate ``dooly-collective``: ``$DOOLYSIM_NS3_BIN`` first, then the
    default build directory.  Returns ``None`` when it has not been built."""
    candidates = []
    env = os.environ.get("DOOLYSIM_NS3_BIN")
    if env:
        candidates.append(Path(env))
    candidates.append(DEFAULT_BINARY)
    for c in candidates:
        if c.is_file() and os.access(c, os.X_OK):
            return c
    return None


def run_collective(
    topology: str | os.PathLike,
    *,
    message_bytes: int,
    collective: str = "all_reduce",
    ranks: int = 0,
    ring_order: Sequence[int] | None = None,
    algorithm: str = "ring",
    seed: int = 1,
    max_sim_seconds: float = 10.0,
    binary: str | os.PathLike | None = None,
    timeout: float = 600.0,
) -> dict[str, Any]:
    """Simulate one collective and return the scenario's JSON result.

    ``ranks=0`` means every rank in the topology; ``ring_order`` lists the
    participating ranks in ring order and overrides ``ranks``.  Raises
    :class:`Ns3RunnerError` if the binary is missing, reports an error, or
    the collective did not complete; the partial result, when there is one,
    is attached as ``.result``.
    """
    if collective not in COLLECTIVES:
        raise ValueError(f"unknown collective {collective!r}; expected one of {COLLECTIVES}")
    if message_bytes <= 0:
        raise ValueError("message_bytes must be positive")

    exe = Path(binary) if binary else find_binary()
    if exe is None:
        raise Ns3RunnerError(
            "dooly-collective binary not found; run ns3_backend/setup_ns3.sh and "
            "ns3_backend/build.sh, or point DOOLYSIM_NS3_BIN at the executable"
        )

    args = [
        str(exe),
        f"--topology={os.fspath(topology)}",
        f"--collective={collective}",
        f"--algorithm={algorithm}",
        f"--ranks={ranks}",
        f"--message-bytes={message_bytes}",
        f"--seed={seed}",
        f"--max-sim-seconds={max_sim_seconds}",
    ]
    if ring_order:
        args.append("--ring-order=" + ",".join(str(r) for r in ring_order))
    proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout)

    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    try:
        result = json.loads(lines[-1])
    except (IndexError, json.JSONDecodeError):
        raise Ns3RunnerError(
            f"dooly-collective produced no JSON (exit {proc.returncode}): {proc.stderr.strip()}"
        ) from None

    if result.get("schema_version") != SCHEMA_VERSION:
        raise Ns3RunnerError(
            f"unsupported result schema_version {result.get('schema_version')!r}", result
        )
    if "error" in result:
        raise Ns3RunnerError(f"dooly-collective: {result['error']}", result)
    if not result.get("completed"):
        raise Ns3RunnerError(
            f"collective did not complete (dropped_packets={result.get('dropped_packets')})",
            result,
        )
    return result


# --------------------------------------------------------------------------
# Unit parsing (mirrors what ns-3's DataRate and Time accept)
# --------------------------------------------------------------------------
_RATE_UNITS_BPS = {
    "bps": 1.0, "b/s": 1.0,
    "kbps": 1e3, "Kbps": 1e3, "kb/s": 1e3,
    "Mbps": 1e6, "Mb/s": 1e6,
    "Gbps": 1e9, "Gb/s": 1e9,
    "Tbps": 1e12, "Tb/s": 1e12,
    "B/s": 8.0, "kB/s": 8e3, "KB/s": 8e3, "MB/s": 8e6, "GB/s": 8e9, "TB/s": 8e12,
}

_TIME_UNITS_NS = {
    "ps": 1e-3, "ns": 1.0, "us": 1e3, "ms": 1e6, "s": 1e9, "min": 60e9, "h": 3600e9,
}

_QUANTITY = re.compile(r"^\s*([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*([A-Za-z/]+)\s*$")


def parse_rate_bps(rate: str) -> float:
    m = _QUANTITY.match(rate)
    if not m or m.group(2) not in _RATE_UNITS_BPS:
        raise ValueError(f"cannot parse link rate {rate!r}")
    return float(m.group(1)) * _RATE_UNITS_BPS[m.group(2)]


def parse_time_ns(delay: str) -> float:
    m = _QUANTITY.match(delay)
    if not m or m.group(2) not in _TIME_UNITS_NS:
        raise ValueError(f"cannot parse time {delay!r}")
    return float(m.group(1)) * _TIME_UNITS_NS[m.group(2)]


# --------------------------------------------------------------------------
# Analytical model
# --------------------------------------------------------------------------
def ring_steps(collective: str, ranks: int) -> int:
    if collective == "all_reduce":
        return 2 * (ranks - 1)
    if collective in ("reduce_scatter", "all_gather"):
        return ranks - 1
    if collective == "p2p":
        return 1
    raise ValueError(f"unknown collective {collective!r}")


def chunk_bytes(collective: str, message_bytes: int, ranks: int) -> int:
    """Bytes each rank sends per step (matches the C++ scenario)."""
    if collective in ("all_gather", "p2p"):
        return message_bytes
    return -(-message_bytes // ranks)  # ceil(S / N)


def wire_bytes(payload: int, packet_bytes: int) -> int:
    """Bytes on the wire for ``payload`` bytes split into ``packet_bytes`` packets."""
    packets = -(-payload // packet_bytes)
    return payload + packets * HEADER_BYTES


def serialization_ns(payload: int, packet_bytes: int, rate: str) -> float:
    """Time to clock ``payload`` bytes (plus per-packet headers) onto a link."""
    return wire_bytes(payload, packet_bytes) * 8 / parse_rate_bps(rate) * 1e9


def transfer_time_ns(payload: int, packet_bytes: int, rate: str, delay: str) -> float:
    """Time for the last bit of ``payload`` to arrive over one idle link:
    serialization of every packet plus one propagation delay."""
    return serialization_ns(payload, packet_bytes, rate) + parse_time_ns(delay)


def ring_completion_ns(links_ns: Sequence[tuple[float, float]], steps: int) -> float:
    """Completion time of a ring collective with per-link timings.

    ``links_ns[i]`` is ``(serialization_ns, propagation_ns)`` for one chunk on
    the link from rank i to rank i+1.  The recurrence mirrors the simulator:

    * rank i starts transmitting step 0 at time 0;
    * its chunk for step k has fully left the link ``serialization`` later
      and reaches rank i+1 one ``propagation`` after that;
    * rank i may start step k+1 only once it has received step k from rank
      i-1 (data dependency) *and* its own link has finished sending step k
      (the link is work-conserving FIFO).

    The collective ends when the last rank has received its last step.  With
    uniform links this reduces to ``steps * (serialization + propagation)``;
    a slow link instead serialises every step that crosses it back to back.
    """
    n = len(links_ns)
    start = [0.0] * n  # when rank i begins transmitting the current step
    recv = [0.0] * n   # when rank i has received the current step
    for _ in range(steps):
        tx_end = [start[i] + links_ns[i][0] for i in range(n)]
        recv = [tx_end[(i - 1) % n] + links_ns[(i - 1) % n][1] for i in range(n)]
        start = [max(recv[i], tx_end[i]) for i in range(n)]
    return max(recv)


def load_topology(path: str | os.PathLike) -> dict[str, Any]:
    with open(path) as f:
        topo = json.load(f)
    if topo.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported topology schema_version {topo.get('schema_version')!r}")
    return topo


class Graph:
    """The topology as the C++ scenario builds it: node ``i`` is rank ``i``
    for ``i < ranks`` and switch ``i - ranks`` otherwise; each node's ports
    are listed in topology-file order, which is what breaks routing ties."""

    def __init__(self, topo: dict[str, Any]):
        self.ranks = int(topo["ranks"])
        self.switches = int(topo.get("switches", 0))
        n = self.ranks + self.switches
        self.ports: list[list[tuple[int, dict[str, Any]]]] = [[] for _ in range(n)]
        self._next_hop_cache: dict[int, list[int]] = {}

        defaults = topo.get("defaults", {})
        links: list[tuple[str, str, dict[str, Any]]] = []
        gen = topo.get("generator")
        if gen == "ring":
            for i in range(1 if self.ranks == 2 else self.ranks):
                links.append((f"r{i}", f"r{(i + 1) % self.ranks}", {}))
        elif gen == "star":
            links.extend((f"r{i}", "s0", {}) for i in range(self.ranks))
        elif gen:
            raise ValueError(f"unknown generator {gen!r}")
        links.extend((l["a"], l["b"], l) for l in topo.get("links", []))

        for a, b, o in links:
            spec = {**defaults, **{k: v for k, v in o.items() if k not in ("a", "b")}}
            ia, ib = self.index(a), self.index(b)
            self.ports[ia].append((ib, spec))
            self.ports[ib].append((ia, spec))

    def index(self, name: str) -> int:
        kind, num = name[0], int(name[1:])
        if kind == "r" and 0 <= num < self.ranks:
            return num
        if kind == "s" and 0 <= num < self.switches:
            return self.ranks + num
        raise ValueError(f"unknown node name {name!r}")

    def _next_hops(self, src: int) -> list[int]:
        """BFS from ``src``: port index at ``src`` on the first-found shortest
        path to every node (-1 for ``src`` itself / unreachable)."""
        if src in self._next_hop_cache:
            return self._next_hop_cache[src]
        first = [-1] * len(self.ports)
        seen = [False] * len(self.ports)
        seen[src] = True
        queue = [src]
        while queue:
            u = queue.pop(0)
            for pi, (v, _) in enumerate(self.ports[u]):
                if seen[v]:
                    continue
                seen[v] = True
                first[v] = pi if u == src else first[u]
                queue.append(v)
        self._next_hop_cache[src] = first
        return first

    def route(self, src: int, dst: int) -> list[dict[str, Any]]:
        """Link specs along the hop-by-hop forwarding path from src to dst."""
        path = []
        u = src
        while u != dst:
            pi = self._next_hops(u)[dst]
            if pi < 0:
                raise ValueError(f"no route from node {src} to node {dst}")
            v, spec = self.ports[u][pi]
            path.append(spec)
            u = v
        return path


def path_transfer_ns(payload: int, packet_bytes: int,
                     path: Sequence[dict[str, Any]]) -> tuple[float, float]:
    """Pipelined store-and-forward of ``payload`` along an idle multi-hop path.

    Returns ``(first_hop_busy_ns, last_arrival_ns)``: when the sender's own
    link has finished clocking out the last packet, and when the last packet
    reaches the destination.  Each hop forwards a packet as soon as it has
    fully arrived and the hop's link is free, exactly as ns-3's point-to-point
    devices do.
    """
    full, rem = divmod(payload, packet_bytes)
    wires = [packet_bytes + HEADER_BYTES] * full + ([rem + HEADER_BYTES] if rem else [])
    rates = [parse_rate_bps(l["rate"]) for l in path]
    delays = [parse_time_ns(l["delay"]) for l in path]
    busy = [0.0] * len(path)  # when each hop's link finishes its previous packet
    arrival = 0.0
    for w in wires:
        t = 0.0  # the packet is available at the source from the start
        for h in range(len(path)):
            start = max(t, busy[h])
            busy[h] = start + w * 8 / rates[h] * 1e9
            t = busy[h] + delays[h]
        arrival = t
    return busy[0], arrival


def analytical_time_ns(
    topology: str | os.PathLike | dict[str, Any],
    *,
    collective: str,
    message_bytes: int,
    ranks: int = 0,
    ring_order: Sequence[int] | None = None,
) -> float:
    """Ideal (contention-free) completion time of ``collective`` on a topology.

    Every ring edge is routed exactly as the simulator routes it and timed
    with :func:`path_transfer_ns`; the per-rank recurrence of
    :func:`ring_completion_ns` then chains the steps.  The result is exact
    whenever no two edges share a link direction; with contention it is a
    lower bound.
    """
    topo = load_topology(topology) if not isinstance(topology, dict) else topology
    g = Graph(topo)
    order = list(ring_order) if ring_order else list(range(ranks or g.ranks))
    n = len(order)
    packet = int(topo.get("packet_bytes", 4096))
    chunk = chunk_bytes(collective, message_bytes, n)

    if collective == "p2p":
        return path_transfer_ns(chunk, packet, g.route(order[0], order[1]))[1]

    edges = []
    for i in range(n):
        busy, arrival = path_transfer_ns(chunk, packet, g.route(order[i], order[(i + 1) % n]))
        edges.append((busy, arrival - busy))
    return ring_completion_ns(edges, ring_steps(collective, n))


def _main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Run one collective through the ns-3 backend.")
    p.add_argument("--topology", required=True)
    p.add_argument("--collective", default="all_reduce", choices=COLLECTIVES)
    p.add_argument("--message-bytes", type=int, required=True)
    p.add_argument("--ranks", type=int, default=0)
    p.add_argument("--ring-order", type=lambda s: [int(x) for x in s.split(",")], default=None,
                   help="participating ranks in ring order, e.g. 0,4,1,5")
    p.add_argument("--analytical", action="store_true",
                   help="also print the contention-free model time for comparison")
    a = p.parse_args(argv)

    result = run_collective(a.topology, collective=a.collective,
                            message_bytes=a.message_bytes, ranks=a.ranks,
                            ring_order=a.ring_order)
    if a.analytical:
        result["analytical_time_ns"] = analytical_time_ns(
            a.topology, collective=a.collective, message_bytes=a.message_bytes,
            ranks=a.ranks, ring_order=a.ring_order)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
