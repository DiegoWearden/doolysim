"""Validate the dooly-collective ns-3 scenario against the analytical ring model.

Skipped unless ``ns3_backend/build/dooly-collective`` (or ``$DOOLYSIM_NS3_BIN``)
exists; see ns3_backend/README.md for how to build it.
"""

from pathlib import Path

import pytest

from doolysim.network import ns3_runner as r

NET = Path(__file__).resolve().parents[1] / "examples" / "network"
RING = {n: NET / f"ring-{n}.json" for n in (2, 4, 8)}

pytestmark = pytest.mark.skipif(r.find_binary() is None,
                                reason="dooly-collective not built")


def assert_close(got_ns: int, want_ns: float) -> None:
    """The model is deterministic and ns-3 runs at picosecond resolution, so
    the only slack needed is the 1 ns truncation of the reported integer."""
    assert abs(got_ns - want_ns) <= 1.0 + 1e-6 * want_ns, (got_ns, want_ns)


# --------------------------------------------------------------------------
# Single flow: serialization + propagation
# --------------------------------------------------------------------------
def test_single_flow_is_serialization_plus_propagation():
    size = 1 << 20
    res = r.run_collective(RING[2], collective="p2p", message_bytes=size)
    want = r.analytical_time_ns(RING[2], collective="p2p", message_bytes=size)
    assert_close(res["completion_time_ns"], want)
    assert res["bytes_transferred"] == size
    assert res["steps"] == 1
    assert res["dropped_packets"] == 0


# --------------------------------------------------------------------------
# Ring AllReduce on 2, 4 and 8 ranks across message sizes
# --------------------------------------------------------------------------
@pytest.mark.parametrize("ranks", [2, 4, 8])
@pytest.mark.parametrize("message_bytes", [64 << 10, 1 << 20, 16 << 20])
def test_ring_all_reduce_matches_analytical_model(ranks, message_bytes):
    res = r.run_collective(RING[ranks], collective="all_reduce", message_bytes=message_bytes)
    want = r.analytical_time_ns(RING[ranks], collective="all_reduce", message_bytes=message_bytes)
    assert res["ranks"] == ranks
    assert res["steps"] == 2 * (ranks - 1)
    assert_close(res["completion_time_ns"], want)
    assert res["bytes_transferred"] == res["steps"] * ranks * res["chunk_bytes"]
    assert res["dropped_packets"] == 0


@pytest.mark.parametrize("collective", ["reduce_scatter", "all_gather"])
def test_other_ring_collectives_match_analytical_model(collective):
    size = 1 << 20
    res = r.run_collective(RING[4], collective=collective, message_bytes=size)
    want = r.analytical_time_ns(RING[4], collective=collective, message_bytes=size)
    assert res["steps"] == 3
    assert_close(res["completion_time_ns"], want)
    if collective == "all_gather":
        assert res["chunk_bytes"] == size
    else:
        assert res["chunk_bytes"] == size // 4


def test_spec_example_byte_count():
    """The project spec's example: 16 MiB AllReduce over 8 ranks moves
    2(N-1) * S = 234881024 bytes."""
    res = r.run_collective(RING[8], collective="all_reduce", message_bytes=16777216)
    assert res["bytes_transferred"] == 234881024
    assert res["steps"] == 14


# --------------------------------------------------------------------------
# Topology effects and determinism
# --------------------------------------------------------------------------
def test_slow_link_changes_completion_predictably():
    """One 100 Gbps link in a 400 Gbps ring: the rank feeding that link
    serialises every step on it, exactly as the per-link recurrence predicts."""
    size = 4 << 20
    fast = r.run_collective(RING[4], collective="all_reduce", message_bytes=size)
    slow = r.run_collective(NET / "ring-4-slow-link.json", collective="all_reduce",
                            message_bytes=size)
    want = r.analytical_time_ns(NET / "ring-4-slow-link.json", collective="all_reduce",
                                message_bytes=size)
    assert slow["completion_time_ns"] > fast["completion_time_ns"]
    assert_close(slow["completion_time_ns"], want)


def test_queue_never_exceeds_configured_capacity():
    res = r.run_collective(RING[8], collective="all_reduce", message_bytes=16 << 20)
    assert 0 < res["max_queue_bytes"] <= 1048576


# --------------------------------------------------------------------------
# Switched topologies: forwarding, rank placement and contention
# --------------------------------------------------------------------------
@pytest.mark.parametrize("topology", ["star-8.json", "two-switch-8.json"])
def test_switched_topology_matches_multi_hop_model(topology):
    size = 16 << 20
    res = r.run_collective(NET / topology, collective="all_reduce", message_bytes=size)
    want = r.analytical_time_ns(NET / topology, collective="all_reduce", message_bytes=size)
    assert_close(res["completion_time_ns"], want)
    assert res["bytes_transferred"] == 234881024
    assert res["max_buffer_bytes"] == 0      # one flow per link direction: nothing queues
    assert res["dropped_packets"] == 0


def test_sub_ring_forwards_through_idle_ranks():
    """Ranks 0,2,4,6 on ring-8: every edge is two hops through a rank that
    is not in the collective and just forwards."""
    size = 4 << 20
    order = [0, 2, 4, 6]
    res = r.run_collective(RING[8], collective="all_reduce", message_bytes=size,
                           ring_order=order)
    want = r.analytical_time_ns(RING[8], collective="all_reduce", message_bytes=size,
                                ring_order=order)
    assert res["ring_order"] == order
    assert res["steps"] == 6
    assert_close(res["completion_time_ns"], want)


def test_contention_on_shared_uplink():
    """Interleaving the ring across two switches puts four ring edges on
    each direction of the inter-switch link.  Those four flows share one
    400 Gbps port, so every step takes about four serializations instead of
    one; the model's contention-free time is only a lower bound."""
    size = 16 << 20
    topo = NET / "two-switch-8.json"
    natural = r.run_collective(topo, collective="all_reduce", message_bytes=size)
    crossing = r.run_collective(topo, collective="all_reduce", message_bytes=size,
                                ring_order=[0, 4, 1, 5, 2, 6, 3, 7])
    bound = r.analytical_time_ns(topo, collective="all_reduce", message_bytes=size,
                                 ring_order=[0, 4, 1, 5, 2, 6, 3, 7])

    ratio = crossing["completion_time_ns"] / natural["completion_time_ns"]
    assert 3.5 < ratio < 4.2
    assert crossing["completion_time_ns"] > bound
    assert 0 < crossing["max_buffer_bytes"] <= 4194304   # buffered at the uplink port
    assert crossing["dropped_packets"] == 0               # lossless fabric


def test_runs_are_deterministic():
    kwargs = dict(collective="all_reduce", message_bytes=1 << 20)
    a = r.run_collective(RING[8], **kwargs)
    b = r.run_collective(RING[8], **kwargs)
    for k in ("wall_time_ms", "peak_rss_kb"):
        a.pop(k), b.pop(k)
    assert a == b


def test_missing_topology_reports_error():
    with pytest.raises(r.Ns3RunnerError, match="cannot open"):
        r.run_collective(NET / "does-not-exist.json", message_bytes=1024)
