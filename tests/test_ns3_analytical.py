"""Pure-Python checks of the analytical ring model and unit parsing.

These never touch ns-3; the packet simulation itself is validated against
this model in test_ns3_ring.py.
"""

import pytest

from doolysim.network import ns3_runner as r


@pytest.mark.parametrize("rate,bps", [("400Gbps", 400e9), ("100Gb/s", 100e9),
                                      ("50GB/s", 400e9), ("1Mbps", 1e6)])
def test_parse_rate(rate, bps):
    assert r.parse_rate_bps(rate) == pytest.approx(bps)


@pytest.mark.parametrize("delay,ns", [("500ns", 500), ("1us", 1000),
                                      ("2.5ms", 2.5e6), ("1s", 1e9)])
def test_parse_time(delay, ns):
    assert r.parse_time_ns(delay) == pytest.approx(ns)


@pytest.mark.parametrize("bad", ["400", "fast", "1 parsec"])
def test_bad_units_rejected(bad):
    with pytest.raises(ValueError):
        r.parse_rate_bps(bad)
    with pytest.raises(ValueError):
        r.parse_time_ns(bad)


def test_wire_bytes_counts_headers_per_packet():
    assert r.wire_bytes(4096, 4096) == 4096 + 18
    assert r.wire_bytes(4097, 4096) == 4097 + 2 * 18


@pytest.mark.parametrize("collective,ranks,steps", [
    ("all_reduce", 8, 14), ("reduce_scatter", 8, 7), ("all_gather", 8, 7),
    ("all_reduce", 2, 2), ("p2p", 2, 1),
])
def test_ring_steps(collective, ranks, steps):
    assert r.ring_steps(collective, ranks) == steps


def test_chunk_bytes():
    assert r.chunk_bytes("all_reduce", 16777216, 8) == 2097152
    assert r.chunk_bytes("all_reduce", 1001, 4) == 251       # ceil
    assert r.chunk_bytes("all_gather", 1001, 4) == 1001
    assert r.chunk_bytes("p2p", 1001, 4) == 1001


def test_uniform_ring_is_steps_times_link_time():
    assert r.ring_completion_ns([(10.0, 1.0)] * 4, 6) == pytest.approx(66.0)


def test_slow_link_serialises_every_step_that_crosses_it():
    # Link r2->r3 is four times slower.  Rank 2 cannot start a step before
    # its previous chunk has left that link, so it sends six chunks back to
    # back: 6 * 40 + one propagation delay.
    links = [(10.0, 1.0), (10.0, 1.0), (40.0, 1.0), (10.0, 1.0)]
    assert r.ring_completion_ns(links, 6) == pytest.approx(241.0)


DEFAULTS = {"rate": "400Gbps", "delay": "500ns", "queue_bytes": 1 << 20}
RING4 = {"schema_version": 1, "ranks": 4, "packet_bytes": 4096,
         "generator": "ring", "defaults": DEFAULTS}
STAR4 = {"schema_version": 1, "ranks": 4, "switches": 1, "packet_bytes": 4096,
         "generator": "star", "defaults": DEFAULTS}
TWO_SWITCH = {
    "schema_version": 1, "ranks": 4, "switches": 2, "packet_bytes": 4096,
    "defaults": DEFAULTS,
    "links": [{"a": "r0", "b": "s0"}, {"a": "r1", "b": "s0"},
              {"a": "r2", "b": "s1"}, {"a": "r3", "b": "s1"},
              {"a": "s0", "b": "s1", "rate": "100Gbps"}],
}


def test_analytical_time_reads_topology_dict():
    chunk = 1 << 20
    step = r.wire_bytes(chunk, 4096) * 8 / 400e9 * 1e9 + 500
    assert r.analytical_time_ns(RING4, collective="all_reduce",
                                message_bytes=4 * chunk) == pytest.approx(6 * step)


def test_routes_follow_link_order_and_shortest_paths():
    g = r.Graph(TWO_SWITCH)
    assert [l["rate"] for l in g.route(0, 1)] == ["400Gbps", "400Gbps"]        # r0-s0-r1
    assert [l["rate"] for l in g.route(0, 3)] == ["400Gbps", "100Gbps", "400Gbps"]
    assert g.route(2, 2) == []
    assert len(r.Graph(RING4).route(0, 2)) == 2


def test_multi_hop_path_is_pipelined_store_and_forward():
    hop = {"rate": "400Gbps", "delay": "500ns"}
    chunk = 1 << 20
    busy, arrival = r.path_transfer_ns(chunk, 4096, [hop, hop])
    ser = r.wire_bytes(chunk, 4096) * 8 / 400e9 * 1e9
    pkt = (4096 + r.HEADER_BYTES) * 8 / 400e9 * 1e9
    assert busy == pytest.approx(ser)
    # second hop adds one packet serialization plus its propagation delay
    assert arrival == pytest.approx(ser + pkt + 2 * 500)


def test_star_adds_one_hop_per_step():
    size = 4 << 20
    ring = r.analytical_time_ns(RING4, collective="all_reduce", message_bytes=size)
    star = r.analytical_time_ns(STAR4, collective="all_reduce", message_bytes=size)
    pkt = (4096 + r.HEADER_BYTES) * 8 / 400e9 * 1e9
    assert star - ring == pytest.approx(6 * (pkt + 500))


def test_ring_order_changes_which_links_are_used():
    size = 4 << 20
    natural = r.analytical_time_ns(TWO_SWITCH, collective="all_reduce", message_bytes=size)
    crossing = r.analytical_time_ns(TWO_SWITCH, collective="all_reduce", message_bytes=size,
                                    ring_order=[0, 2, 1, 3])
    assert crossing > natural


def test_sub_ring_and_p2p_use_ring_order():
    size = 1 << 20
    two = r.analytical_time_ns(RING4, collective="p2p", message_bytes=size, ring_order=[0, 2])
    one = r.analytical_time_ns(RING4, collective="p2p", message_bytes=size, ring_order=[0, 1])
    assert two > one
