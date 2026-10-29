/*
 * dooly-collective: packet-level simulation of one collective operation over a
 * configurable accelerator fabric, built on ns-3 point-to-point links and
 * drop-tail queues.
 *
 * DoolySim invokes this program once per collective it wants timed:
 *
 *   dooly-collective --topology=examples/network/ring-8.json \
 *                    --collective=all_reduce --message-bytes=16777216
 *
 * and reads a single JSON object from stdout (see ns3_backend/README.md for the
 * schema).  Nothing in here depends on DoolySim; it is a standalone ns-3
 * scenario so that the network model can be validated on its own.
 *
 * Fabric abstraction
 * ------------------
 *  - Every accelerator rank and every switch is an ns-3 Node.
 *  - Every link is a full-duplex PointToPointChannel with a DataRate, a
 *    propagation Delay and a per-direction DropTailQueue of queue_bytes (the
 *    link's transmit queue).
 *  - There is no IP stack.  Packets carry a 16-byte CollHeader (src rank,
 *    dst rank, step, seq, payload) and are forwarded hop by hop along
 *    shortest paths computed at start-up.
 *  - The fabric is lossless, like the credit-based flow control of NVLink or
 *    InfiniBand class interconnects.  Every port of a forwarding node has a
 *    buffer of buffer_bytes in front of its transmit queue, and a node only
 *    puts a packet on a link once the next hop has reserved room for it in
 *    the buffer of the port that packet will leave through.  Credits are
 *    granted instantaneously (no pause-frame latency is modelled).  Nothing
 *    is ever dropped; the drop counter in the output is a sanity check.
 *  - Reduction compute at each step is assumed to be free.
 *
 * Ring collectives
 * ----------------
 *  all_reduce      2(N-1) steps, each rank sends ceil(S/N) bytes to the next
 *  reduce_scatter    N-1  steps, each rank sends ceil(S/N) bytes to the next
 *  all_gather        N-1  steps, each rank sends S bytes to the next
 *  p2p               1    step,  the first rank sends S bytes to the second
 *
 * S is --message-bytes: the per-rank buffer handed to the collective.  The
 * ring order defaults to r0, r1, ..., r(N-1) and can be set with
 * --ring-order so that rank placement on the topology can be studied.  Step
 * k+1 at a rank starts only once that rank has received all of step k from
 * its predecessor, which is the data dependency of a ring reduction.
 */

#include "nlohmann/json.hpp"

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-module.h"

#include <sys/resource.h>

#include <chrono>
#include <deque>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

using namespace ns3;
using json = nlohmann::json;

namespace
{

constexpr int kSchemaVersion = 1;
constexpr uint16_t kProtocol = 0x0800; // PointToPointNetDevice only maps IPv4/IPv6 to PPP; the
                                       // payload is our own CollHeader, no IP is involved
constexpr uint32_t kPppHeaderBytes = 2; // added on the wire by PointToPointNetDevice

// ---------------------------------------------------------------------------
// Wire header carried by every packet.
// ---------------------------------------------------------------------------
class CollHeader : public Header
{
  public:
    static constexpr uint32_t kSize = 16;

    uint16_t src = 0;
    uint16_t dst = 0;
    uint16_t step = 0;
    uint16_t flags = 0;
    uint32_t seq = 0;
    uint32_t payload = 0;

    static TypeId GetTypeId()
    {
        static TypeId tid =
            TypeId("DoolyCollHeader").SetParent<Header>().AddConstructor<CollHeader>();
        return tid;
    }

    TypeId GetInstanceTypeId() const override
    {
        return GetTypeId();
    }

    uint32_t GetSerializedSize() const override
    {
        return kSize;
    }

    void Serialize(Buffer::Iterator i) const override
    {
        i.WriteHtonU16(src);
        i.WriteHtonU16(dst);
        i.WriteHtonU16(step);
        i.WriteHtonU16(flags);
        i.WriteHtonU32(seq);
        i.WriteHtonU32(payload);
    }

    uint32_t Deserialize(Buffer::Iterator i) override
    {
        src = i.ReadNtohU16();
        dst = i.ReadNtohU16();
        step = i.ReadNtohU16();
        flags = i.ReadNtohU16();
        seq = i.ReadNtohU32();
        payload = i.ReadNtohU32();
        return kSize;
    }

    void Print(std::ostream& os) const override
    {
        os << "src=" << src << " dst=" << dst << " step=" << step << " seq=" << seq
           << " payload=" << payload;
    }
};

// ---------------------------------------------------------------------------
// Topology description (parsed from JSON).
// ---------------------------------------------------------------------------
struct LinkSpec
{
    std::string a;
    std::string b;
    std::string rate;
    std::string delay;
    uint64_t queueBytes = 0;
    uint64_t bufferBytes = 0;
};

struct Topology
{
    std::string name;
    int ranks = 0;
    int switches = 0;
    uint32_t packetBytes = 4096;
    std::vector<LinkSpec> links;

    int NumNodes() const
    {
        return ranks + switches;
    }

    // "r<i>" is rank i (node i); "s<i>" is switch i (node ranks+i).
    int NodeIndex(const std::string& nm) const
    {
        if (nm.size() >= 2 && (nm[0] == 'r' || nm[0] == 's'))
        {
            int idx = std::stoi(nm.substr(1));
            if (nm[0] == 'r' && idx >= 0 && idx < ranks)
            {
                return idx;
            }
            if (nm[0] == 's' && idx >= 0 && idx < switches)
            {
                return ranks + idx;
            }
        }
        throw std::runtime_error("topology: unknown node name '" + nm + "'");
    }
};

Topology
LoadTopology(const std::string& path)
{
    std::ifstream in(path);
    if (!in)
    {
        throw std::runtime_error("topology: cannot open " + path);
    }
    json j = json::parse(in);

    int ver = j.value("schema_version", 0);
    if (ver != kSchemaVersion)
    {
        throw std::runtime_error("topology: schema_version " + std::to_string(ver) +
                                 " not supported (expected " + std::to_string(kSchemaVersion) +
                                 ")");
    }

    Topology t;
    t.name = j.value("name", path);
    t.ranks = j.at("ranks").get<int>();
    t.switches = j.value("switches", 0);
    t.packetBytes = j.value("packet_bytes", 4096u);
    if (t.ranks < 1 || t.packetBytes == 0)
    {
        throw std::runtime_error("topology: ranks must be >= 1 and packet_bytes > 0");
    }

    json defaults = j.value("defaults", json::object());
    auto addLink = [&](const std::string& a, const std::string& b, const json& o) {
        LinkSpec l;
        l.a = a;
        l.b = b;
        l.rate = o.value("rate", defaults.value("rate", ""));
        l.delay = o.value("delay", defaults.value("delay", ""));
        l.queueBytes = o.value("queue_bytes", defaults.value("queue_bytes", 0ull));
        l.bufferBytes = o.value("buffer_bytes", defaults.value("buffer_bytes", 4ull << 20));
        if (l.rate.empty() || l.delay.empty() || l.queueBytes == 0 || l.bufferBytes == 0)
        {
            throw std::runtime_error("topology: link " + a + "-" + b +
                                     " needs rate, delay, queue_bytes and buffer_bytes "
                                     "(or defaults)");
        }
        t.NodeIndex(a); // validate names early
        t.NodeIndex(b);
        t.links.push_back(l);
    };

    std::string gen = j.value("generator", "");
    if (gen == "ring")
    {
        // A 2-rank "ring" is a single full-duplex link.
        int n = t.ranks == 2 ? 1 : t.ranks;
        for (int i = 0; i < n; ++i)
        {
            addLink("r" + std::to_string(i),
                    "r" + std::to_string((i + 1) % t.ranks),
                    json::object());
        }
    }
    else if (gen == "star")
    {
        if (t.switches < 1)
        {
            throw std::runtime_error("topology: star generator needs switches >= 1");
        }
        for (int i = 0; i < t.ranks; ++i)
        {
            addLink("r" + std::to_string(i), "s0", json::object());
        }
    }
    else if (!gen.empty())
    {
        throw std::runtime_error("topology: unknown generator '" + gen + "'");
    }

    for (const auto& o : j.value("links", json::array()))
    {
        addLink(o.at("a").get<std::string>(), o.at("b").get<std::string>(), o);
    }
    if (t.links.empty())
    {
        throw std::runtime_error("topology: no links defined");
    }
    return t;
}

// ---------------------------------------------------------------------------
// The fabric: nodes, links, routing, lossless forwarding and statistics.
// ---------------------------------------------------------------------------
class Fabric
{
  public:
    // Called whenever payload arrives at `rank` from `src` for `step`;
    // `total` is the number of bytes received so far for that (src, step).
    std::function<void(int rank, int src, int step, uint64_t total)> onBytes;

    explicit Fabric(const Topology& t)
        : m_topo(t)
    {
    }

    void Build()
    {
        const int n = m_topo.NumNodes();
        m_nodes.Create(n);
        m_ports.assign(n, {});
        m_upstream.assign(n, {});
        m_outbox.assign(m_topo.ranks, {});

        for (const auto& l : m_topo.links)
        {
            int a = m_topo.NodeIndex(l.a);
            int b = m_topo.NodeIndex(l.b);

            PointToPointHelper p2p;
            p2p.SetDeviceAttribute("DataRate", StringValue(l.rate));
            p2p.SetDeviceAttribute("Mtu",
                                   UintegerValue(m_topo.packetBytes + CollHeader::kSize));
            p2p.SetChannelAttribute("Delay", StringValue(l.delay));
            p2p.SetQueue("ns3::DropTailQueue<Packet>",
                         "MaxSize",
                         StringValue(std::to_string(l.queueBytes) + "B"));
            NetDeviceContainer devs = p2p.Install(m_nodes.Get(a), m_nodes.Get(b));

            AddPort(a, b, devs.Get(0), l);
            AddPort(b, a, devs.Get(1), l);
        }

        ComputeRoutes();
    }

    // Queue `bytes` for transmission from `src` to `dst` as part of `step`.
    void Send(int src, int dst, int step, uint64_t bytes)
    {
        if (!PortToward(src, dst))
        {
            throw std::runtime_error("no route from r" + std::to_string(src) + " to r" +
                                     std::to_string(dst));
        }
        m_outbox[src].push_back(Message{dst, step, bytes});
        Simulator::ScheduleNow(&Fabric::Pump, this, src);
    }

    uint64_t BytesDelivered() const
    {
        return m_bytesDelivered;
    }

    uint64_t PacketsSent() const
    {
        return m_packetsSent;
    }

    uint32_t MaxQueueBytes() const
    {
        return m_maxQueueBytes;
    }

    uint64_t MaxBufferBytes() const
    {
        return m_maxBufferBytes;
    }

    uint64_t Drops() const
    {
        return m_drops;
    }

  private:
    struct Buffered
    {
        Ptr<Packet> pkt;
        int dst;
        uint32_t wire;
    };

    struct Port
    {
        Ptr<PointToPointNetDevice> dev;
        Ptr<Queue<Packet>> queue;
        int peer = -1;
        uint64_t queueCap = 0;  // link transmit queue (ns-3 DropTailQueue)
        uint64_t bufferCap = 0; // forwarding buffer in front of it
        std::deque<Buffered> pending;
        uint64_t pendingBytes = 0; // bytes held in `pending`
        uint64_t reserved = 0;     // bytes in flight towards this port, credit already granted
    };

    struct Message
    {
        int dst;
        int step;
        uint64_t bytes;
        uint64_t sent = 0;
        uint32_t seq = 0;
    };

    struct Upstream
    {
        int node;
        int port;
    };

    void AddPort(int node, int peer, Ptr<NetDevice> nd, const LinkSpec& l)
    {
        Port p;
        p.dev = DynamicCast<PointToPointNetDevice>(nd);
        p.queue = p.dev->GetQueue();
        p.peer = peer;
        p.queueCap = l.queueBytes;
        p.bufferCap = l.bufferBytes;
        p.dev->SetReceiveCallback(MakeCallback(&Fabric::Receive, this));

        int idx = static_cast<int>(m_ports[node].size());
        bool ok = true;
        ok &= p.queue->TraceConnectWithoutContext("BytesInQueue",
                                                  MakeCallback(&Fabric::OnBytesInQueue, this));
        ok &= p.queue->TraceConnectWithoutContext("Drop", MakeCallback(&Fabric::OnDrop, this));
        // Wake the port when its link drains so buffered packets (and the
        // rank's own messages) can follow.  Deferred to a fresh event so we
        // never re-enter the device's transmit state machine.
        ok &= p.queue->TraceConnectWithoutContext(
            "Dequeue",
            MakeBoundCallback(&Fabric::OnDequeue, this, node, idx));
        NS_ABORT_MSG_UNLESS(ok, "failed to connect queue trace sources");

        m_ports[node].push_back(p);
        m_upstream[peer].push_back(Upstream{node, idx});
    }

    // Shortest-path (hop count) next hop from every node to every rank.
    // Ties are broken by link order in the topology file, so routes are
    // reproducible.
    void ComputeRoutes()
    {
        const int n = m_topo.NumNodes();
        m_nextHop.assign(n, std::vector<int>(m_topo.ranks, -1));
        for (int srcNode = 0; srcNode < n; ++srcNode)
        {
            std::vector<int> firstPort(n, -1);
            std::vector<bool> seen(n, false);
            std::queue<int> q;
            seen[srcNode] = true;
            q.push(srcNode);
            while (!q.empty())
            {
                int u = q.front();
                q.pop();
                for (size_t pi = 0; pi < m_ports[u].size(); ++pi)
                {
                    int v = m_ports[u][pi].peer;
                    if (seen[v])
                    {
                        continue;
                    }
                    seen[v] = true;
                    firstPort[v] = (u == srcNode) ? static_cast<int>(pi) : firstPort[u];
                    q.push(v);
                }
            }
            for (int dst = 0; dst < m_topo.ranks; ++dst)
            {
                m_nextHop[srcNode][dst] = (dst == srcNode) ? -1 : firstPort[dst];
            }
        }
    }

    Port* PortToward(int node, int dst)
    {
        int pi = m_nextHop[node][dst];
        return pi < 0 ? nullptr : &m_ports[node][pi];
    }

    bool LinkHasRoom(const Port& out, uint32_t wire) const
    {
        return out.queue->GetNBytes() + wire <= out.queueCap;
    }

    // Credit check: may a packet for `dst` leave through `out` now?  The
    // next hop must have room for it in the buffer of the port it will
    // continue through.  A destination rank always accepts.
    bool DownstreamHasRoom(const Port& out, int dst, uint32_t wire)
    {
        if (out.peer == dst)
        {
            return true;
        }
        Port* next = PortToward(out.peer, dst);
        NS_ABORT_MSG_IF(!next, "node " << out.peer << " has no route to rank " << dst);
        return next->pendingBytes + next->reserved + wire <= next->bufferCap;
    }

    // Put a packet on the wire; the caller has checked link room and credit.
    void Transmit(Port& out, Ptr<Packet> pkt, int dst, uint32_t wire)
    {
        if (out.peer != dst)
        {
            PortToward(out.peer, dst)->reserved += wire;
        }
        bool ok = out.dev->Send(pkt, Address(), kProtocol);
        NS_ABORT_MSG_UNLESS(ok, "lossless send was dropped");
        ++m_packetsSent;
    }

    // Push as many packets of the rank's pending messages as link room and
    // downstream credit allow.  Forwarded traffic waiting on the same port
    // goes first (see Drain).
    void Pump(int rank)
    {
        auto& box = m_outbox[rank];
        while (!box.empty())
        {
            Message& m = box.front();
            Port& port = *PortToward(rank, m.dst);
            if (!port.pending.empty())
            {
                return; // Drain will call us once the forwarded packets are out
            }
            while (m.sent < m.bytes)
            {
                uint32_t payload = static_cast<uint32_t>(
                    std::min<uint64_t>(m_topo.packetBytes, m.bytes - m.sent));
                uint32_t wire = payload + CollHeader::kSize + kPppHeaderBytes;
                if (!LinkHasRoom(port, wire) || !DownstreamHasRoom(port, m.dst, wire))
                {
                    return; // woken by OnDequeue or by a downstream credit release
                }
                CollHeader h;
                h.src = static_cast<uint16_t>(rank);
                h.dst = static_cast<uint16_t>(m.dst);
                h.step = static_cast<uint16_t>(m.step);
                h.seq = m.seq++;
                h.payload = payload;
                Ptr<Packet> p = Create<Packet>(payload);
                p->AddHeader(h);
                Transmit(port, p, m.dst, wire);
                m.sent += payload;
            }
            box.pop_front();
        }
    }

    // Move buffered packets of one port onto its link while room and credit
    // allow, release the credit they held, then let the rank (if any) send.
    void Drain(int node, int portIdx)
    {
        Port& port = m_ports[node][portIdx];
        bool released = false;
        while (!port.pending.empty())
        {
            Buffered& b = port.pending.front();
            if (!LinkHasRoom(port, b.wire) || !DownstreamHasRoom(port, b.dst, b.wire))
            {
                break;
            }
            Transmit(port, b.pkt, b.dst, b.wire);
            port.pendingBytes -= b.wire;
            port.pending.pop_front();
            released = true;
        }
        if (released)
        {
            // Buffer space freed: upstream senders blocked on our credit may go.
            for (const Upstream& u : m_upstream[node])
            {
                Simulator::ScheduleNow(&Fabric::Drain, this, u.node, u.port);
            }
        }
        if (node < m_topo.ranks)
        {
            Pump(node);
        }
    }

    static void OnDequeue(Fabric* self, int node, int portIdx, Ptr<const Packet>)
    {
        Simulator::ScheduleNow(&Fabric::Drain, self, node, portIdx);
    }

    bool Receive(Ptr<NetDevice> dev, Ptr<const Packet> pkt, uint16_t, const Address&)
    {
        int node = static_cast<int>(dev->GetNode()->GetId());
        CollHeader h;
        pkt->PeekHeader(h);

        if (node == h.dst)
        {
            uint64_t& got = m_rx[std::make_tuple(node, h.src, h.step)];
            got += h.payload;
            m_bytesDelivered += h.payload;
            if (onBytes)
            {
                onBytes(node, h.src, h.step, got);
            }
            return true;
        }

        // Forward.  The sender reserved room for this packet in `out`'s
        // buffer before transmitting it; convert that reservation into
        // either an immediate transmission or a buffered packet.
        uint32_t wire = pkt->GetSize() + kPppHeaderBytes;
        Port* out = PortToward(node, h.dst);
        NS_ABORT_MSG_IF(!out, "node " << node << " has no route to rank " << h.dst);
        NS_ABORT_MSG_IF(out->reserved < wire, "credit accounting underflow");
        out->reserved -= wire;

        Ptr<Packet> copy = pkt->Copy();
        if (out->pending.empty() && LinkHasRoom(*out, wire) &&
            DownstreamHasRoom(*out, h.dst, wire))
        {
            Transmit(*out, copy, h.dst, wire);
        }
        else
        {
            out->pending.push_back(Buffered{copy, h.dst, wire});
            out->pendingBytes += wire;
            NS_ABORT_MSG_IF(out->pendingBytes > out->bufferCap, "forwarding buffer overflow");
            m_maxBufferBytes = std::max(m_maxBufferBytes, out->pendingBytes);
        }
        return true;
    }

    void OnBytesInQueue(uint32_t, uint32_t now)
    {
        m_maxQueueBytes = std::max(m_maxQueueBytes, now);
    }

    void OnDrop(Ptr<const Packet>)
    {
        ++m_drops;
    }

    Topology m_topo;
    NodeContainer m_nodes;
    std::vector<std::vector<Port>> m_ports;         // [node] -> ports
    std::vector<std::vector<Upstream>> m_upstream;  // [node] -> ports on neighbours facing it
    std::vector<std::vector<int>> m_nextHop;        // [node][dstRank] -> port index or -1
    std::vector<std::deque<Message>> m_outbox;      // [rank] -> pending messages
    std::map<std::tuple<int, int, int>, uint64_t> m_rx; // (rank, src, step) -> bytes

    uint64_t m_bytesDelivered = 0;
    uint64_t m_packetsSent = 0;
    uint32_t m_maxQueueBytes = 0;
    uint64_t m_maxBufferBytes = 0;
    uint64_t m_drops = 0;
};

// ---------------------------------------------------------------------------
// Ring collective driver.
// ---------------------------------------------------------------------------
struct CollectiveSpec
{
    std::string type;      // all_reduce | reduce_scatter | all_gather | p2p
    std::string algorithm; // ring
    std::vector<int> order; // participating ranks in ring order
    uint64_t messageBytes = 0;

    int Ranks() const
    {
        return static_cast<int>(order.size());
    }

    int Steps() const
    {
        if (type == "all_reduce")
        {
            return 2 * (Ranks() - 1);
        }
        if (type == "reduce_scatter" || type == "all_gather")
        {
            return Ranks() - 1;
        }
        if (type == "p2p")
        {
            return 1;
        }
        throw std::runtime_error("unknown collective '" + type + "'");
    }

    uint64_t ChunkBytes() const
    {
        if (type == "all_gather" || type == "p2p")
        {
            return messageBytes;
        }
        return (messageBytes + Ranks() - 1) / Ranks(); // ceil(S / N)
    }
};

class RingCollective
{
  public:
    RingCollective(Fabric& f, const CollectiveSpec& s, int topologyRanks)
        : m_fabric(f),
          m_spec(s),
          m_steps(s.Steps()),
          m_chunk(s.ChunkBytes()),
          m_pos(topologyRanks, -1),
          m_done(topologyRanks, 0)
    {
        for (int i = 0; i < m_spec.Ranks(); ++i)
        {
            m_pos[m_spec.order[i]] = i;
        }
        m_fabric.onBytes = [this](int rank, int src, int step, uint64_t total) {
            OnBytes(rank, src, step, total);
        };
    }

    void Start()
    {
        if (m_spec.type == "p2p")
        {
            m_fabric.Send(m_spec.order[0], m_spec.order[1], 0, m_chunk);
            return;
        }
        for (int r : m_spec.order)
        {
            m_fabric.Send(r, Next(r), 0, m_chunk);
        }
    }

    bool Completed() const
    {
        return m_completed;
    }

    Time CompletionTime() const
    {
        return m_completion;
    }

    int Steps() const
    {
        return m_steps;
    }

    uint64_t Chunk() const
    {
        return m_chunk;
    }

  private:
    int Next(int r) const
    {
        return m_spec.order[(m_pos[r] + 1) % m_spec.Ranks()];
    }

    int Prev(int r) const
    {
        return m_spec.order[(m_pos[r] + m_spec.Ranks() - 1) % m_spec.Ranks()];
    }

    void OnBytes(int rank, int src, int step, uint64_t total)
    {
        if (total < m_chunk)
        {
            return;
        }
        NS_ABORT_MSG_IF(total > m_chunk, "rank " << rank << " over-received step " << step);
        NS_ABORT_MSG_IF(m_pos[rank] < 0, "rank " << rank << " is not in the collective");
        NS_ABORT_MSG_IF(m_spec.type != "p2p" && src != Prev(rank),
                        "rank " << rank << " received step " << step << " from r" << src);
        NS_ABORT_MSG_IF(m_done[rank] != step, "rank " << rank << " got step " << step
                                                      << " before step " << m_done[rank]);
        m_done[rank] = step + 1;

        if (step + 1 < m_steps)
        {
            m_fabric.Send(rank, Next(rank), step + 1, m_chunk);
            return;
        }

        int needed = (m_spec.type == "p2p") ? 1 : m_spec.Ranks();
        if (++m_finished == needed)
        {
            m_completed = true;
            m_completion = Simulator::Now();
            Simulator::Stop();
        }
    }

    Fabric& m_fabric;
    CollectiveSpec m_spec;
    int m_steps;
    uint64_t m_chunk;
    std::vector<int> m_pos;  // rank -> position in ring order, -1 if absent
    std::vector<int> m_done; // rank -> steps fully received
    int m_finished = 0;
    bool m_completed = false;
    Time m_completion;
};

std::vector<int>
ParseRingOrder(const std::string& text, uint32_t ranks, int topologyRanks)
{
    std::vector<int> order;
    if (!text.empty())
    {
        std::stringstream ss(text);
        std::string tok;
        while (std::getline(ss, tok, ','))
        {
            order.push_back(std::stoi(tok));
        }
    }
    else
    {
        int n = ranks == 0 ? topologyRanks : static_cast<int>(ranks);
        for (int i = 0; i < n; ++i)
        {
            order.push_back(i);
        }
    }

    std::set<int> seen;
    for (int r : order)
    {
        if (r < 0 || r >= topologyRanks || !seen.insert(r).second)
        {
            throw std::runtime_error("ring order must list distinct ranks within the topology");
        }
    }
    if (order.size() < 2)
    {
        throw std::runtime_error("a collective needs at least two ranks");
    }
    if (order.size() > 65535)
    {
        throw std::runtime_error("too many ranks for the wire header");
    }
    return order;
}

long
PeakRssKb()
{
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_maxrss;
}

} // namespace

int
main(int argc, char* argv[])
{
    Time::SetResolution(Time::PS);

    std::string topologyPath;
    std::string collective = "all_reduce";
    std::string algorithm = "ring";
    std::string ringOrder;
    uint32_t ranks = 0;
    uint64_t messageBytes = 0;
    uint32_t seed = 1;
    double maxSimSeconds = 10.0;

    CommandLine cmd(__FILE__);
    cmd.AddValue("topology", "Path to a topology JSON file", topologyPath);
    cmd.AddValue("collective", "all_reduce | reduce_scatter | all_gather | p2p", collective);
    cmd.AddValue("algorithm", "Collective algorithm (ring)", algorithm);
    cmd.AddValue("ranks", "Participating ranks r0..r(N-1); 0 = every rank in the topology", ranks);
    cmd.AddValue("ring-order",
                 "Comma-separated ranks in ring order, e.g. 0,4,1,5 (overrides --ranks)",
                 ringOrder);
    cmd.AddValue("message-bytes", "Per-rank buffer size handed to the collective", messageBytes);
    cmd.AddValue("seed", "RNG seed (the model is deterministic; kept for reproducibility)", seed);
    cmd.AddValue("max-sim-seconds", "Abort if the collective has not finished by then", maxSimSeconds);
    cmd.Parse(argc, argv);

    try
    {
        if (topologyPath.empty() || messageBytes == 0)
        {
            throw std::runtime_error("--topology and --message-bytes are required");
        }
        if (algorithm != "ring")
        {
            throw std::runtime_error("unsupported algorithm '" + algorithm + "'");
        }

        RngSeedManager::SetSeed(seed);
        RngSeedManager::SetRun(1);

        Topology topo = LoadTopology(topologyPath);

        CollectiveSpec spec;
        spec.type = collective;
        spec.algorithm = algorithm;
        spec.order = ParseRingOrder(ringOrder, ranks, topo.ranks);
        spec.messageBytes = messageBytes;
        spec.Steps(); // validates the collective name

        Fabric fabric(topo);
        fabric.Build();
        RingCollective coll(fabric, spec, topo.ranks);

        auto wallStart = std::chrono::steady_clock::now();
        Simulator::Schedule(Seconds(0), &RingCollective::Start, &coll);
        Simulator::Stop(Seconds(maxSimSeconds));
        Simulator::Run();
        Simulator::Destroy();
        auto wallEnd = std::chrono::steady_clock::now();

        json out;
        out["schema_version"] = kSchemaVersion;
        out["topology"] = topo.name;
        out["collective"] = spec.type;
        out["algorithm"] = spec.algorithm;
        out["ranks"] = spec.Ranks();
        out["ring_order"] = spec.order;
        out["message_bytes"] = spec.messageBytes;
        out["chunk_bytes"] = coll.Chunk();
        out["steps"] = coll.Steps();
        out["packet_bytes"] = topo.packetBytes;
        out["completed"] = coll.Completed();
        out["completion_time_ns"] =
            coll.Completed() ? coll.CompletionTime().GetNanoSeconds() : -1;
        out["bytes_transferred"] = fabric.BytesDelivered();
        out["packets_sent"] = fabric.PacketsSent();
        out["max_queue_bytes"] = fabric.MaxQueueBytes();
        out["max_buffer_bytes"] = fabric.MaxBufferBytes();
        out["dropped_packets"] = fabric.Drops();
        out["wall_time_ms"] =
            std::chrono::duration<double, std::milli>(wallEnd - wallStart).count();
        out["peak_rss_kb"] = PeakRssKb();
        std::cout << out.dump() << std::endl;
        return coll.Completed() ? 0 : 3;
    }
    catch (const std::exception& e)
    {
        json err;
        err["schema_version"] = kSchemaVersion;
        err["error"] = e.what();
        std::cout << err.dump() << std::endl;
        return 2;
    }
}
