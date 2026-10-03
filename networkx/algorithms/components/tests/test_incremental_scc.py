"""Tests for the insertion-only SCC maintainer."""

import random

import pytest

import networkx as nx
from networkx.algorithms.components.incremental_scc import (
    IncrementalSCC,
    StaleComponentError,
    UnknownComponentError,
)


# ------------------------------------------------------------- basic behaviour


def test_initial_partition_matches_tarjan():
    G = nx.DiGraph(
        [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 3), (4, 5)]
    )  # SCCs {0,1,2}, {3,4} and singleton {5}
    m = IncrementalSCC(G)
    assert m.verify()
    assert m.number_of_components() == 3
    ref = nx.condensation(G)
    assert m.condensation_graph().number_of_edges() == ref.number_of_edges()
    # initial ids follow nx.strongly_connected_components enumeration
    initial_ids = [m.component_of(n) for n in G]
    assert sorted(set(initial_ids)) == list(range(m.number_of_components()))
    ref_components = list(nx.strongly_connected_components(G))
    for cid, component in enumerate(ref_components):
        for n in component:
            assert m.component_of(n) == cid


def test_add_node_creates_singleton():
    m = IncrementalSCC(nx.DiGraph())
    assert m.add_node("a") is True
    assert m.add_node("a") is False  # repeated insert is a no-op
    assert m.add_node("b") is True
    assert m.number_of_components() == 2
    assert m.component(m.component_of("a")) == {"a"}
    assert m.verify()


def test_add_edge_missing_endpoints_auto_added():
    m = IncrementalSCC(nx.DiGraph())
    assert m.add_edge("a", "b") == "acyclic"
    assert m.number_of_components() == 2
    assert m.add_edge("b", "a") == "merged"
    assert m.number_of_components() == 1
    assert m.verify()


def test_status_internal_duplicate_redundant():
    G = nx.DiGraph()
    m = IncrementalSCC(G)
    assert m.add_edge(0, 1) == "acyclic"
    # parallel edge between the same two SCCs, condensation edge already there
    assert m.add_edge(0, 1) == "duplicate"
    # second distinct raw edge between the same pair of SCCs
    G2 = nx.DiGraph(
        [(0, 1), (1, 0), (2, 3), (3, 2), (0, 2)]
    )  # SCCs {0,1} and {2,3}, condensation edge A -> B already present
    m2 = IncrementalSCC(G2)
    assert m2.add_edge(1, 3) == "redundant"
    # edge inside an SCC after it formed
    m3 = IncrementalSCC(nx.DiGraph())
    m3.add_edge("x", "y")
    m3.add_edge("y", "x")
    assert m3.add_edge("x", "y") == "duplicate"
    assert m3.add_edge("y", "x") == "duplicate"
    assert m3.verify()


def test_self_loop_is_internal():
    m = IncrementalSCC(nx.DiGraph())
    assert m.add_edge(0, 0) == "internal"
    assert m.add_edge(0, 0) == "duplicate"
    assert m.number_of_components() == 1
    assert m.verify()


# ------------------------------------------------------------ id stability / tombstones


def test_unaffected_ids_remain_stable():
    G = nx.DiGraph()
    m = IncrementalSCC(G)
    m.add_edges_from([(0, 1), (2, 3), (4, 5)])
    before = {n: m.component_of(n) for n in range(6)}
    # close 0<->1; the other components must keep their ids
    status = m.add_edge(1, 0)
    assert status == "merged"
    new_id = m.component_of(0)
    assert new_id != before[0]
    for n in (2, 3, 4, 5):
        assert m.component_of(n) == before[n]
    expected_ids = sorted(
        [before[2], before[3], before[4], before[5], new_id]
    )
    assert sorted(m.component_ids()) == expected_ids
    assert m.is_stale(before[0]) and m.is_stale(before[1])
    assert m.verify()


def test_tombstone_lookup_rules():
    m = IncrementalSCC(nx.DiGraph())
    m.add_edges_from([(0, 1), (1, 2)])
    old = [m.component_of(i) for i in range(3)]
    m.add_edge(2, 0)
    new = m.component_of(0)

    assert m.resolve_cid(old[0]) == new
    assert m.resolve_cid(new) == new
    assert m.is_stale(old[1]) is True
    with pytest.raises(StaleComponentError) as exc:
        m.component(old[1])
    assert exc.value.successor == new
    with pytest.raises(UnknownComponentError):
        m.component(999)
    with pytest.raises(UnknownComponentError):
        m.resolve_cid(999)
    # ids are never reused
    assert new not in old
    assert set(old).isdisjoint(m.component_ids())


def test_chained_merges_follow_tombstone_chain():
    G = nx.DiGraph()
    m = IncrementalSCC(G)
    m.add_edges_from([(0, 1), (2, 3), (4, 5)])
    first_ids = {n: m.component_of(n) for n in range(6)}
    m.add_edge(1, 0)  # merge {0,1}
    mid_01 = m.component_of(0)
    assert m.resolve_cid(first_ids[0]) == mid_01
    # merge the new SCC with {2,3}: connect 1 -> 2 then 3 -> 0
    m.add_edge(1, 2)
    m.add_edge(3, 0)
    final = m.component_of(0)
    assert final != mid_01
    for old_id in (
        first_ids[0],
        first_ids[1],
        first_ids[2],
        first_ids[3],
        mid_01,
    ):
        assert m.resolve_cid(old_id) == final
    # components of 4 and 5 stay untouched (and remain distinct SCCs)
    assert m.component_of(4) == first_ids[4]
    assert m.component_of(5) == first_ids[5]
    assert first_ids[4] != first_ids[5]
    assert m.verify()


# ------------------------------------------------------------------- batching


def test_add_nodes_from_skips_duplicates():
    m = IncrementalSCC(nx.DiGraph())
    added = m.add_nodes_from([1, 2, 2, 3])
    assert added == 3
    assert m.number_of_components() == 3


def test_add_edges_from_counter():
    m = IncrementalSCC(nx.DiGraph())
    result = m.add_edges_from([(0, 1), (1, 2), (2, 0), (0, 1), (3, 3)])
    assert result["acyclic"] == 2
    assert result["merged"] == 1
    assert result["duplicate"] == 1
    assert result["internal"] == 1
    assert m.verify()


def test_duplicate_edge_updates_attributes():
    m = IncrementalSCC(nx.DiGraph())
    m.add_edge(0, 1, weight=3)
    assert m.add_edge(0, 1, color="red") == "duplicate"
    assert m.graph[0][1]["weight"] == 3
    assert m.graph[0][1]["color"] == "red"
    assert m.number_of_components() == 2
    assert m.verify()


def test_init_with_nonempty_graph():
    G = nx.DiGraph([(0, 1), (1, 0), (1, 2)])
    m = IncrementalSCC(G)
    assert m.number_of_components() == 2
    assert m.component_of(0) == m.component_of(1)
    assert m.verify()


def test_batch_with_3_tuples():
    m = IncrementalSCC(nx.DiGraph())
    m.add_edges_from([(0, 1, {"w": 2}), (1, 2, {"label": "x"})])
    assert m.graph[0][1]["w"] == 2
    assert m.verify()


# ------------------------------------------------------- external graph changes


def test_graph_is_frozen_against_direct_mutation():
    G = nx.DiGraph([(0, 1)])
    m = IncrementalSCC(G)
    with pytest.raises(nx.NetworkXError):
        G.add_edge(1, 2)
    with pytest.raises(nx.NetworkXError):
        G.add_node(9)
    with pytest.raises(nx.NetworkXError):
        G.remove_edge(0, 1)
    # maintainer still works on the bypass path
    assert m.add_edge(1, 0) == "merged"
    assert m.verify()


def test_views_and_non_digraphs_rejected():
    base = nx.DiGraph([(0, 1)])
    with pytest.raises(TypeError):
        IncrementalSCC(nx.reverse_view(base))
    # an explicit copy is accepted
    IncrementalSCC(base.reverse())
    with pytest.raises(TypeError):
        IncrementalSCC(base.subgraph([0, 1]))
    with pytest.raises(nx.NetworkXNotImplemented):
        IncrementalSCC(nx.Graph([(0, 1)]))


def test_unfrozen_external_mutation_detected_by_verify():
    G = nx.DiGraph([(0, 1)])
    m = IncrementalSCC(G, freeze_input=False)
    # caller bypasses the maintainer and edits the graph directly
    G.add_edge(1, 0)
    with pytest.raises(AssertionError):
        m.verify()


def test_condensation_snapshot_is_independent():
    m = IncrementalSCC(nx.DiGraph([(0, 1)]))
    snap = m.condensation_graph()
    assert nx.is_frozen(snap)
    m.add_edge(1, 0)
    assert snap.number_of_nodes() == 2  # snapshot unchanged after a merge


# ----------------------------------------------------------------- fuzz checks


def _partition_of(G):
    return sorted(
        (sorted(c) for c in nx.strongly_connected_components(G)),
        key=sorted,
    )


@pytest.mark.parametrize("seed", [0, 1, 7, 23, 101])
def test_fuzz_compare_with_full_scc(seed):
    rng = random.Random(seed)
    m = IncrementalSCC(nx.DiGraph())
    nodes = list(range(40))
    for step in range(400):
        u, v = rng.sample(nodes, 2) if rng.random() < 0.7 else (
            rng.choice(nodes),
            rng.choice(nodes),
        )
        m.add_edge(u, v)
        if step % 25 == 0:
            assert m.verify()
    assert m.verify()
    # condensation as unlabeled DAG has same edges/order size as reference
    ref = nx.condensation(m.graph)
    got = m.condensation_graph()
    assert got.number_of_edges() == ref.number_of_edges()
    assert got.number_of_nodes() == ref.number_of_nodes()
    assert nx.is_directed_acyclic_graph(got)


def test_fuzz_nodes_and_edges_mixed():
    rng = random.Random(54321)
    m = IncrementalSCC(nx.DiGraph())
    next_node = 0
    for _ in range(300):
        if rng.random() < 0.3:
            m.add_node(next_node)
            next_node += 1
        elif m.number_of_components() and next_node > 1:
            u = rng.randrange(next_node)
            v = rng.randrange(next_node)
            m.add_edge(u, v)
        if next_node % 40 == 0 and next_node:
            assert m.verify()
    assert m.verify()


def test_fuzz_duplicates_and_batches_agree():
    rng = random.Random(909)
    edges = [(rng.randrange(15), rng.randrange(15)) for _ in range(120)]
    m = IncrementalSCC(nx.DiGraph())
    m.add_edges_from(edges)
    m.add_edges_from(edges)  # everything is a duplicate the second time
    ref = nx.DiGraph(edges)
    assert _partition_of(m.graph) == _partition_of(ref)
    assert m.verify()


# -------------------------------------------------------------- locality checks


def test_acyclic_insertion_between_isolated_nodes_is_constant_work():
    # N isolated components, then connect two fresh isolates acyclically.
    counts = {}
    for n in (100, 1000, 10000):
        G = nx.DiGraph()
        G.add_nodes_from(range(n))
        m = IncrementalSCC(G)
        m.reset_probe_visits()
        status = m.add_edge(0, 1)
        assert status == "acyclic"
        counts[n] = m.probe_visits
        assert m.verify()
    assert counts[100] == counts[1000] == counts[10000]
    # the probe expands only source's succs (empty) and target's preds (empty)
    assert counts[10000] <= 2


def test_acyclic_insertion_between_disjoint_chains_is_local():
    # N disjoint chains, each a fixed-size SCC-free region.  Connecting two
    # unrelated chains with an acyclic edge explores a constant number of
    # components regardless of N.
    counts = {}
    for n in (50, 500, 5000):
        G = nx.DiGraph()
        for k in range(n):
            base = 4 * k
            G.add_edges_from(
                [(base, base + 1), (base + 1, base + 2), (base + 2, base + 3)]
            )
        m = IncrementalSCC(G)
        assert m.number_of_components() == 4 * n
        m.reset_probe_visits()
        # edge from the end of chain 0 to the start of chain 1: no path back
        status = m.add_edge(3, 4)
        assert status == "acyclic"
        counts[n] = m.probe_visits
        assert m.verify()
    assert counts[50] == counts[500] == counts[5000]
    assert counts[5000] <= 8


def test_merge_work_is_local_to_the_cycle():
    # N disjoint chains; close the cycle inside one chain.  Work depends only
    # on the chain length, not on N.
    counts = {}
    chain_len = 6
    for n in (50, 1000, 4000):
        G = nx.DiGraph()
        for k in range(n):
            base = chain_len * k
            for j in range(chain_len - 1):
                G.add_edge(base + j, base + j + 1)
        m = IncrementalSCC(G)
        m.reset_probe_visits()
        status = m.add_edge(chain_len - 1, 0)  # close first chain
        assert status == "merged"
        counts[n] = m.probe_visits
        assert m.component_of(0) == m.component_of(chain_len - 1)
        assert m.number_of_components() == chain_len * n - (chain_len - 1)
        assert m.verify()
    assert counts[50] == counts[1000] == counts[4000]
