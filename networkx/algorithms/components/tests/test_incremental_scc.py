import random

import pytest

import networkx as nx
from networkx.algorithms.components import IncrementalSCC
from networkx import NetworkXError, NetworkXNotImplemented


def _scc_partition(G):
    return {frozenset(c) for c in nx.strongly_connected_components(G)}


def _assert_matches(m, G):
    assert m.verify()
    assert {
        frozenset(members) for _, members in m.components()
    } == _scc_partition(G)
    for cid, members in m.components():
        assert m.resolve(cid) == cid
        for node in members:
            assert m.component_of(node) == cid
    C = m.condensation()
    assert nx.is_directed_acyclic_graph(C)
    assert m.number_of_components() == C.number_of_nodes()


class TestIncrementalSCC:
    def test_path_then_back_edge(self):
        G = nx.path_graph(5, create_using=nx.DiGraph)
        m = IncrementalSCC(G)
        _assert_matches(m, G)
        assert m.number_of_components() == 5
        assert m.add_edge(4, 0) == "merged"
        G.add_edge(4, 0)
        _assert_matches(m, G)
        assert m.number_of_components() == 1
        assert m.members(0) == set(range(5))

    def test_acyclic_insert_keeps_components(self):
        G = nx.DiGraph([(0, 1), (2, 3)])
        m = IncrementalSCC(G)
        assert m.add_edge(1, 2) == "acyclic"
        G.add_edge(1, 2)
        _assert_matches(m, G)
        assert m.number_of_components() == 4

    def test_partial_merge_in_chain(self):
        # 0 -> 1 -> 2 -> 3 ; closing 2 -> 1 merges {1, 2} only.
        G = nx.path_graph(4, create_using=nx.DiGraph)
        m = IncrementalSCC(G)
        assert m.add_edge(2, 1) == "merged"
        G.add_edge(2, 1)
        _assert_matches(m, G)
        merged = m.members(m.component_of(1))
        assert merged == {1, 2}
        assert m.number_of_components() == 3

    def test_self_loop_and_internal_edge(self):
        G = nx.path_graph(3, create_using=nx.DiGraph)
        m = IncrementalSCC(G)
        assert m.add_edge(0, 0) == "internal"
        # redundant intra-component edge
        assert m.add_edge(0, 0) == "internal"
        assert m.last_visit_count == 0
        G.add_edge(0, 0)
        _assert_matches(m, G)

    def test_duplicate_edges(self):
        m = IncrementalSCC(nx.DiGraph())
        assert m.add_edge(0, 1) == "acyclic"
        # a duplicate edge between still-distinct SCCs changes nothing
        assert m.add_edge(0, 1) == "acyclic"
        assert m.last_visit_count == 0
        assert m.add_edge(1, 0) == "merged"
        assert m.add_edge(0, 1) == "internal"
        assert m.add_edge(1, 0) == "internal"
        H = nx.DiGraph([(0, 1), (1, 0)])
        _assert_matches(m, H)

    def test_duplicate_condensation_arc(self):
        # Two distinct edges crossing the same pair of SCCs add no arc twice.
        G = nx.DiGraph([(0, 2), (1, 2)])
        m = IncrementalSCC(G)
        assert m.add_edge(0, 3) == "acyclic"
        G.add_edge(0, 3)
        assert m.add_edge(1, 3) == "acyclic"
        G.add_edge(1, 3)
        # arcs: {0}->{2}, {1}->{2}, {0}->{3}, {1}->{3}
        assert m.condensation().number_of_edges() == 4
        _assert_matches(m, G)

    def test_add_nodes(self):
        m = IncrementalSCC(nx.DiGraph())
        assert m.add_node("a") is True
        assert m.add_node("a") is False
        assert m.add_nodes_from(["a", "b", "c"]) == 2
        assert m.number_of_components() == 3
        assert m.verify()

    def test_add_edge_creates_endpoints(self):
        m = IncrementalSCC(nx.DiGraph())
        m.add_edge("x", "y")
        assert m.component_of("x") != m.component_of("y")
        assert m.verify()

    def test_batch_operations(self):
        m = IncrementalSCC(nx.DiGraph())
        counts = m.add_edges_from([(0, 1), (1, 2), (2, 0), (0, 1)])
        # after the cycle closes, the last (0, 1) is inside the single SCC
        assert counts == {"internal": 1, "acyclic": 2, "merged": 1}
        assert m.number_of_components() == 1
        assert m.add_nodes_from([3, 4, 3]) == 2
        G = nx.DiGraph([(0, 1), (1, 2), (2, 0)])
        G.add_nodes_from([3, 4])
        _assert_matches(m, G)

    def test_batch_bad_edge_tuple(self):
        m = IncrementalSCC(nx.DiGraph())
        with pytest.raises(NetworkXError):
            m.add_edges_from([(0,)])

    def test_unstable_ids_alias_chain(self):
        # Build two disjoint 2-cycles, link them, then close the cycle so
        # a component that itself absorbed another is absorbed again.
        G = nx.DiGraph([(0, 1), (2, 3)])
        m = IncrementalSCC(G)
        a0 = m.component_of(0)
        a1 = m.component_of(1)
        b0 = m.component_of(2)
        b1 = m.component_of(3)
        assert m.add_edge(1, 0) == "merged"  # {0,1} -> survivor min
        assert m.add_edge(3, 2) == "merged"  # {2,3} -> survivor min
        assert m.add_edge(1, 2) == "acyclic"  # first SCC -> second SCC
        assert m.add_edge(3, 0) == "merged"  # closes cycle: everything merges
        G = nx.DiGraph(
            [(0, 1), (2, 3), (1, 0), (3, 2), (1, 2), (3, 0)]
        )
        _assert_matches(m, G)

        survivor = min(a0, a1, b0, b1)
        assert m.component_ids() == {survivor}
        assert m.members(survivor) == {0, 1, 2, 3}
        # Direct aliases and a two-step alias chain both resolve.
        aliases = {a0, a1, b0, b1} - {survivor}
        for dead in aliases:
            assert m.is_merged_alias(dead)
            assert m.resolve(dead) == survivor
            assert m.members(dead) == {0, 1, 2, 3}
        assert not m.is_merged_alias(survivor)
        with pytest.raises(NetworkXError):
            m.resolve(99)
        with pytest.raises(NetworkXError):
            m.component_of("missing")
        with pytest.raises(NetworkXError):
            m.members(99)

    def test_new_ids_never_reuse_dead(self):
        m = IncrementalSCC(nx.path_graph(2, create_using=nx.DiGraph))
        m.add_edge(1, 0)  # ids 0,1 -> survivor 0
        assert m.add_node("z") is True
        new_id = m.component_of("z")
        assert new_id >= 2
        assert m.resolve(1) == 0
        assert m.members(new_id) == {"z"}

    def test_condensation_snapshot_is_frozen(self):
        m = IncrementalSCC(nx.path_graph(2, create_using=nx.DiGraph))
        C = m.condensation()
        assert nx.is_frozen(C)
        c0, c1 = m.component_of(0), m.component_of(1)
        assert C.graph["mapping"] == {0: c0, 1: c1}
        assert C.nodes[c0]["members"] == {0}
        with pytest.raises(nx.NetworkXError):
            C.add_edge(c0, c1)
        # internal graph snapshot is frozen too
        assert nx.is_frozen(m.graph())
        # mutating the returned mapping dict does not corrupt state
        C.graph["mapping"].clear()
        assert m.component_of(0) == c0

    def test_external_mutation_is_isolated_and_detected(self):
        G = nx.path_graph(3, create_using=nx.DiGraph)
        m = IncrementalSCC(G)
        G.add_edge(2, 0)  # mutate source behind the maintainer's back
        # maintained state is untouched because it owns a copy
        assert m.number_of_components() == 3
        assert m.verify()
        with pytest.raises(NetworkXError):
            m.check_source()

    def test_check_source_clean(self):
        G = nx.path_graph(3, create_using=nx.DiGraph)
        m = IncrementalSCC(G)
        assert m.check_source()

    def test_undirected_rejected(self):
        with pytest.raises(NetworkXNotImplemented):
            IncrementalSCC(nx.Graph([(0, 1)]))

    def test_multidigraph_rejected(self):
        with pytest.raises(NetworkXNotImplemented):
            IncrementalSCC(nx.MultiDiGraph([(0, 1)]))

    def test_random_incremental_matches_full_scc(self):
        rng = random.Random(20261004)
        n = 25
        G = nx.DiGraph()
        G.add_nodes_from(range(n))
        m = IncrementalSCC(G)
        possible = [(u, v) for u in range(n) for v in range(n) if u != v]
        rng.shuffle(possible)
        # verify at every step, including isolated initial state
        _assert_matches(m, G)
        for step, (u, v) in enumerate(possible[:300]):
            status = m.add_edge(u, v)
            G.add_edge(u, v)
            if status == "merged":
                assert m.number_of_components() < len(G)
            _assert_matches(m, G)

    def test_random_batches_and_new_nodes(self):
        rng = random.Random(4242)
        G = nx.DiGraph()
        m = IncrementalSCC(G)
        num_nodes = 0
        for _ in range(30):
            if rng.random() < 0.3:
                new_nodes = [num_nodes, num_nodes + 1]
                num_nodes += 2
                assert m.add_nodes_from(new_nodes) == 2
                G.add_nodes_from(new_nodes)
            if num_nodes == 0:
                continue
            edges = [
                (rng.randrange(num_nodes), rng.randrange(num_nodes))
                for _ in range(rng.randrange(1, 6))
            ]
            G.add_edges_from(edges)
            m.add_edges_from(edges)
            _assert_matches(m, G)

    def test_merge_redirects_in_and_out_edges(self):
        # Two SCCs A={0,1}, B={2,3}, a predecessor p and successor s.
        G = nx.DiGraph(
            [(0, 1), (1, 0), (2, 3), (3, 2), (4, 0), (2, 5), (1, 2)]
        )
        m = IncrementalSCC(G)
        # 1 -> 2 already links A to B, so 3 -> 0 closes the cycle.
        assert m.add_edge(3, 0) == "merged"
        G.add_edge(3, 0)
        _assert_matches(m, G)
        merged_id = m.component_of(0)
        C = m.condensation()
        cp, cs = m.component_of(4), m.component_of(5)
        assert set(C.successors(merged_id)) == {cs}
        assert set(C.predecessors(merged_id)) == {cp}
        assert C.number_of_edges() == 2

    def test_local_acyclic_insert_is_constant_work(self):
        # Two giant SCCs linked once, plus a small gadget. A local
        # acyclic insertion inside the gadget must not touch the giants.
        n = 4000
        giant0 = list(range(0, n, 2))
        giant1 = list(range(1, n, 2))
        G = nx.DiGraph()
        G.add_edges_from(
            (giant0[i], giant0[(i + 1) % len(giant0)]) for i in range(len(giant0))
        )
        G.add_edges_from(
            (giant1[i], giant1[(i + 1) % len(giant1)]) for i in range(len(giant1))
        )
        G.add_edge(giant0[0], giant1[0])  # single condensation arc
        m = IncrementalSCC(G)
        assert m.number_of_components() == 2
        m.add_nodes_from(["a", "b", "c"])
        m.add_edge("a", "b")
        m.add_edge("b", "c")
        # local gadget chain a -> b -> c ; adding c -> a would merge it;
        # instead extend forward: no cycle, search stays near endpoints.
        m.add_node("d")
        visits = m.total_visit_count
        assert m.add_edge("c", "d") == "acyclic"
        assert m.last_visit_count == 1
        assert m.total_visit_count - visits == 1
        assert m.number_of_components() == 6
        m.verify()

    def test_merge_work_is_local_to_condensation(self):
        # A long chain of SCCs: merging two adjacent ones is constant work
        # in the number of condensation vertices visited, independent of
        # the sizes of the SCCs.
        k = 2000
        G = nx.DiGraph()
        for i in range(k - 1):
            G.add_edge((i, 0), (i + 1, 0))
        # give each "SCC" a second member so they are non-trivial-looking
        for i in range(k):
            G.add_edge((i, 0), (i, 1))
            G.add_edge((i, 1), (i, 0))
        m = IncrementalSCC(G)
        assert m.number_of_components() == k
        m.add_edge((k - 1, 0), (k - 2, 0))
        G.add_edge((k - 1, 0), (k - 2, 0))
        assert m.last_visit_count < 10
        assert m.number_of_components() == k - 1
        _assert_matches(m, G)

    def test_initial_condensation_matches_nx(self):
        G = nx.DiGraph(
            [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 3)]
        )
        m = IncrementalSCC(G)
        ref = nx.condensation(G)
        C = m.condensation()
        assert C.number_of_nodes() == ref.number_of_nodes()
        assert C.number_of_edges() == ref.number_of_edges()
        _assert_matches(m, G)
