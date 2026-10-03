"""Tests for the enumeration of all minimum (s, t)-cuts."""

import pytest

import networkx as nx
from networkx.algorithms.flow import (
    boykov_kolmogorov,
    dinitz,
    edmonds_karp,
    preflow_push,
    shortest_augmenting_path,
)

flow_funcs = [
    boykov_kolmogorov,
    dinitz,
    edmonds_karp,
    preflow_push,
    shortest_augmenting_path,
]


def brute_force_minimum_cuts(G, s, t, capacity="capacity"):
    """All minimum (s, t)-cut partitions by exhaustive enumeration."""
    nodes = list(G)
    others = [n for n in nodes if n != s and n != t]
    best_value = None
    best_partitions = set()
    for bits in range(2 ** len(others)):
        source = {s} | {n for i, n in enumerate(others) if bits >> i & 1}
        target = set(nodes) - source
        value = sum(
            data.get(capacity, float("inf"))
            for u, v, data in G.edges(data=True)
            if u in source and v in target
        )
        partition = (frozenset(source), frozenset(target))
        if best_value is None or value < best_value:
            best_value = value
            best_partitions = {partition}
        elif value == best_value:
            best_partitions.add(partition)
    return best_value, best_partitions


def validate_cuts(G, s, t, capacity="capacity", flow_func=None, **kwargs):
    """Check the enumerated cuts against brute force enumeration."""
    expected_value, expected_partitions = brute_force_minimum_cuts(
        G, s, t, capacity
    )
    cuts = list(nx.all_minimum_cuts(G, s, t, capacity, flow_func, **kwargs))
    partitions = set()
    for cut_value, (source, target), cutset in cuts:
        # The partition covers all nodes and separates s from t.
        assert source.isdisjoint(target)
        assert source | target == set(G)
        assert s in source and t in target
        # The cut value is the minimum cut value.
        assert cut_value == expected_value
        # The cutset is exactly the set of edges from source to target,
        # and its capacity sums to the cut value.
        expected_cutset = {
            (u, v) for u, v in G.edges(source) if v in target
        }
        assert cutset == expected_cutset
        assert (
            sum(G.edges[u, v].get(capacity, float("inf")) for u, v in cutset)
            == cut_value
        )
        partition = (frozenset(source), frozenset(target))
        # No partition is yielded twice.
        assert partition not in partitions
        partitions.add(partition)
    # No minimum cut is missing.
    assert partitions == expected_partitions


class TestAllMinimumCuts:
    def test_two_disjoint_paths(self):
        G = nx.DiGraph()
        G.add_edge("s", "a", capacity=1)
        G.add_edge("s", "b", capacity=1)
        G.add_edge("a", "t", capacity=1)
        G.add_edge("b", "t", capacity=1)
        cuts = list(nx.all_minimum_cuts(G, "s", "t"))
        assert len(cuts) == 4
        assert all(value == 2 for value, _, _ in cuts)
        sources = {frozenset(source) for _, (source, _), _ in cuts}
        assert sources == {
            frozenset({"s"}),
            frozenset({"s", "a"}),
            frozenset({"s", "b"}),
            frozenset({"s", "a", "b"}),
        }

    def test_unique_cut(self):
        # Example graph from the minimum_cut documentation.
        G = nx.DiGraph()
        G.add_edge("x", "a", capacity=3.0)
        G.add_edge("x", "b", capacity=1.0)
        G.add_edge("a", "c", capacity=3.0)
        G.add_edge("b", "c", capacity=5.0)
        G.add_edge("b", "d", capacity=4.0)
        G.add_edge("d", "e", capacity=2.0)
        G.add_edge("c", "y", capacity=2.0)
        G.add_edge("e", "y", capacity=3.0)
        cuts = list(nx.all_minimum_cuts(G, "x", "y"))
        assert len(cuts) == 1
        cut_value, (source, target), cutset = cuts[0]
        assert cut_value == 3.0
        assert source == {"x", "a", "c"}
        assert target == {"b", "d", "e", "y"}
        assert cutset == {("c", "y"), ("x", "b")}

    def test_zero_capacity_edges(self):
        G = nx.DiGraph()
        G.add_edge("s", "a", capacity=0)
        G.add_edge("a", "t", capacity=0)
        G.add_edge("s", "t", capacity=0)
        validate_cuts(G, "s", "t")
        cuts = list(nx.all_minimum_cuts(G, "s", "t"))
        assert all(value == 0 for value, _, _ in cuts)
        # "a" is free: it may lie on either side of the cut.
        assert len(cuts) == 2

    def test_disconnected_component(self):
        # Nodes unreachable from s and unable to reach t are free.
        G = nx.DiGraph()
        G.add_edge("s", "t", capacity=2)
        G.add_edge("u", "v", capacity=5)
        G.add_node("w")
        cuts = list(nx.all_minimum_cuts(G, "s", "t"))
        # "w" is completely free; "v" may move across the cut freely and
        # "u" may only be on the source side if "v" is (u -> v has spare
        # residual capacity), giving 2 * 3 = 6 cuts.
        assert len(cuts) == 6
        validate_cuts(G, "s", "t")

    def test_no_path(self):
        G = nx.DiGraph()
        G.add_edge("s", "a", capacity=3)
        G.add_edge("b", "t", capacity=3)
        validate_cuts(G, "s", "t")

    def test_exponential_number_of_cuts(self):
        k = 10
        G = nx.DiGraph()
        for i in range(k):
            G.add_edge("s", i, capacity=1)
            G.add_edge(i, "t", capacity=1)
        assert sum(1 for _ in nx.all_minimum_cuts(G, "s", "t")) == 2**k

    def test_cutoff(self):
        G = nx.DiGraph()
        for i in range(10):
            G.add_edge("s", i, capacity=1)
            G.add_edge(i, "t", capacity=1)
        assert sum(1 for _ in nx.all_minimum_cuts(G, "s", "t", cutoff=5)) == 5
        assert list(nx.all_minimum_cuts(G, "s", "t", cutoff=0)) == []
        # A cutoff larger than the number of cuts yields all of them.
        assert (
            sum(1 for _ in nx.all_minimum_cuts(G, "s", "t", cutoff=10**6)) == 2**10
        )

    def test_cutoff_negative(self):
        G = nx.DiGraph([("s", "t", {"capacity": 1})])
        with pytest.raises(nx.NetworkXError):
            nx.all_minimum_cuts(G, "s", "t", cutoff=-1)

    @pytest.mark.parametrize("flow_func", flow_funcs)
    def test_flow_funcs(self, flow_func):
        G = nx.DiGraph()
        G.add_edge("x", "a", capacity=3)
        G.add_edge("x", "b", capacity=1)
        G.add_edge("a", "c", capacity=3)
        G.add_edge("b", "c", capacity=5)
        G.add_edge("b", "d", capacity=4)
        G.add_edge("d", "e", capacity=2)
        G.add_edge("c", "y", capacity=2)
        G.add_edge("e", "y", capacity=3)
        validate_cuts(G, "x", "y", flow_func=flow_func)

    @pytest.mark.parametrize("seed", range(20))
    def test_random_digraphs_against_brute_force(self, seed):
        import random

        rng = random.Random(seed)
        n = rng.randint(2, 7)
        G = nx.DiGraph()
        G.add_nodes_from(range(n))
        for u in range(n):
            for v in range(n):
                if u != v and rng.random() < 0.4:
                    G.add_edge(u, v, capacity=rng.randint(0, 5))
        validate_cuts(G, 0, n - 1)

    def test_custom_capacity_attribute(self):
        G = nx.DiGraph()
        G.add_edge("s", "a", weight=1)
        G.add_edge("s", "b", weight=1)
        G.add_edge("a", "t", weight=1)
        G.add_edge("b", "t", weight=1)
        validate_cuts(G, "s", "t", capacity="weight")

    def test_undirected_graph(self):
        G = nx.Graph()
        G.add_edge("s", "a", capacity=1)
        G.add_edge("s", "b", capacity=1)
        G.add_edge("a", "t", capacity=1)
        G.add_edge("b", "t", capacity=1)
        validate_cuts(G, "s", "t")

    def test_multigraph_not_supported(self):
        G = nx.MultiDiGraph([("s", "t", {"capacity": 1})])
        with pytest.raises(nx.NetworkXError):
            nx.all_minimum_cuts(G, "s", "t")

    def test_source_equals_target(self):
        G = nx.DiGraph([("s", "t", {"capacity": 1})])
        with pytest.raises(nx.NetworkXError):
            nx.all_minimum_cuts(G, "s", "s")

    def test_node_not_in_graph(self):
        G = nx.DiGraph([("s", "t", {"capacity": 1})])
        with pytest.raises(nx.NetworkXError):
            nx.all_minimum_cuts(G, "s", "missing")
        with pytest.raises(nx.NetworkXError):
            nx.all_minimum_cuts(G, "missing", "t")

    def test_flow_func_not_callable(self):
        G = nx.DiGraph([("s", "t", {"capacity": 1})])
        with pytest.raises(nx.NetworkXError):
            nx.all_minimum_cuts(G, "s", "t", flow_func="not callable")

    def test_kwargs_without_flow_func(self):
        G = nx.DiGraph([("s", "t", {"capacity": 1})])
        with pytest.raises(nx.NetworkXError):
            nx.all_minimum_cuts(G, "s", "t", global_relabel_freq=1)

    def test_unbounded(self):
        G = nx.DiGraph([("s", "t")])  # no capacity attribute: infinite
        with pytest.raises(nx.NetworkXUnbounded):
            nx.all_minimum_cuts(G, "s", "t")

    def test_laziness(self):
        # Only one maximum flow is computed, and the enumeration does
        # not run to completion when the iterator is not exhausted.
        G = nx.DiGraph()
        for i in range(20):
            G.add_edge("s", i, capacity=1)
            G.add_edge(i, "t", capacity=1)
        flow_calls = 0

        def counting_flow_func(*args, **kwargs):
            nonlocal flow_calls
            flow_calls += 1
            return preflow_push(*args, **kwargs)

        it = nx.all_minimum_cuts(G, "s", "t", flow_func=counting_flow_func)
        assert flow_calls == 1
        first = next(it)
        assert first[0] == 20
        for _ in range(4):
            next(it)
        assert flow_calls == 1
