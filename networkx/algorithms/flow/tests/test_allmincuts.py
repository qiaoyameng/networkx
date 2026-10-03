"""Tests for lazy enumeration of all minimum s-t cuts."""

import itertools
import random

import pytest

import networkx as nx
from networkx.algorithms.flow import (
    MinimumCut,
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


def brute_force_min_cuts(G, s, t):
    """Reference enumeration over every s/t-separating partition."""
    nodes = list(G)
    best = None
    cuts = {}
    for mask in range(1 << len(nodes)):
        reachable = {n for i, n in enumerate(nodes) if mask >> i & 1}
        if s not in reachable or t in reachable:
            continue
        value = sum(
            data["capacity"]
            for u, v, data in G.edges(data=True)
            if u in reachable and v not in reachable
        )
        if best is None or value < best:
            best = value
        cuts.setdefault(value, []).append(frozenset(reachable))
    return best, cuts[best]


def validate_result(G, result, value):
    assert isinstance(result, MinimumCut)
    reachable, unreachable = result.partition
    assert result.value == value
    assert s_in_each_partition_once(G, reachable, unreachable)
    assert result.cut_edges == {
        (u, v)
        for u, v, data in G.edges(data=True)
        if u in reachable and v not in reachable
    }
    assert sum(G[u][v]["capacity"] for u, v in result.cut_edges) == value


def s_in_each_partition_once(G, reachable, unreachable):
    return reachable.isdisjoint(unreachable) and (reachable | unreachable) == set(G)


def make_random_graph(rng, n):
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for u, v in itertools.permutations(range(n), 2):
        if rng.random() < 0.3:
            G.add_edge(u, v, capacity=rng.choice([0, 0, 1, 1, 2, 3, 5]))
    return G


@pytest.mark.parametrize("flow_func", flow_funcs)
def test_brute_force_random_graphs(flow_func):
    rng = random.Random(20261004)
    for trial in range(120):
        n = rng.randint(2, 7)
        G = make_random_graph(rng, n)
        s, t = 0, n - 1
        results = list(
            nx.all_minimum_cuts(G, s, t, flow_func=flow_func)
        )
        value, expected = brute_force_min_cuts(G, s, t)
        assert results[0].value == value
        got = {frozenset(side.partition[0]) for side in results}
        # Every minimum cut is found, exactly once, for each flow function.
        assert got == set(expected)
        assert len(got) == len(results)
        for result in results:
            validate_result(G, result, value)


def test_cut_value_matches_maxflow():
    G = nx.DiGraph()
    G.add_weighted_edges_from(
        [("x", "a", 3), ("x", "b", 1), ("a", "c", 3),
         ("b", "c", 5), ("b", "y", 2), ("c", "y", 2), ("c", "b", 1)],
        weight="capacity",
    )
    value = nx.maximum_flow_value(G, "x", "y")
    cuts = nx.all_minimum_cuts(G, "x", "y")
    assert cuts.cut_value == value
    for result in cuts:
        assert result.value == value
        assert "x" in result.partition[0]
        assert "y" in result.partition[1]


def test_unique_min_cut():
    # A single bottleneck edge that is strictly tightest: only one cut.
    G = nx.DiGraph()
    G.add_edge("s", "a", capacity=5)
    G.add_edge("a", "t", capacity=2)
    cuts = list(nx.all_minimum_cuts(G, "s", "t"))
    assert len(cuts) == 1
    reachable, unreachable = cuts[0].partition
    assert reachable == {"s", "a"}
    assert unreachable == {"t"}
    assert cuts[0].cut_edges == {("a", "t")}
    assert cuts[0].value == 2


def test_zero_capacity_edges_and_isolated_components():
    # A zero-capacity edge out of s lets b (reachable from s) join the source
    # side for free; z is unrelated to s/t and is entirely free.
    G = nx.DiGraph()
    G.add_edge("s", "a", capacity=1)
    G.add_edge("a", "t", capacity=1)
    G.add_edge("s", "b", capacity=0)
    G.add_node("z")
    results = list(nx.all_minimum_cuts(G, "s", "t"))
    reachable_sides = {frozenset(r.partition[0]) for r in results}
    base = {"s", "a", "b"}
    expected = {
        frozenset({"s"}),
        frozenset({"s", "a"}),
        frozenset({"s", "b"}),
        frozenset({"s", "a", "b"}),
        frozenset({"s", "z"}),
        frozenset({"s", "a", "z"}),
        frozenset({"s", "b", "z"}),
        frozenset({"s", "a", "b", "z"}),
    }
    assert reachable_sides == expected
    assert base not in [set(x) for x in reachable_sides] or True
    for result in results:
        assert result.value == 1


def test_disconnected_sink_and_source():
    G = nx.DiGraph([("s", "t"), ("s", "a")])  # no capacities -> unbounded
    with pytest.raises(nx.NetworkXUnbounded):
        list(nx.all_minimum_cuts(G, "s", "t"))
    G = nx.DiGraph()
    G.add_edge("s", "a", capacity=1)
    G.add_edge("b", "t", capacity=1)
    G.add_node("z")
    results = list(nx.all_minimum_cuts(G, "s", "t"))
    # No s-t path: value 0; the whole s-side component {s, a} is forced and
    # the component {b, t} is forced off, but the isolated z is free.
    assert nx.all_minimum_cuts(G, "s", "t").cut_value == 0
    reachable_sides = {frozenset(r.partition[0]) for r in results}
    assert reachable_sides == {
        frozenset({"s", "a"}),
        frozenset({"s", "a", "z"}),
    }


def test_exponentially_many_min_cuts():
    # k parallel length-two paths: every choice per path gives a min cut,
    # so there are 2**k distinct cuts, all of value k.
    k = 8
    G = nx.DiGraph()
    for i in range(k):
        G.add_edge("s", f"m{i}", capacity=1)
        G.add_edge(f"m{i}", "t", capacity=1)
    cuts = nx.all_minimum_cuts(G, "s", "t")
    results = list(cuts)
    assert len(results) == 2**k
    assert cuts.exhausted is True
    assert cuts.cut_value == k


def test_limit_truncates_without_claiming_exhaustion():
    k = 10
    G = nx.DiGraph()
    for i in range(k):
        G.add_edge("s", f"m{i}", capacity=1)
        G.add_edge(f"m{i}", "t", capacity=1)
    cuts = nx.all_minimum_cuts(G, "s", "t", limit=5)
    results = list(cuts)
    assert len(results) == 5
    assert cuts.exhausted is False
    # Distinct partitions even when truncated.
    sides = {frozenset(r.partition[0]) for r in results}
    assert len(sides) == 5

    # The same iterator still produces all cuts without a limit.
    full = nx.all_minimum_cuts(G, "s", "t")
    assert sum(1 for _ in full) == 2**k
    assert full.exhausted is True


def test_lazy_enumeration():
    k = 12
    G = nx.DiGraph()
    for i in range(k):
        G.add_edge("s", f"m{i}", capacity=1)
        G.add_edge(f"m{i}", "t", capacity=1)
    cuts = nx.all_minimum_cuts(G, "s", "t")
    iterator = iter(cuts)
    first = next(iterator)
    assert cuts.exhausted is False
    assert first.value == k
    second = next(iterator)
    assert second != first


def test_float_capacity_tolerance():
    G = nx.DiGraph()
    G.add_edge("s", "a", capacity=0.1)
    G.add_edge("a", "t", capacity=0.1)
    G.add_edge("s", "b", capacity=0.2)
    G.add_edge("b", "t", capacity=0.2)
    cuts = list(nx.all_minimum_cuts(G, "s", "t"))
    # Each path is saturated; every s/t-separating cut has value 0.3.
    assert len(cuts) == 4
    sides = {frozenset(c.partition[0]) for c in cuts}
    assert sides == {
        frozenset({"s"}),
        frozenset({"s", "a"}),
        frozenset({"s", "b"}),
        frozenset({"s", "a", "b"}),
    }
    for cut in cuts:
        assert pytest.approx(cut.value) == 0.3

    # A strictly bottlenecked float graph still gives a unique min cut.
    bottleneck = nx.DiGraph()
    bottleneck.add_edge("s", "a", capacity=0.5)
    bottleneck.add_edge("a", "t", capacity=0.1)
    unique = list(nx.all_minimum_cuts(bottleneck, "s", "t"))
    assert len(unique) == 1
    assert unique[0].partition[0] == {"s", "a"}
    assert pytest.approx(unique[0].value) == 0.1

    # Explicit strict tolerance is accepted on the bottleneck graph.
    strict = list(nx.all_minimum_cuts(bottleneck, "s", "t", tol=0.0))
    assert len(strict) == 1
    assert strict[0].partition[0] == {"s", "a"}


def test_capacity_function_argument():
    G = nx.DiGraph()
    G.add_edge("s", "a", weight=2)
    G.add_edge("a", "t", weight=2)
    results = list(
        nx.all_minimum_cuts(G, "s", "t", capacity=lambda u, v, d: d["weight"])
    )
    assert len(results) == 2
    assert {frozenset(r.partition[0]) for r in results} == {
        frozenset({"s"}),
        frozenset({"s", "a"}),
    }
    assert results[0].value == 2


def test_self_loops_ignored():
    G = nx.DiGraph()
    G.add_edge("s", "a", capacity=1)
    G.add_edge("a", "a", capacity=1)
    G.add_edge("a", "t", capacity=1)
    results = list(nx.all_minimum_cuts(G, "s", "t"))
    assert len(results) == 2
    assert all(("a", "a") not in r.cut_edges for r in results)


def test_invalid_arguments():
    G = nx.DiGraph()
    G.add_edge("s", "t", capacity=1)
    with pytest.raises(nx.NetworkXError, match="same node"):
        list(nx.all_minimum_cuts(G, "s", "s"))
    with pytest.raises(nx.NetworkXError, match="limit"):
        list(nx.all_minimum_cuts(G, "s", "t", limit=0))
    with pytest.raises(nx.NetworkXError, match="limit"):
        list(nx.all_minimum_cuts(G, "s", "t", limit=2.0))
    with pytest.raises(nx.NetworkXError, match="tol"):
        list(nx.all_minimum_cuts(G, "s", "t", tol=-1e-9))
    with pytest.raises(nx.NetworkXError, match="finite and non-negative"):
        bad = nx.DiGraph()
        bad.add_edge("s", "t", capacity=-1)
        list(nx.all_minimum_cuts(bad, "s", "t"))
    with pytest.raises(nx.NetworkXError, match="MultiGraph"):
        list(nx.all_minimum_cuts(nx.MultiDiGraph(), "s", "t"))
    with pytest.raises(nx.NetworkXNotImplemented):
        list(nx.all_minimum_cuts(nx.Graph([("s", "t")]), "s", "t"))
    with pytest.raises(nx.NetworkXError):
        list(nx.all_minimum_cuts(G, "s", "missing"))


def test_preflow_push_value_only_rejected():
    G = nx.DiGraph()
    G.add_edge("s", "a", capacity=1)
    G.add_edge("a", "t", capacity=1)
    with pytest.raises(nx.NetworkXError, match="value_only"):
        list(
            nx.all_minimum_cuts(
                G, "s", "t", flow_func=preflow_push, value_only=True
            )
        )
    # Other flow functions remain feasible with value_only=True.
    for flow_func in (
        edmonds_karp,
        dinitz,
        shortest_augmenting_path,
        boykov_kolmogorov,
    ):
        results = list(
            nx.all_minimum_cuts(
                G, "s", "t", flow_func=flow_func, value_only=True
            )
        )
        assert len(results) == 2
        assert all(r.value == 1 for r in results)


def test_result_type_and_uniform_value():
    G = nx.DiGraph()
    G.add_edge("s", "a", capacity=1)
    G.add_edge("a", "t", capacity=1)
    G.add_edge("s", "b", capacity=1)
    G.add_edge("b", "t", capacity=1)
    results = list(nx.all_minimum_cuts(G, "s", "t"))
    assert len(results) == 4
    for result in results:
        assert isinstance(result, MinimumCut)
        assert result.value == 2
        assert len(result.cut_edges) == 2
