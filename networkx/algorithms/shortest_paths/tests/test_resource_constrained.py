"""Tests for the exact resource-constrained shortest path algorithm."""

import math
import random

import pytest

import networkx as nx
from networkx.algorithms.shortest_paths.resource_constrained import (
    resource_constrained_path,
)


def _bruteforce_front(G, source, target, budget):
    """Enumerate all node-simple feasible walks and return the set of
    nondominated (cost, resource) pairs (exhaustive reference)."""
    results = set()

    def walk(node, cost, res, visited):
        if node == target:
            results.add((cost, res))
        for neighbor, edges in G[node].items():
            if neighbor in visited:
                continue
            if G.is_multigraph():
                candidates = list(edges)
            else:
                candidates = [None]
            for key in candidates:
                data = edges if key is None else edges[key]
                edge_cost = data["cost"]
                edge_res = data["resource"]
                new_res = res + edge_res
                if new_res <= budget:
                    walk(neighbor, cost + edge_cost, new_res, visited | {neighbor})

    walk(source, 0, 0, {source})

    def nondominated(pairs):
        return {
            pair
            for pair in pairs
            if not any(
                other != pair
                and other[0] <= pair[0]
                and other[1] <= pair[1]
                and (other[0] < pair[0] or other[1] < pair[1])
                for other in pairs
            )
        }

    return nondominated(results)


def _path_achieves(G, path, want_cost, want_res):
    """Whether some choice of parallel edge keys realizes (want_cost,
    want_res) along the given node sequence."""
    states = {(0, 0)}
    for u, v in zip(path, path[1:]):
        edges = G[u][v].values() if G.is_multigraph() else [G[u][v]]
        states = {
            (cost + data["cost"], res + data["resource"])
            for cost, res in states
            for data in edges
        }
    return (want_cost, want_res) in states


@pytest.mark.parametrize("multigraph", [False, True])
@pytest.mark.parametrize("directed", [False, True])
def test_matches_exhaustive_enumeration(directed, multigraph, seed=0):
    rng = random.Random(123456)
    for trial in range(40):
        n = rng.randint(8, 13)
        if multigraph:
            G = nx.MultiDiGraph() if directed else nx.MultiGraph()
        else:
            G = nx.DiGraph() if directed else nx.Graph()
        G.add_nodes_from(range(n))
        for u in range(n):
            for v in range(n):
                if u == v or not rng.random() < 0.28:
                    continue
                if not directed and v < u:
                    continue
                n_edges = rng.randint(1, 3) if multigraph else 1
                for _ in range(n_edges):
                    # Include exact zeros to exercise zero/zero cycles.
                    edge_cost = rng.choice([0, 0, 1, 2, 3])
                    edge_res = rng.choice([0, 0, 1, 2, 3])
                    if multigraph:
                        G.add_edge(u, v, cost=edge_cost, resource=edge_res)
                    else:
                        G.add_edge(u, v, cost=edge_cost, resource=edge_res)
        source = 0
        target = n - 1
        for budget in range(7):
            expected = _bruteforce_front(G, source, target, budget)
            if not expected:
                with pytest.raises(nx.NetworkXNoPath):
                    resource_constrained_path(G, source, target, budget)
                continue
            expected_cost = min(cost for cost, _ in expected)
            got_cost, got_path = resource_constrained_path(
                G, source, target, budget
            )
            got_res = min(
                res for c, res in expected if c == got_cost
            )
            assert got_cost == expected_cost
            assert got_path[0] == source and got_path[-1] == target
            assert _path_achieves(G, got_path, got_cost, got_res)

            _, _, front = resource_constrained_path(
                G, source, target, budget, return_front=True
            )
            front_pairs = {(cost, res) for cost, res, _ in front}
            assert front_pairs == expected
            # Front entries are sorted and pairwise nondominated.
            assert [pair[:2] for pair in front] == sorted(
                pair[:2] for pair in front
            )
            for front_cost, res, path in front:
                assert res <= budget
                assert _path_achieves(G, path, front_cost, res)


def test_single_label_dijkstra_counterexample():
    # Naive single-label Dijkstra that keeps only the cheapest label per
    # node (pruning edges by resource budget) is wrong for this graph.
    G = nx.DiGraph()
    G.add_edge("s", "a", cost=1, resource=2)
    G.add_edge("s", "b", cost=2, resource=1)
    G.add_edge("a", "t", cost=1, resource=2)
    G.add_edge("b", "a", cost=1, resource=0)
    G.add_edge("b", "t", cost=5, resource=0)

    def naive_single_label_dijkstra(budget):
        dist = {"s": 0}
        res = {"s": 0}
        fringe = [(0, "s")]
        import heapq

        while fringe:
            d, u = heapq.heappop(fringe)
            if d != dist[u]:
                continue
            for v, data in G[u].items():
                new_res = res[u] + data["resource"]
                new_cost = d + data["cost"]
                if new_res > budget:
                    continue
                if v not in dist or new_cost < dist[v]:
                    dist[v] = new_cost
                    res[v] = new_res
                    heapq.heappush(fringe, (new_cost, v))
        return dist.get("t")

    budget = 3
    assert naive_single_label_dijkstra(budget) == 7  # s-b-t (wrong)
    cost, path = resource_constrained_path(G, "s", "t", budget)
    assert cost == 4  # s-b-a-t
    assert path == ["s", "b", "a", "t"]


def test_zero_cost_zero_resource_cycles_terminate():
    G = nx.DiGraph()
    G.add_edge("s", "a", cost=1, resource=0)
    G.add_edge("a", "s", cost=0, resource=0)  # zero/zero back edge
    G.add_edge("a", "a", cost=0, resource=0)  # zero/zero self loop
    G.add_edge("a", "b", cost=0, resource=0)
    G.add_edge("b", "a", cost=0, resource=0)  # zero/zero cycle
    G.add_edge("b", "t", cost=2, resource=1)
    # A tight cap proves labels do not multiply around zero/zero loops.
    cost, path = resource_constrained_path(
        G, "s", "t", budget=1, max_labels=10
    )
    assert cost == 3
    assert path == ["s", "a", "b", "t"]


def test_parallel_edges_deduplicated_and_compared():
    G = nx.MultiDiGraph()
    G.add_edge("s", "a", cost=2, resource=0)
    G.add_edge("s", "a", cost=2, resource=0)  # exact duplicate label
    G.add_edge("s", "a", cost=1, resource=2)  # alternative label
    G.add_edge("a", "t", cost=0, resource=0)
    cost, path = resource_constrained_path(G, "s", "t", budget=0)
    assert (cost, path) == (2, ["s", "a", "t"])
    _, _, front = resource_constrained_path(
        G, "s", "t", budget=2, return_front=True
    )
    assert {(c, r) for c, r, _ in front} == {(1, 2), (2, 0)}


def test_edge_keys_reconstruct_exact_walk():
    G = nx.MultiDiGraph()
    G.add_edge("s", "a", cost=2, resource=0)       # key 0, feasible
    G.add_edge("s", "a", cost=0, resource=2)       # key 1
    G.add_edge("a", "t", cost=1, resource=0)       # key 0
    G.add_edge("a", "t", cost=10, resource=0)      # key 1, costly
    cost, (nodes, keys) = resource_constrained_path(
        G, "s", "t", budget=0, return_edge_keys=True
    )
    assert cost == 3
    assert nodes == ["s", "a", "t"]
    assert keys == [0, 0]
    total_cost = total_res = 0
    for u, v, key in zip(nodes, nodes[1:], keys):
        total_cost += G[u][v][key]["cost"]
        total_res += G[u][v][key]["resource"]
    assert (total_cost, total_res) == (3, 0)
    _, _, front = resource_constrained_path(
        G, "s", "t", budget=2, return_front=True, return_edge_keys=True
    )
    for front_cost, front_res, (f_nodes, f_keys) in front:
        total_cost = total_res = 0
        for u, v, key in zip(f_nodes, f_nodes[1:], f_keys):
            total_cost += G[u][v][key]["cost"]
            total_res += G[u][v][key]["resource"]
        assert (total_cost, total_res) == (front_cost, front_res)


def test_source_is_target():
    G = nx.DiGraph()
    G.add_edge(0, 1, cost=1, resource=1)
    assert resource_constrained_path(G, 0, 0, budget=0) == (0, [0])
    _, _, front = resource_constrained_path(
        G, 0, 0, budget=0, return_front=True
    )
    assert front == [(0, 0, [0])]


def test_missing_and_negative_attributes():
    G = nx.DiGraph()
    G.add_edge(0, 1)
    G.add_edge(1, 2, cost=1, resource=1)
    with pytest.raises(ValueError, match="cost"):
        resource_constrained_path(G, 0, 2, budget=1)
    G2 = nx.DiGraph()
    G2.add_edge(0, 1, cost=-1, resource=1)
    with pytest.raises(ValueError, match="cost"):
        resource_constrained_path(G2, 0, 1, budget=1)
    G3 = nx.DiGraph()
    G3.add_edge(0, 1, cost=1, resource=math.nan)
    with pytest.raises(ValueError, match="resource"):
        resource_constrained_path(G3, 0, 1, budget=1)


@pytest.mark.parametrize("bad_budget", [-1, math.nan, math.inf, "5", None, -0.5])
def test_invalid_budget(bad_budget):
    G = nx.path_graph(3, create_using=nx.DiGraph)
    nx.set_edge_attributes(
        G, {(0, 1): {"cost": 1, "resource": 1}, (1, 2): {"cost": 1, "resource": 1}}
    )
    with pytest.raises((ValueError, TypeError)):
        resource_constrained_path(G, 0, 2, bad_budget)


def test_missing_nodes():
    G = nx.DiGraph()
    G.add_edge(0, 1, cost=1, resource=1)
    with pytest.raises(nx.NodeNotFound):
        resource_constrained_path(G, 5, 1, budget=1)
    with pytest.raises(nx.NodeNotFound):
        resource_constrained_path(G, 0, 5, budget=1)


def test_label_cap_reports_incomplete():
    # Each layer doubles the number of distinct (cost, resource) labels.
    G = nx.DiGraph()
    layers = 6
    for layer in range(layers):
        u = layer
        v = layer + 1
        G.add_edge(u, v, cost=1, resource=0)
        G.add_edge(u, v, cost=0, resource=1)
    with pytest.raises(nx.ExceededMaxIterations):
        resource_constrained_path(G, 0, layers, budget=layers, max_labels=5)
    # With enough labels the exact answer is available.
    cost, _ = resource_constrained_path(
        G, 0, layers, budget=layers, max_labels=10_000
    )
    assert cost == 0


def test_incomparable_node_types():
    target = ("target",)
    G = nx.DiGraph()
    G.add_edge(0, "a", cost=1, resource=1)
    G.add_edge("a", target, cost=1, resource=0)
    G.add_edge(0, target, cost=5, resource=0)
    cost, path = resource_constrained_path(G, 0, target, budget=1)
    assert cost == 2 and path == [0, "a", target]


def test_callable_and_hidden_edges():
    G = nx.DiGraph()
    G.add_edge(0, 1, c=1, r=2, allowed=True)
    G.add_edge(0, 2, c=3, r=0, allowed=False)
    G.add_edge(2, 1, c=0, r=0, allowed=True)
    cost = lambda u, v, d: d["c"]
    resource = lambda u, v, d: d["r"] if d["allowed"] else None
    cost_value, path = resource_constrained_path(
        G, 0, 1, budget=2, cost=cost, resource=resource
    )
    assert cost_value == 1 and path == [0, 1]


def test_docstring_example():
    G = nx.DiGraph()
    G.add_edge(0, 1, cost=5, resource=0)
    G.add_edge(0, 2, cost=1, resource=1)
    G.add_edge(2, 1, cost=1, resource=1)
    assert resource_constrained_path(G, 0, 1, budget=2) == (2, [0, 2, 1])
