import itertools
import math
import random

import pytest

import networkx as nx
from networkx.algorithms.signed import MinimumFrustrationResult


def brute_force_cost(G, sign="sign", weight="weight"):
    """Minimum frustration by exhaustive enumeration of all 2-colorings."""
    nodes = list(G.nodes)
    best = math.inf
    for sides in itertools.product((0, 1), repeat=len(nodes)):
        side = dict(zip(nodes, sides))
        cost = 0.0
        edges = G.edges(data=True)
        for u, v, d in edges:
            s = d[sign]
            w = d.get(weight, 1.0)
            if (side[u] != side[v]) if s > 0 else (side[u] == side[v]):
                cost += w
        best = min(best, cost)
    return best


def check_result_consistent(G, res, sign="sign", weight="weight"):
    """The reported cost and frustrated edges must match the partition."""
    cost = 0.0
    expected_frustrated = []
    for e in G.edges(keys=True, data=True) if G.is_multigraph() else G.edges(data=True):
        u, v, d = e[0], e[1], e[-1]
        s = d[sign]
        w = d.get(weight, 1.0)
        violated = (res.partition[u] != res.partition[v]) if s > 0 else (
            res.partition[u] == res.partition[v]
        )
        if violated:
            cost += w
            expected_frustrated.append(e[:3] if G.is_multigraph() else e[:2])
    assert res.cost == pytest.approx(cost)
    assert sorted(map(str, res.frustrated_edges)) == sorted(
        map(str, expected_frustrated)
    )
    assert res.lower_bound <= res.cost + 1e-9


class TestMinimumFrustration:
    def test_frustrated_triangle(self):
        G = nx.Graph()
        G.add_edge(0, 1, sign=1, weight=2.0)
        G.add_edge(1, 2, sign=1, weight=2.0)
        G.add_edge(0, 2, sign=-1, weight=1.0)
        res = nx.minimum_frustration(G)
        assert res.cost == 1.0
        assert res.is_optimal
        assert res.lower_bound == pytest.approx(res.cost)
        check_result_consistent(G, res)

    def test_balanced_graph_zero_cost(self):
        # Two factions: positive inside, negative across -> balanced.
        G = nx.Graph()
        group = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}
        for u, v in itertools.combinations(group, 2):
            s = 1 if group[u] == group[v] else -1
            G.add_edge(u, v, sign=s, weight=1.0 + (u + v) % 3)
        res = nx.minimum_frustration(G)
        assert res.cost == 0.0
        assert res.is_optimal
        assert res.frustrated_edges == []
        # Partition must reproduce a balanced 2-coloring: recoverable.
        for u, v, d in G.edges(data=True):
            same = res.partition[u] == res.partition[v]
            assert same == (d["sign"] > 0)
        # The recovered grouping matches the planted one up to a global flip.
        flips = {res.partition[u] != res.partition[v] for u in group for v in group
                 if group[u] != group[v]}
        assert flips == {True}

    def test_empty_and_edgeless(self):
        G = nx.Graph()
        G.add_nodes_from(range(4))
        res = nx.minimum_frustration(G)
        assert res.cost == 0.0
        assert res.is_optimal
        assert set(res.partition) == set(G.nodes)
        assert nx.minimum_frustration(nx.Graph()).cost == 0.0

    def test_component_symmetry_broken(self):
        G = nx.Graph()
        G.add_edge(0, 1, sign=-1)
        G.add_edge(2, 3, sign=-1)  # second component
        res = nx.minimum_frustration(G)
        assert res.cost == 0.0
        # One vertex of each component is pinned to group 0.
        assert res.partition[0] == 0
        assert res.partition[2] == 0

    def test_negative_self_loop_always_frustrated(self):
        G = nx.Graph()
        G.add_edge(0, 0, sign=-1, weight=3.0)
        G.add_edge(0, 1, sign=1, weight=5.0)
        res = nx.minimum_frustration(G)
        assert res.cost == 3.0
        assert res.is_optimal
        assert res.frustrated_edges == [(0, 0)]
        check_result_consistent(G, res)

    def test_parallel_edges_charged_separately(self):
        G = nx.MultiGraph()
        G.add_edge(0, 1, sign=1, weight=2.0)
        G.add_edge(0, 1, sign=-1, weight=4.0)
        res = nx.minimum_frustration(G)
        # One of the two parallel edges must be violated; the cheaper wins.
        assert res.cost == 2.0
        assert len(res.frustrated_edges) == 1
        assert res.frustrated_edges[0] in [(0, 1, 0), (0, 1, 1)]
        check_result_consistent(G, res)

    @pytest.mark.parametrize("seed", range(10))
    def test_matches_brute_force_graph(self, seed):
        rng = random.Random(seed)
        G = nx.Graph()
        n = 7
        G.add_nodes_from(range(n))
        for u in range(n):
            for v in range(u, n):
                if rng.random() < 0.5:
                    G.add_edge(
                        u, v, sign=rng.choice([1, -1]), weight=rng.randint(0, 5)
                    )
        res = nx.minimum_frustration(G)
        assert res.is_optimal
        assert res.cost == pytest.approx(brute_force_cost(G))
        assert res.lower_bound == pytest.approx(res.cost)
        check_result_consistent(G, res)

    @pytest.mark.parametrize("seed", range(5))
    def test_matches_brute_force_multigraph(self, seed):
        rng = random.Random(100 + seed)
        G = nx.MultiGraph()
        n = 6
        G.add_nodes_from(range(n))
        for u in range(n):
            for v in range(u, n):
                for _ in range(rng.randint(0, 2)):
                    if rng.random() < 0.6:
                        G.add_edge(
                            u, v, sign=rng.choice([1, -1]), weight=rng.randint(0, 4)
                        )
        res = nx.minimum_frustration(G)
        assert res.is_optimal
        assert res.cost == pytest.approx(brute_force_cost(G))
        check_result_consistent(G, res)

    def test_node_budget_returns_feasible_with_bound(self):
        rng = random.Random(7)
        G = nx.Graph()
        n = 10
        G.add_nodes_from(range(n))
        for u, v in itertools.combinations(range(n), 2):
            if rng.random() < 0.6:
                G.add_edge(u, v, sign=rng.choice([1, -1]), weight=rng.randint(1, 4))
        res = nx.minimum_frustration(G, max_nodes=0)
        assert not res.is_optimal
        assert res.nodes_explored >= 1
        # Still a feasible solution: reported cost matches the partition.
        check_result_consistent(G, res)
        # The certified bound brackets the true optimum.
        assert res.lower_bound <= brute_force_cost(G) + 1e-9
        assert res.cost >= brute_force_cost(G) - 1e-9

    def test_unlimited_finds_optimum(self):
        rng = random.Random(11)
        G = nx.Graph()
        n = 9
        G.add_nodes_from(range(n))
        for u, v in itertools.combinations(range(n), 2):
            if rng.random() < 0.5:
                G.add_edge(u, v, sign=rng.choice([1, -1]), weight=rng.randint(1, 3))
        res = nx.minimum_frustration(G, max_nodes=None)
        assert res.is_optimal
        assert res.cost == pytest.approx(brute_force_cost(G))

    def test_sign_strings_accepted(self):
        G = nx.Graph()
        G.add_edge(0, 1, sign="+")
        G.add_edge(1, 2, sign="-")
        res = nx.minimum_frustration(G)
        assert res.cost == 0.0
        assert res.partition[0] == res.partition[1] != res.partition[2]

    @pytest.mark.parametrize("bad_sign", [0, 2, "x", "positive", None, True])
    def test_invalid_sign_raises(self, bad_sign):
        G = nx.Graph()
        G.add_edge(0, 1, sign=bad_sign)
        with pytest.raises(nx.NetworkXError, match="invalid sign"):
            nx.minimum_frustration(G)

    @pytest.mark.parametrize("bad_weight", [-1.0, math.inf, -math.inf, math.nan])
    def test_invalid_weight_raises(self, bad_weight):
        G = nx.Graph()
        G.add_edge(0, 1, sign=1, weight=bad_weight)
        with pytest.raises(nx.NetworkXError, match="invalid weight"):
            nx.minimum_frustration(G)

    def test_sign_error_reported_before_weight_error(self):
        G = nx.Graph()
        G.add_edge(0, 1, sign=1, weight=-5.0)
        G.add_edge(1, 2, sign="bogus", weight=1.0)
        with pytest.raises(nx.NetworkXError, match="invalid sign"):
            nx.minimum_frustration(G)

    def test_missing_sign_raises(self):
        G = nx.Graph()
        G.add_edge(0, 1)
        with pytest.raises(nx.NetworkXError, match="invalid sign"):
            nx.minimum_frustration(G)

    def test_directed_not_implemented(self):
        G = nx.DiGraph()
        G.add_edge(0, 1, sign=1)
        with pytest.raises(nx.NetworkXNotImplemented):
            nx.minimum_frustration(G)

    def test_invalid_max_nodes(self):
        G = nx.Graph()
        G.add_edge(0, 1, sign=1)
        with pytest.raises(nx.NetworkXError):
            nx.minimum_frustration(G, max_nodes=-1)
        with pytest.raises(nx.NetworkXError):
            nx.minimum_frustration(G, max_nodes=1.5)

    def test_result_type_and_fields(self):
        G = nx.Graph()
        G.add_edge(0, 1, sign=-1, weight=2.0)
        res = nx.minimum_frustration(G)
        assert isinstance(res, MinimumFrustrationResult)
        assert res._fields == (
            "partition",
            "frustrated_edges",
            "cost",
            "lower_bound",
            "is_optimal",
            "nodes_explored",
        )
