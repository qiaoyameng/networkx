"""Tests for exact minimum frustration of signed graphs."""

import itertools

import pytest

import networkx as nx


def brute_force_frustration(G):
    """Minimum frustration by enumerating all bipartitions."""
    nodes = list(G)
    multi = G.is_multigraph()

    def cost(assignment):
        total = 0
        iterator = G.edges(keys=True, data=True) if multi else G.edges(data=True)
        for edge in iterator:
            if multi:
                u, v, _, data = edge
            else:
                u, v, data = edge
            weight = data.get("weight", 1)
            if u == v:
                if not data["sign"]:
                    total += weight
            else:
                different = assignment[u] != assignment[v]
                violated = different if data["sign"] else not different
                if violated:
                    total += weight
        return total

    best = None
    for bits in itertools.product((0, 1), repeat=len(nodes)):
        value = cost(dict(zip(nodes, bits)))
        best = value if best is None else min(best, value)
    return best


def verify_partition(G, result):
    multi = G.is_multigraph()
    total = 0
    for eid in result.frustrated_edges:
        data = G.get_edge_data(*eid)
        total += data.get("weight", 1)
        u, v = eid[0], eid[1]
        if u != v:
            different = result.groups[u] != result.groups[v]
            if data["sign"]:
                assert different
            else:
                assert not different
    assert total == result.cost


def test_empty_graph():
    result = nx.minimum_frustration(nx.Graph())
    assert result.groups == {}
    assert result.frustrated_edges == set()
    assert result.cost == 0
    assert result.lower_bound == 0
    assert result.proven_optimal


@pytest.mark.parametrize("seed", range(40))
def test_matches_brute_force(seed):
    G = nx.gnp_random_graph(6, 0.5, seed=seed)
    for u, v in G.edges():
        G[u][v]["sign"] = ((seed + u + v) % 3) != 0
        G[u][v]["weight"] = (seed * 7 + u + v) % 4
    result = nx.minimum_frustration(G)
    assert result.proven_optimal
    assert result.cost == result.lower_bound == brute_force_frustration(G)
    verify_partition(G, result)


def test_float_weights_match_brute_force():
    G = nx.Graph()
    G.add_edge(0, 1, sign=True, weight=0.5)
    G.add_edge(1, 2, sign=False, weight=2.25)
    G.add_edge(0, 2, sign=True, weight=1.75)
    result = nx.minimum_frustration(G)
    assert result.cost == result.lower_bound == brute_force_frustration(G)
    assert result.cost == pytest.approx(0.5)


def test_weighted_triangle():
    G = nx.Graph()
    G.add_edge(0, 1, sign=True, weight=3)
    G.add_edge(1, 2, sign=True, weight=4)
    G.add_edge(0, 2, sign=False, weight=10)
    result = nx.minimum_frustration(G)
    assert result.cost == 3
    assert result.frustrated_edges == {(0, 1)}


def test_balanced_graph_has_zero_cost():
    true_groups = {"a": 0, "b": 0, "c": 1, "d": 1}
    G = nx.Graph()
    G.add_edges_from(
        [("a", "b"), ("c", "d"), ("a", "c"), ("b", "d"), ("a", "d"), ("b", "c")]
    )
    for u, v in G.edges():
        G[u][v]["sign"] = true_groups[u] == true_groups[v]
        G[u][v]["weight"] = abs(hash((u, v))) % 7 + 1
    result = nx.minimum_frustration(G)
    assert result.cost == 0
    assert result.lower_bound == 0
    assert result.frustrated_edges == set()
    assert result.proven_optimal
    # Re-verify independently that every edge relation is satisfied.
    for u, v, data in G.edges(data=True):
        same = result.groups[u] == result.groups[v]
        assert same == data["sign"]


def test_all_positive_tree_is_balanced():
    G = nx.path_graph(5)
    for _, _, data in G.edges(data=True):
        data["sign"] = True
        data["weight"] = 2
    result = nx.minimum_frustration(G)
    assert result.cost == 0
    assert result.frustrated_edges == set()
    assert set(result.groups.values()) == {0}


def test_parallel_positive_and_negative_edges():
    G = nx.MultiGraph()
    G.add_edge(0, 1, key="pos", sign=True, weight=3)
    G.add_edge(0, 1, key="neg", sign=False, weight=2)
    result = nx.minimum_frustration(G)
    assert result.cost == 2
    assert result.frustrated_edges == {(0, 1, "neg")}
    verify_partition(G, result)


@pytest.mark.parametrize("seed", range(20))
def test_multigraph_matches_brute_force(seed):
    G = nx.MultiGraph()
    for u, v in itertools.combinations(range(5), 2):
        for key in range((seed + u + v) % 3):
            G.add_edge(u, v, key=key, sign=((seed + key) % 2) == 0, weight=(key + 1))
    result = nx.minimum_frustration(G)
    assert result.proven_optimal
    assert result.cost == result.lower_bound == brute_force_frustration(G)
    verify_partition(G, result)


def test_self_loops():
    G = nx.Graph()
    G.add_edge(0, 1, sign=True, weight=5)
    G.add_edge(0, 0, sign=False, weight=4)
    G.add_edge(1, 1, sign=True, weight=9)
    result = nx.minimum_frustration(G)
    assert result.cost == 4
    assert result.frustrated_edges == {(0, 0)}
    assert result.lower_bound == 4


def test_parallel_self_loops_multigraph():
    G = nx.MultiGraph()
    G.add_edge(0, 0, key="a", sign=False, weight=2)
    G.add_edge(0, 0, key="b", sign=False, weight=3)
    G.add_edge(0, 0, key="c", sign=True, weight=7)
    result = nx.minimum_frustration(G)
    assert result.cost == 5
    assert result.frustrated_edges == {(0, 0, "a"), (0, 0, "b")}


def test_disconnected_components_independent():
    G = nx.Graph()
    G.add_edge(0, 1, sign=True, weight=5)
    G.add_edge(2, 3, sign=False, weight=7)
    result = nx.minimum_frustration(G)
    assert result.cost == 0
    assert result.groups[0] == result.groups[1]
    assert result.groups[2] != result.groups[3]
    # Each component has one node pinned to 0.
    assert 0 in (result.groups[0],) and result.groups[0] == 0
    assert result.groups[2] == 0


def test_custom_attribute_names():
    G = nx.Graph()
    G.add_edge(0, 1, relation=True, cost=2.5)
    G.add_edge(1, 2, relation=False, cost=1.5)
    G.add_edge(0, 2, relation=True, cost=3.0)
    result = nx.minimum_frustration(G, sign_attr="relation", weight="cost")
    assert result.cost == 1.5
    unit = nx.minimum_frustration(G, sign_attr="relation", weight=None)
    assert unit.cost == 1


def test_zero_weights():
    G = nx.Graph()
    G.add_edge(0, 1, sign=False, weight=0)
    G.add_edge(0, 2, sign=False, weight=0)
    G.add_edge(1, 2, sign=False, weight=0)
    result = nx.minimum_frustration(G)
    assert result.cost == 0
    assert result.proven_optimal


def test_node_budget_reports_incumbent_bound_and_flag():
    G = nx.complete_graph(12)
    for u, v in G.edges():
        G[u][v]["sign"] = ((u + v) % 4) != 0
        G[u][v]["weight"] = (u * 3 + v) % 9 + 1
    optimum = nx.minimum_frustration(G)
    bounded = nx.minimum_frustration(G, node_budget=10)
    assert bounded.proven_optimal is False
    assert bounded.nodes_explored <= 10
    assert bounded.node_budget == 10
    assert bounded.lower_bound <= optimum.cost <= bounded.cost
    verify_partition(G, bounded)
    # Unpacking gives the feasible incumbent directly.
    groups, frustrated, cost = bounded
    assert cost == bounded.cost and frustrated is bounded.frustrated_edges


def test_budget_large_enough_proves_optimum():
    G = nx.cycle_graph(8)
    for u, v in G.edges():
        G[u][v]["sign"] = (u % 2) == 0
        G[u][v]["weight"] = 1
    result = nx.minimum_frustration(G, node_budget=10_000)
    assert result.proven_optimal
    assert result.lower_bound == result.cost


def test_result_fields():
    G = nx.Graph([(0, 1, {"sign": True})])
    result = nx.minimum_frustration(G)
    groups, frustrated, cost = result
    assert groups == result.groups
    assert frustrated == result.frustrated_edges
    assert cost == result.cost == 0
    assert isinstance(frustrated, set)


def test_missing_sign_raises():
    G = nx.Graph()
    G.add_edge(0, 1)
    with pytest.raises(ValueError, match="sign"):
        nx.minimum_frustration(G)


def test_invalid_sign_raises():
    for bad_sign in (1, 0, "positive", None):
        G = nx.Graph()
        G.add_edge(0, 1, sign=bad_sign)
        with pytest.raises(ValueError, match="invalid sign"):
            nx.minimum_frustration(G)


def test_negative_weight_raises():
    G = nx.Graph()
    G.add_edge(0, 1, sign=True, weight=-0.1)
    with pytest.raises(ValueError, match="non-negative"):
        nx.minimum_frustration(G)


def test_non_finite_weight_raises():
    for value in (float("inf"), float("-inf"), float("nan")):
        G = nx.Graph()
        G.add_edge(0, 1, sign=True, weight=value)
        with pytest.raises(ValueError, match="finite"):
            nx.minimum_frustration(G)


def test_invalid_node_budget_raises():
    G = nx.Graph([(0, 1, {"sign": True})])
    for budget in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="node_budget"):
            nx.minimum_frustration(G, node_budget=budget)


def test_directed_graph_raises():
    G = nx.DiGraph()
    G.add_edge(0, 1, sign=True)
    with pytest.raises(nx.NetworkXNotImplemented):
        nx.minimum_frustration(G)
