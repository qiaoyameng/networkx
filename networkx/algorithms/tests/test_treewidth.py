"""Tests for exact algorithms on bounded-treewidth graphs."""

import itertools
import random

import pytest

import networkx as nx
from networkx.algorithms.treewidth import _independent_subsets


def _brute_force_mwis(G, weight):
    """Reference maximum-weight independent set by exhaustive search."""
    if weight is None:
        weights = {v: 1 for v in G}
    else:
        weights = dict(G.nodes(data=weight, default=1))
    vertices = list(G)
    best_value = 0
    best_set = frozenset()
    for bits in range(1 << len(vertices)):
        chosen = frozenset(
            v for i, v in enumerate(vertices) if bits & (1 << i)
        )
        if any(G.has_edge(u, v) for u, v in itertools.combinations(chosen, 2)):
            continue
        value = sum((weights[v] for v in chosen), 0)
        if value > best_value:
            best_value = value
            best_set = chosen
    return best_value, best_set


def _path_decomposition(G):
    bags = [frozenset((u, v)) for u, v in G.edges]
    T = nx.Graph()
    T.add_nodes_from(bags)
    T.add_edges_from(zip(bags, bags[1:]))
    if len(bags) == 1:
        T.add_node(bags[0])
    return T


def _check_solution(G, T, weight="weight"):
    value, nodes = nx.max_weight_independent_set(G, T, weight=weight)
    expected_value, expected_set = _brute_force_mwis(G, weight)
    assert value == expected_value
    assert nodes == set(nodes)  # result is a plain set
    assert all(
        not G.has_edge(u, v) for u, v in itertools.combinations(nodes, 2)
    )
    weights = (
        {v: 1 for v in G}
        if weight is None
        else dict(G.nodes(data=weight, default=1))
    )
    assert sum((weights[v] for v in nodes), 0) == value
    assert set(nodes) <= set(G)
    return value, nodes


class TestPathDecomposition:
    def test_unweighted_path(self):
        G = nx.path_graph(6)
        value, nodes = _check_solution(G, _path_decomposition(G))
        assert value == 3
        assert len(nodes) == 3

    def test_single_edge(self):
        G = nx.path_graph(2)
        value, nodes = _check_solution(G, _path_decomposition(G))
        assert value == 1
        assert nodes == {0} or nodes == {1}

    def test_single_vertex(self):
        G = nx.path_graph(1)
        T = nx.Graph()
        T.add_node(frozenset({0}))
        value, nodes = _check_solution(G, T)
        assert value == 1 and nodes == {0}

    def test_empty_graph(self):
        value, nodes = nx.max_weight_independent_set(
            nx.empty_graph(), nx.Graph()
        )
        assert value == 0 and nodes == set()

    def test_signed_weights(self):
        G = nx.path_graph(5)
        nx.set_node_attributes(
            G, {0: -1, 1: 2.5, 2: 0, 3: -4, 4: 3}, "weight"
        )
        value, nodes = _check_solution(G, _path_decomposition(G))
        assert value == 5.5
        assert nodes == {1, 4}

    def test_all_nonpositive_weights_gives_empty_set(self):
        G = nx.path_graph(4)
        nx.set_node_attributes(G, {0: -1, 1: 0, 2: -2, 3: -0.5}, "weight")
        value, nodes = nx.max_weight_independent_set(
            G, _path_decomposition(G)
        )
        assert value == 0
        assert nodes == set()

    def test_zero_weights_empty_set(self):
        G = nx.path_graph(3)
        for v in G:
            G.nodes[v]["weight"] = 0
        value, nodes = nx.max_weight_independent_set(
            G, _path_decomposition(G)
        )
        assert value == 0
        assert nodes == set()

    def test_weight_none_uses_unit_weights(self):
        G = nx.path_graph(4)
        value, nodes = nx.max_weight_independent_set(
            G, _path_decomposition(G), weight=None
        )
        assert value == 2
        assert len(nodes) == 2

    def test_custom_weight_attribute(self):
        G = nx.path_graph(3)
        nx.set_node_attributes(G, {0: 5, 1: 1, 2: 5}, "cost")
        value, nodes = nx.max_weight_independent_set(
            G, _path_decomposition(G), weight="cost"
        )
        assert value == 10
        assert nodes == {0, 2}

    def test_non_integer_labels(self):
        G = nx.path_graph(["a", "b", "c", "d"])
        value, nodes = _check_solution(G, _path_decomposition(G))
        assert value == 2
        assert len(nodes) == 2 and all(isinstance(v, str) for v in nodes)

    def test_tuple_labels(self):
        G = nx.path_graph([(0, 0), (1, 1), (2, 2)])
        value, nodes = _check_solution(G, _path_decomposition(G))
        assert value == 2
        assert nodes == {(0, 0), (2, 2)}


class TestJoinDecomposition:
    @staticmethod
    def _triangle_tail():
        # triangle a-b-c with a tail c-d; the decomposition branches at
        # the central bag, exercising a genuine join.
        G = nx.Graph()
        G.add_edges_from([("a", "b"), ("b", "c"), ("a", "c"), ("c", "d")])
        central = frozenset(("a", "b", "c"))
        T = nx.Graph()
        T.add_edge(central, frozenset(("a", "c")))
        T.add_edge(central, frozenset(("c", "d")))
        return G, T, central

    def test_join_unweighted(self):
        G, T, _ = self._triangle_tail()
        value, nodes = _check_solution(G, T)
        assert value == 2
        assert nodes == {"a", "d"} or nodes == {"b", "d"}

    def test_join_weights(self):
        G, T, _ = self._triangle_tail()
        nx.set_node_attributes(
            G, {"a": 10, "b": 1, "c": 4, "d": 3}, "weight"
        )
        value, nodes = _check_solution(G, T)
        assert value == 13
        assert nodes == {"a", "d"}

    def test_join_shared_separator_choice(self):
        # Two branches share separator {a, c}; choosing a in one branch
        # must force a in the other and forbid c everywhere.
        G = nx.Graph()
        G.add_edges_from([("a", "c"), ("c", "d"), ("a", "e")])
        central = frozenset(("a", "c"))
        T = nx.Graph()
        T.add_edge(central, frozenset(("c", "d")))
        T.add_edge(central, frozenset(("a", "e")))
        nx.set_node_attributes(G, {"a": 5, "c": 1, "d": 5, "e": 5}, "weight")
        value, nodes = _check_solution(G, T)
        assert value == 10
        assert nodes == {"a", "d"} or nodes == {"e", "d"}

    def test_three_branch_join(self):
        # Star centered at c with three leaves; decomposition branches
        # three ways from the central bag {c}.
        G = nx.star_graph(3)
        central = frozenset({0})
        T = nx.Graph()
        for leaf in (1, 2, 3):
            T.add_edge(central, frozenset({0, leaf}))
        value, nodes = _check_solution(G, T)
        assert value == 3
        assert nodes == {1, 2, 3}


class TestExhaustiveComparison:
    @pytest.mark.parametrize("seed", range(40))
    def test_random_graphs_heuristic_decomposition(self, seed):
        G = nx.gnp_random_graph(8, 0.35, seed=seed)
        G = nx.relabel_nodes(G, {i: f"n{i}" for i in G})
        weights = {
            v: w
            for v, w in zip(
                G, [-2, -1.5, 0, 1, 2.5, 3, -0.5, 4][: len(G)]
            )
        }
        nx.set_node_attributes(G, weights, "weight")
        _, T = nx.approximation.treewidth_min_fill_in(G)
        _check_solution(G, T)

    def test_disconnected_graph(self):
        G = nx.Graph()
        G.add_edges_from([(0, 1), (2, 3)])
        # connect component decompositions via a shared (empty-overlap)
        # edge; the resulting tree is a valid decomposition
        T = nx.Graph()
        T.add_edge(frozenset({0, 1}), frozenset({2, 3}))
        value, nodes = _check_solution(G, T)
        assert value == 2

    @pytest.mark.parametrize("seed", range(30))
    def test_random_branching_decompositions(self, seed):
        # Closed neighborhoods of a random tree give a valid (often
        # branching, hence with real join nodes) decomposition of any
        # subgraph of that tree.
        random.seed(seed)
        n = random.randint(4, 10)
        tree = nx.random_labeled_tree(n, seed=seed)
        G = nx.Graph()
        G.add_nodes_from(tree.nodes)
        for u, v in tree.edges:
            if random.random() < 0.8:
                G.add_edge(u, v)
        G = nx.relabel_nodes(G, {i: f"v{i}" for i in G})
        tree = nx.relabel_nodes(tree, {i: f"v{i}" for i in tree})

        T = nx.Graph()
        bags = {}
        for v in G:
            bag = frozenset({v, *tree.neighbors(v)})
            bags[v] = bag
            T.add_node(bag)
        if T.number_of_nodes() != G.number_of_nodes():
            pytest.skip("duplicate neighborhood bags")
        for u, v in tree.edges:
            T.add_edge(bags[u], bags[v])

        weights = {v: random.choice([-2, -1, 0, 1, 2, 3, 0.5]) for v in G}
        nx.set_node_attributes(G, weights, "weight")
        assert nx.is_valid_tree_decomposition(G, T)
        _check_solution(G, T)

    def test_complete_graph_single_bag(self):
        G = nx.complete_graph(5)
        T = nx.Graph()
        T.add_node(frozenset(G.nodes))
        value, nodes = _check_solution(G, T)
        assert value == 1
        assert len(nodes) == 1


class TestIndependentSubsets:
    def test_independent_subsets(self):
        G = nx.path_graph(3)
        result = {frozenset(s) for s in _independent_subsets(G, [0, 1, 2])}
        expected = {
            frozenset(),
            frozenset({0}),
            frozenset({1}),
            frozenset({2}),
            frozenset({0, 2}),
        }
        assert result == expected

    def test_independent_subsets_on_clique(self):
        G = nx.complete_graph(3)
        result = _independent_subsets(G, [0, 1, 2])
        assert frozenset() in result
        assert all(len(s) <= 1 for s in result)


class TestValidation:
    def setup_method(self):
        self.G = nx.path_graph(4)

    def test_valid_decomposition(self):
        T = _path_decomposition(self.G)
        assert nx.is_valid_tree_decomposition(self.G, T)

    def test_missing_vertex(self):
        T = nx.Graph()
        T.add_edge(frozenset({0, 1}), frozenset({1}))
        assert not nx.is_valid_tree_decomposition(self.G, T)

    def test_missing_edge(self):
        G = nx.path_graph(3)
        T = nx.Graph()
        T.add_edge(frozenset({0, 1}), frozenset({1}))
        T.add_node(frozenset({2}))
        assert not nx.is_valid_tree_decomposition(G, T)

    def test_running_intersection_violated(self):
        G = nx.Graph()
        G.add_edges_from([("a", "b"), ("b", "c"), ("c", "d")])
        T = nx.Graph()
        T.add_edge(frozenset({"a", "b"}), frozenset({"b", "c"}))
        T.add_edge(frozenset({"b", "c"}), frozenset({"c", "d"}))
        T.add_edge(frozenset({"c", "d"}), frozenset({"a"}))
        assert not nx.is_valid_tree_decomposition(G, T)

    def test_decomposition_not_a_tree(self):
        T = nx.Graph()
        T.add_edge(frozenset({0, 1}), frozenset({1, 2}))
        T.add_node(frozenset({2, 3}))
        assert not nx.is_valid_tree_decomposition(self.G, T)

    def test_bag_with_foreign_vertex(self):
        T = nx.Graph()
        T.add_edge(frozenset({0, 1}), frozenset({1, 9}))
        assert not nx.is_valid_tree_decomposition(nx.path_graph(2), T)

    def test_empty_decomposition_nonempty_graph(self):
        assert not nx.is_valid_tree_decomposition(
            nx.path_graph(1), nx.Graph()
        )

    def test_invalid_decomposition_raises_value_error(self):
        T = nx.Graph()
        T.add_edge(frozenset({0, 1}), frozenset({1}))
        with pytest.raises(ValueError, match="not a valid tree decomposition"):
            nx.max_weight_independent_set(self.G, T)


class TestBagSizeBound:
    def test_within_bound(self):
        G = nx.path_graph(3)
        T = _path_decomposition(G)
        value, nodes = nx.max_weight_independent_set(
            G, T, max_bag_size=2
        )
        assert value == 2

    def test_bound_exceeded(self):
        G = nx.complete_graph(4)
        T = nx.Graph()
        T.add_node(frozenset(G.nodes))
        with pytest.raises(nx.NetworkXTreewidthBoundExceeded):
            nx.max_weight_independent_set(G, T, max_bag_size=3)

    def test_bound_checked_before_validation(self):
        # a foreign vertex makes the decomposition invalid, but the
        # oversized bag must still be rejected via the bound exception
        G = nx.complete_graph(4)
        T = nx.Graph()
        T.add_node(frozenset({0, 1, 2, 9}))
        with pytest.raises(nx.NetworkXTreewidthBoundExceeded):
            nx.max_weight_independent_set(G, T, max_bag_size=2)


def test_alias():
    assert (
        nx.maximum_weight_independent_set is nx.max_weight_independent_set
    )


def test_directed_graph_raises():
    G = nx.path_graph(3).to_directed()
    T = _path_decomposition(nx.path_graph(3))
    with pytest.raises(nx.NetworkXNotImplemented):
        nx.max_weight_independent_set(G, T)
    with pytest.raises(nx.NetworkXNotImplemented):
        nx.is_valid_tree_decomposition(G, T)


def test_decomposition_reused_not_required_optimal():
    # a valid but non-minimal decomposition still gives exact results
    G = nx.path_graph(3)
    T = nx.Graph()
    T.add_node(frozenset({0, 1, 2}))
    assert nx.is_valid_tree_decomposition(G, T)
    value, nodes = nx.max_weight_independent_set(G, T)
    assert value == 2
    assert nodes == {0, 2}
