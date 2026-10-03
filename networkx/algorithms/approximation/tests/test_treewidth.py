import itertools
import random

import pytest

import networkx as nx
from networkx.algorithms.approximation import (
    maximum_weight_independent_set,
    treewidth_min_degree,
    treewidth_min_fill_in,
)
from networkx.algorithms.approximation.treewidth import (
    MinDegreeHeuristic,
    min_fill_in_heuristic,
)


def is_tree_decomp(graph, decomp):
    """Check if the given tree decomposition is valid."""
    for x in graph.nodes():
        appear_once = False
        for bag in decomp.nodes():
            if x in bag:
                appear_once = True
                break
        assert appear_once

    # Check if each connected pair of nodes are at least once together in a bag
    for x, y in graph.edges():
        appear_together = False
        for bag in decomp.nodes():
            if x in bag and y in bag:
                appear_together = True
                break
        assert appear_together

    # Check if the nodes associated with vertex v form a connected subset of T
    for v in graph.nodes():
        subset = []
        for bag in decomp.nodes():
            if v in bag:
                subset.append(bag)
        sub_graph = decomp.subgraph(subset)
        assert nx.is_connected(sub_graph)


class TestTreewidthMinDegree:
    """Unit tests for the min_degree function"""

    @classmethod
    def setup_class(cls):
        """Setup for different kinds of trees"""
        cls.complete = nx.Graph()
        cls.complete.add_edge(1, 2)
        cls.complete.add_edge(2, 3)
        cls.complete.add_edge(1, 3)

        cls.small_tree = nx.Graph()
        cls.small_tree.add_edge(1, 3)
        cls.small_tree.add_edge(4, 3)
        cls.small_tree.add_edge(2, 3)
        cls.small_tree.add_edge(3, 5)
        cls.small_tree.add_edge(5, 6)
        cls.small_tree.add_edge(5, 7)
        cls.small_tree.add_edge(6, 7)

        cls.deterministic_graph = nx.Graph()
        cls.deterministic_graph.add_edge(0, 1)  # deg(0) = 1

        cls.deterministic_graph.add_edge(1, 2)  # deg(1) = 2

        cls.deterministic_graph.add_edge(2, 3)
        cls.deterministic_graph.add_edge(2, 4)  # deg(2) = 3

        cls.deterministic_graph.add_edge(3, 4)
        cls.deterministic_graph.add_edge(3, 5)
        cls.deterministic_graph.add_edge(3, 6)  # deg(3) = 4

        cls.deterministic_graph.add_edge(4, 5)
        cls.deterministic_graph.add_edge(4, 6)
        cls.deterministic_graph.add_edge(4, 7)  # deg(4) = 5

        cls.deterministic_graph.add_edge(5, 6)
        cls.deterministic_graph.add_edge(5, 7)
        cls.deterministic_graph.add_edge(5, 8)
        cls.deterministic_graph.add_edge(5, 9)  # deg(5) = 6

        cls.deterministic_graph.add_edge(6, 7)
        cls.deterministic_graph.add_edge(6, 8)
        cls.deterministic_graph.add_edge(6, 9)  # deg(6) = 6

        cls.deterministic_graph.add_edge(7, 8)
        cls.deterministic_graph.add_edge(7, 9)  # deg(7) = 5

        cls.deterministic_graph.add_edge(8, 9)  # deg(8) = 4

    def test_petersen_graph(self):
        """Test Petersen graph tree decomposition result"""
        G = nx.petersen_graph()
        _, decomp = treewidth_min_degree(G)
        is_tree_decomp(G, decomp)

    def test_small_tree_treewidth(self):
        """Test small tree

        Test if the computed treewidth of the known self.small_tree is 2.
        As we know which value we can expect from our heuristic, values other
        than two are regressions
        """
        G = self.small_tree
        # the order of removal should be [1,2,4]3[5,6,7]
        # (with [] denoting any order of the containing nodes)
        # resulting in treewidth 2 for the heuristic
        treewidth, _ = treewidth_min_fill_in(G)
        assert treewidth == 2

    def test_heuristic_abort(self):
        """Test heuristic abort condition for fully connected graph"""
        graph = {}
        for u in self.complete:
            graph[u] = set()
            for v in self.complete[u]:
                if u != v:  # ignore self-loop
                    graph[u].add(v)

        deg_heuristic = MinDegreeHeuristic(graph)
        node = deg_heuristic.best_node(graph)
        if node is None:
            pass
        else:
            assert False

    def test_empty_graph(self):
        """Test empty graph"""
        G = nx.Graph()
        _, _ = treewidth_min_degree(G)

    def test_two_component_graph(self):
        G = nx.Graph()
        G.add_node(1)
        G.add_node(2)
        treewidth, _ = treewidth_min_degree(G)
        assert treewidth == 0

    def test_not_sortable_nodes(self):
        G = nx.Graph([(0, "a")])
        treewidth_min_degree(G)

    def test_heuristic_first_steps(self):
        """Test first steps of min_degree heuristic"""
        graph = {
            n: set(self.deterministic_graph[n]) - {n} for n in self.deterministic_graph
        }
        deg_heuristic = MinDegreeHeuristic(graph)
        elim_node = deg_heuristic.best_node(graph)
        steps = []

        while elim_node is not None:
            steps.append(elim_node)
            nbrs = graph[elim_node]

            for u, v in itertools.permutations(nbrs, 2):
                if v not in graph[u]:
                    graph[u].add(v)

            for u in graph:
                if elim_node in graph[u]:
                    graph[u].remove(elim_node)

            del graph[elim_node]
            elim_node = deg_heuristic.best_node(graph)

        # check only the first 5 elements for equality
        assert steps[:5] == [0, 1, 2, 3, 4]


class TestTreewidthMinFillIn:
    """Unit tests for the treewidth_min_fill_in function."""

    @classmethod
    def setup_class(cls):
        """Setup for different kinds of trees"""
        cls.complete = nx.Graph()
        cls.complete.add_edge(1, 2)
        cls.complete.add_edge(2, 3)
        cls.complete.add_edge(1, 3)

        cls.small_tree = nx.Graph()
        cls.small_tree.add_edge(1, 2)
        cls.small_tree.add_edge(2, 3)
        cls.small_tree.add_edge(3, 4)
        cls.small_tree.add_edge(1, 4)
        cls.small_tree.add_edge(2, 4)
        cls.small_tree.add_edge(4, 5)
        cls.small_tree.add_edge(5, 6)
        cls.small_tree.add_edge(5, 7)
        cls.small_tree.add_edge(6, 7)

        cls.deterministic_graph = nx.Graph()
        cls.deterministic_graph.add_edge(1, 2)
        cls.deterministic_graph.add_edge(1, 3)
        cls.deterministic_graph.add_edge(3, 4)
        cls.deterministic_graph.add_edge(2, 4)
        cls.deterministic_graph.add_edge(3, 5)
        cls.deterministic_graph.add_edge(4, 5)
        cls.deterministic_graph.add_edge(3, 6)
        cls.deterministic_graph.add_edge(5, 6)

    def test_petersen_graph(self):
        """Test Petersen graph tree decomposition result"""
        G = nx.petersen_graph()
        _, decomp = treewidth_min_fill_in(G)
        is_tree_decomp(G, decomp)

    def test_small_tree_treewidth(self):
        """Test if the computed treewidth of the known self.small_tree is 2"""
        G = self.small_tree
        # the order of removal should be [1,2,4]3[5,6,7]
        # (with [] denoting any order of the containing nodes)
        # resulting in treewidth 2 for the heuristic
        treewidth, _ = treewidth_min_fill_in(G)
        assert treewidth == 2

    def test_heuristic_abort(self):
        """Test if min_fill_in returns None for fully connected graph"""
        graph = {}
        for u in self.complete:
            graph[u] = set()
            for v in self.complete[u]:
                if u != v:  # ignore self-loop
                    graph[u].add(v)
        next_node = min_fill_in_heuristic(graph)
        if next_node is None:
            pass
        else:
            assert False

    def test_empty_graph(self):
        """Test empty graph"""
        G = nx.Graph()
        _, _ = treewidth_min_fill_in(G)

    def test_two_component_graph(self):
        G = nx.Graph()
        G.add_node(1)
        G.add_node(2)
        treewidth, _ = treewidth_min_fill_in(G)
        assert treewidth == 0

    def test_not_sortable_nodes(self):
        G = nx.Graph([(0, "a")])
        treewidth_min_fill_in(G)

    def test_heuristic_first_steps(self):
        """Test first steps of min_fill_in heuristic"""
        graph = {
            n: set(self.deterministic_graph[n]) - {n} for n in self.deterministic_graph
        }
        elim_node = min_fill_in_heuristic(graph)
        steps = []

        while elim_node is not None:
            steps.append(elim_node)
            nbrs = graph[elim_node]

            for u, v in itertools.permutations(nbrs, 2):
                if v not in graph[u]:
                    graph[u].add(v)

            for u in graph:
                if elim_node in graph[u]:
                    graph[u].remove(elim_node)

            del graph[elim_node]
            elim_node = min_fill_in_heuristic(graph)

        # check only the first 2 elements for equality
        assert steps[:2] == [6, 5]


def _brute_force_mwis(G, weight="weight"):
    """Maximum weight independent set by enumeration of all subsets."""
    best_weight, best_set = 0, set()
    for r in range(len(G) + 1):
        for combo in itertools.combinations(G, r):
            nodes = set(combo)
            if any(G.has_edge(u, v) for u, v in itertools.combinations(nodes, 2)):
                continue
            if any(G.has_edge(v, v) for v in nodes):
                continue
            w = sum(G.nodes[v].get(weight, 1) for v in nodes)
            if w > best_weight:
                best_weight, best_set = w, nodes
    return best_weight, best_set


class TestMaximumWeightIndependentSet:
    """Unit tests for maximum_weight_independent_set on tree decompositions."""

    def _check_solution(self, G, weight, result):
        w, nodes = result
        assert isinstance(nodes, set)
        assert nodes <= set(G)
        for u, v in itertools.combinations(nodes, 2):
            assert not G.has_edge(u, v)
        if weight:
            expected = sum(G.nodes[v].get(weight, 1) for v in nodes)
        else:
            expected = len(nodes)
        assert w == expected

    def test_path_graph_matches_brute_force(self):
        G = nx.path_graph(8)
        for v, w in zip(G, [3, -1, 0, 5, -2, 4, 1, -7]):
            G.nodes[v]["weight"] = w
        for decomp in (
            treewidth_min_degree(G)[1],
            treewidth_min_fill_in(G)[1],
        ):
            result = maximum_weight_independent_set(G, decomp)
            self._check_solution(G, "weight", result)
            assert result[0] == _brute_force_mwis(G)[0]

    def test_join_decomposition_string_nodes(self):
        # Two triangles sharing node "c"; the root bag joins both branches.
        G = nx.Graph()
        G.add_edges_from(
            [("a", "b"), ("b", "c"), ("c", "a"), ("c", "d"), ("d", "e"), ("e", "c")]
        )
        for v, w in zip("abcde", [2, -3, 4, 1, 5]):
            G.nodes[v]["weight"] = w
        decomp = nx.Graph()
        root = frozenset(["c"])
        decomp.add_edge(root, frozenset(["a", "b", "c"]))
        decomp.add_edge(root, frozenset(["c", "d", "e"]))
        result = maximum_weight_independent_set(G, decomp)
        self._check_solution(G, "weight", result)
        assert result[0] == _brute_force_mwis(G)[0]

    def test_random_graphs_match_brute_force(self):
        rng = random.Random(42)
        for trial in range(20):
            n = rng.randint(1, 9)
            G = nx.gnm_random_graph(n, rng.randint(0, n * (n - 1) // 2), seed=trial)
            for v in G:
                G.nodes[v]["weight"] = rng.randint(-3, 5)
            _, decomp = treewidth_min_degree(G)
            result = maximum_weight_independent_set(G, decomp)
            self._check_solution(G, "weight", result)
            assert result[0] == _brute_force_mwis(G)[0]

    def test_all_negative_weights_empty_set(self):
        G = nx.path_graph(4)
        for v in G:
            G.nodes[v]["weight"] = -2
        _, decomp = treewidth_min_degree(G)
        assert maximum_weight_independent_set(G, decomp) == (0, set())

    def test_unweighted(self):
        G = nx.path_graph(5)
        _, decomp = treewidth_min_degree(G)
        w, nodes = maximum_weight_independent_set(G, decomp, weight=None)
        assert w == 3
        assert len(nodes) == 3

    def test_empty_graph(self):
        assert maximum_weight_independent_set(nx.Graph(), nx.Graph()) == (0, set())

    def test_edge_not_covered_raises(self):
        G = nx.path_graph(3)
        decomp = nx.Graph()
        decomp.add_edge(frozenset([0, 1]), frozenset([0, 2]))
        with pytest.raises(nx.NetworkXError, match="not covered by any bag"):
            maximum_weight_independent_set(G, decomp)

    def test_node_not_in_any_bag_raises(self):
        G = nx.path_graph(3)
        decomp = nx.Graph()
        decomp.add_node(frozenset([0, 1, 2]))
        G.add_node("isolated")
        with pytest.raises(nx.NetworkXError, match="does not appear in any bag"):
            maximum_weight_independent_set(G, decomp)

    def test_disconnected_subtree_raises(self):
        G = nx.path_graph(3)
        decomp = nx.Graph()
        decomp.add_edge(frozenset([0, 1]), frozenset([0, 2]))
        decomp.add_edge(frozenset([0, 2]), frozenset([1, 2]))
        with pytest.raises(nx.NetworkXError, match="connected subtree"):
            maximum_weight_independent_set(G, decomp)

    def test_not_a_tree_raises(self):
        G = nx.path_graph(3)
        decomp = nx.Graph()
        decomp.add_edge(frozenset([0, 1]), frozenset([1, 2]))
        decomp.add_edge(frozenset([1, 2]), frozenset([0, 1, 2]))
        decomp.add_edge(frozenset([0, 1, 2]), frozenset([0, 1]))
        with pytest.raises(nx.NetworkXError, match="must be a tree"):
            maximum_weight_independent_set(G, decomp)

    def test_max_bag_size_exceeded_raises(self):
        G = nx.path_graph(3)
        _, decomp = treewidth_min_degree(G)
        with pytest.raises(nx.NetworkXError, match="max_bag_size"):
            maximum_weight_independent_set(G, decomp, max_bag_size=1)
        # A sufficient limit does not raise.
        maximum_weight_independent_set(G, decomp, max_bag_size=2)
