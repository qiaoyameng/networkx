from itertools import chain, combinations, product

import pytest

import networkx as nx

tree_all_pairs_lca = nx.tree_all_pairs_lowest_common_ancestor
all_pairs_lca = nx.all_pairs_lowest_common_ancestor


def get_pair(dictionary, n1, n2):
    if (n1, n2) in dictionary:
        return dictionary[n1, n2]
    else:
        return dictionary[n2, n1]


class TestTreeLCA:
    @classmethod
    def setup_class(cls):
        cls.DG = nx.DiGraph()
        edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]
        cls.DG.add_edges_from(edges)
        cls.ans = dict(tree_all_pairs_lca(cls.DG, 0))
        gold = {(n, n): n for n in cls.DG}
        gold.update({(0, i): 0 for i in range(1, 7)})
        gold.update(
            {
                (1, 2): 0,
                (1, 3): 1,
                (1, 4): 1,
                (1, 5): 0,
                (1, 6): 0,
                (2, 3): 0,
                (2, 4): 0,
                (2, 5): 2,
                (2, 6): 2,
                (3, 4): 1,
                (3, 5): 0,
                (3, 6): 0,
                (4, 5): 0,
                (4, 6): 0,
                (5, 6): 2,
            }
        )

        cls.gold = gold

    @staticmethod
    def assert_has_same_pairs(d1, d2):
        for a, b in ((min(pair), max(pair)) for pair in chain(d1, d2)):
            assert get_pair(d1, a, b) == get_pair(d2, a, b)

    def test_tree_all_pairs_lca_default_root(self):
        assert dict(tree_all_pairs_lca(self.DG)) == self.ans

    def test_tree_all_pairs_lca_return_subset(self):
        test_pairs = [(0, 1), (0, 1), (1, 0)]
        ans = dict(tree_all_pairs_lca(self.DG, 0, test_pairs))
        assert (0, 1) in ans and (1, 0) in ans
        assert len(ans) == 2

    def test_tree_all_pairs_lca(self):
        all_pairs = chain(combinations(self.DG, 2), ((node, node) for node in self.DG))

        ans = dict(tree_all_pairs_lca(self.DG, 0, all_pairs))
        self.assert_has_same_pairs(ans, self.ans)

    def test_tree_all_pairs_gold_example(self):
        ans = dict(tree_all_pairs_lca(self.DG))
        self.assert_has_same_pairs(self.gold, ans)

    def test_tree_all_pairs_lca_invalid_input(self):
        empty_digraph = tree_all_pairs_lca(nx.DiGraph())
        pytest.raises(nx.NetworkXPointlessConcept, list, empty_digraph)

        bad_pairs_digraph = tree_all_pairs_lca(self.DG, pairs=[(-1, -2)])
        pytest.raises(nx.NodeNotFound, list, bad_pairs_digraph)

    def test_tree_all_pairs_lca_subtrees(self):
        ans = dict(tree_all_pairs_lca(self.DG, 1))
        gold = {
            pair: lca
            for (pair, lca) in self.gold.items()
            if all(n in (1, 3, 4) for n in pair)
        }
        self.assert_has_same_pairs(gold, ans)

    def test_tree_all_pairs_lca_disconnected_nodes(self):
        G = nx.DiGraph()
        G.add_node(1)
        assert {(1, 1): 1} == dict(tree_all_pairs_lca(G))

        G.add_node(0)
        assert {(1, 1): 1} == dict(tree_all_pairs_lca(G, 1))
        assert {(0, 0): 0} == dict(tree_all_pairs_lca(G, 0))

        pytest.raises(nx.NetworkXError, list, tree_all_pairs_lca(G))

    def test_tree_all_pairs_lca_error_if_input_not_tree(self):
        # Cycle
        G = nx.DiGraph([(1, 2), (2, 1)])
        pytest.raises(nx.NetworkXError, list, tree_all_pairs_lca(G))
        # DAG
        G = nx.DiGraph([(0, 2), (1, 2)])
        pytest.raises(nx.NetworkXError, list, tree_all_pairs_lca(G))

    def test_tree_all_pairs_lca_generator(self):
        pairs = iter([(0, 1), (0, 1), (1, 0)])
        some_pairs = dict(tree_all_pairs_lca(self.DG, 0, pairs))
        assert (0, 1) in some_pairs and (1, 0) in some_pairs
        assert len(some_pairs) == 2

    def test_tree_all_pairs_lca_nonexisting_pairs_exception(self):
        lca = tree_all_pairs_lca(self.DG, 0, [(-1, -1)])
        pytest.raises(nx.NodeNotFound, list, lca)
        # check if node is None
        lca = tree_all_pairs_lca(self.DG, None, [(-1, -1)])
        pytest.raises(nx.NodeNotFound, list, lca)

    def test_tree_all_pairs_lca_routine_bails_on_DAGs(self):
        G = nx.DiGraph([(3, 4), (5, 4)])
        pytest.raises(nx.NetworkXError, list, tree_all_pairs_lca(G))

    def test_tree_all_pairs_lca_not_implemented(self):
        NNI = nx.NetworkXNotImplemented
        G = nx.Graph([(0, 1)])
        with pytest.raises(NNI):
            next(tree_all_pairs_lca(G))
        with pytest.raises(NNI):
            next(all_pairs_lca(G))
        pytest.raises(NNI, nx.lowest_common_ancestor, G, 0, 1)
        G = nx.MultiGraph([(0, 1)])
        with pytest.raises(NNI):
            next(tree_all_pairs_lca(G))
        with pytest.raises(NNI):
            next(all_pairs_lca(G))
        pytest.raises(NNI, nx.lowest_common_ancestor, G, 0, 1)

    def test_tree_all_pairs_lca_trees_without_LCAs(self):
        G = nx.DiGraph()
        G.add_node(3)
        ans = list(tree_all_pairs_lca(G))
        assert ans == [((3, 3), 3)]


class TestMultiTreeLCA(TestTreeLCA):
    @classmethod
    def setup_class(cls):
        cls.DG = nx.MultiDiGraph()
        edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]
        cls.DG.add_edges_from(edges)
        cls.ans = dict(tree_all_pairs_lca(cls.DG, 0))
        # add multiedges
        cls.DG.add_edges_from(edges)

        gold = {(n, n): n for n in cls.DG}
        gold.update({(0, i): 0 for i in range(1, 7)})
        gold.update(
            {
                (1, 2): 0,
                (1, 3): 1,
                (1, 4): 1,
                (1, 5): 0,
                (1, 6): 0,
                (2, 3): 0,
                (2, 4): 0,
                (2, 5): 2,
                (2, 6): 2,
                (3, 4): 1,
                (3, 5): 0,
                (3, 6): 0,
                (4, 5): 0,
                (4, 6): 0,
                (5, 6): 2,
            }
        )

        cls.gold = gold


class TestDAGLCA:
    @classmethod
    def setup_class(cls):
        cls.DG = nx.DiGraph()
        nx.add_path(cls.DG, (0, 1, 2, 3))
        nx.add_path(cls.DG, (0, 4, 3))
        nx.add_path(cls.DG, (0, 5, 6, 8, 3))
        nx.add_path(cls.DG, (5, 7, 8))
        cls.DG.add_edge(6, 2)
        cls.DG.add_edge(7, 2)

        cls.root_distance = nx.shortest_path_length(cls.DG, source=0)

        cls.gold = {
            (1, 1): 1,
            (1, 2): 1,
            (1, 3): 1,
            (1, 4): 0,
            (1, 5): 0,
            (1, 6): 0,
            (1, 7): 0,
            (1, 8): 0,
            (2, 2): 2,
            (2, 3): 2,
            (2, 4): 0,
            (2, 5): 5,
            (2, 6): 6,
            (2, 7): 7,
            (2, 8): 7,
            (3, 3): 3,
            (3, 4): 4,
            (3, 5): 5,
            (3, 6): 6,
            (3, 7): 7,
            (3, 8): 8,
            (4, 4): 4,
            (4, 5): 0,
            (4, 6): 0,
            (4, 7): 0,
            (4, 8): 0,
            (5, 5): 5,
            (5, 6): 5,
            (5, 7): 5,
            (5, 8): 5,
            (6, 6): 6,
            (6, 7): 5,
            (6, 8): 6,
            (7, 7): 7,
            (7, 8): 7,
            (8, 8): 8,
        }
        cls.gold.update(((0, n), 0) for n in cls.DG)

    def assert_lca_dicts_same(self, d1, d2, G=None):
        """Checks if d1 and d2 contain the same pairs and
        have a node at the same distance from root for each.
        If G is None use self.DG."""
        if G is None:
            G = self.DG
            root_distance = self.root_distance
        else:
            roots = [n for n, deg in G.in_degree if deg == 0]
            assert len(roots) == 1
            root_distance = nx.shortest_path_length(G, source=roots[0])

        for a, b in ((min(pair), max(pair)) for pair in chain(d1, d2)):
            assert (
                root_distance[get_pair(d1, a, b)] == root_distance[get_pair(d2, a, b)]
            )

    def test_all_pairs_lca_gold_example(self):
        self.assert_lca_dicts_same(dict(all_pairs_lca(self.DG)), self.gold)

    def test_all_pairs_lca_all_pairs_given(self):
        all_pairs = list(product(self.DG.nodes(), self.DG.nodes()))
        ans = all_pairs_lca(self.DG, pairs=all_pairs)
        self.assert_lca_dicts_same(dict(ans), self.gold)

    def test_all_pairs_lca_generator(self):
        all_pairs = product(self.DG.nodes(), self.DG.nodes())
        ans = all_pairs_lca(self.DG, pairs=all_pairs)
        self.assert_lca_dicts_same(dict(ans), self.gold)

    def test_all_pairs_lca_input_graph_with_two_roots(self):
        G = self.DG.copy()
        G.add_edge(9, 10)
        G.add_edge(9, 4)
        gold = self.gold.copy()
        gold[9, 9] = 9
        gold[9, 10] = 9
        gold[9, 4] = 9
        gold[9, 3] = 9
        gold[10, 4] = 9
        gold[10, 3] = 9
        gold[10, 10] = 10

        testing = dict(all_pairs_lca(G))

        G.add_edge(-1, 9)
        G.add_edge(-1, 0)
        self.assert_lca_dicts_same(testing, gold, G)

    def test_all_pairs_lca_nonexisting_pairs_exception(self):
        pytest.raises(nx.NodeNotFound, all_pairs_lca, self.DG, [(-1, -1)])

    def test_all_pairs_lca_pairs_without_lca(self):
        G = self.DG.copy()
        G.add_node(-1)
        gen = all_pairs_lca(G, [(-1, -1), (-1, 0)])
        assert dict(gen) == {(-1, -1): -1}

    def test_all_pairs_lca_null_graph(self):
        pytest.raises(nx.NetworkXPointlessConcept, all_pairs_lca, nx.DiGraph())

    def test_all_pairs_lca_non_dags(self):
        pytest.raises(nx.NetworkXError, all_pairs_lca, nx.DiGraph([(3, 4), (4, 3)]))

    def test_all_pairs_lca_nonempty_graph_without_lca(self):
        G = nx.DiGraph()
        G.add_node(3)
        ans = list(all_pairs_lca(G))
        assert ans == [((3, 3), 3)]

    def test_all_pairs_lca_bug_gh4942(self):
        G = nx.DiGraph([(0, 2), (1, 2), (2, 3)])
        ans = list(all_pairs_lca(G))
        assert len(ans) == 9

    def test_all_pairs_lca_default_kwarg(self):
        G = nx.DiGraph([(0, 1), (2, 1)])
        sentinel = object()
        assert nx.lowest_common_ancestor(G, 0, 2, default=sentinel) is sentinel

    def test_all_pairs_lca_identity(self):
        G = nx.DiGraph()
        G.add_node(3)
        assert nx.lowest_common_ancestor(G, 3, 3) == 3

    def test_all_pairs_lca_issue_4574(self):
        G = nx.DiGraph()
        G.add_nodes_from(range(17))
        G.add_edges_from(
            [
                (2, 0),
                (1, 2),
                (3, 2),
                (5, 2),
                (8, 2),
                (11, 2),
                (4, 5),
                (6, 5),
                (7, 8),
                (10, 8),
                (13, 11),
                (14, 11),
                (15, 11),
                (9, 10),
                (12, 13),
                (16, 15),
            ]
        )

        assert nx.lowest_common_ancestor(G, 7, 9) is None

    def test_all_pairs_lca_one_pair_gh4942(self):
        G = nx.DiGraph()
        # Note: order edge addition is critical to the test
        G.add_edge(0, 1)
        G.add_edge(2, 0)
        G.add_edge(2, 3)
        G.add_edge(4, 0)
        G.add_edge(5, 2)

        assert nx.lowest_common_ancestor(G, 1, 3) == 2


class TestMultiDiGraph_DAGLCA(TestDAGLCA):
    @classmethod
    def setup_class(cls):
        cls.DG = nx.MultiDiGraph()
        nx.add_path(cls.DG, (0, 1, 2, 3))
        # add multiedges
        nx.add_path(cls.DG, (0, 1, 2, 3))
        nx.add_path(cls.DG, (0, 4, 3))
        nx.add_path(cls.DG, (0, 5, 6, 8, 3))
        nx.add_path(cls.DG, (5, 7, 8))
        cls.DG.add_edge(6, 2)
        cls.DG.add_edge(7, 2)

        cls.root_distance = nx.shortest_path_length(cls.DG, source=0)

        cls.gold = {
            (1, 1): 1,
            (1, 2): 1,
            (1, 3): 1,
            (1, 4): 0,
            (1, 5): 0,
            (1, 6): 0,
            (1, 7): 0,
            (1, 8): 0,
            (2, 2): 2,
            (2, 3): 2,
            (2, 4): 0,
            (2, 5): 5,
            (2, 6): 6,
            (2, 7): 7,
            (2, 8): 7,
            (3, 3): 3,
            (3, 4): 4,
            (3, 5): 5,
            (3, 6): 6,
            (3, 7): 7,
            (3, 8): 8,
            (4, 4): 4,
            (4, 5): 0,
            (4, 6): 0,
            (4, 7): 0,
            (4, 8): 0,
            (5, 5): 5,
            (5, 6): 5,
            (5, 7): 5,
            (5, 8): 5,
            (6, 6): 6,
            (6, 7): 5,
            (6, 8): 6,
            (7, 7): 7,
            (7, 8): 7,
            (8, 8): 8,
        }
        cls.gold.update(((0, n), 0) for n in cls.DG)


def test_all_pairs_lca_self_ancestors():
    """Self-ancestors should always be the node itself, i.e. lca of (0, 0) is 0.
    See gh-4458."""
    # DAG for test - note order of node/edge addition is relevant
    G = nx.DiGraph()
    G.add_nodes_from(range(5))
    G.add_edges_from([(1, 0), (2, 0), (3, 2), (4, 1), (4, 3)])

    ap_lca = nx.all_pairs_lowest_common_ancestor
    assert all(u == v == a for (u, v), a in ap_lca(G) if u == v)
    MG = nx.MultiDiGraph(G)
    assert all(u == v == a for (u, v), a in ap_lca(MG) if u == v)
    MG.add_edges_from([(1, 0), (2, 0)])
    assert all(u == v == a for (u, v), a in ap_lca(MG) if u == v)


def test_lca_on_null_graph():
    G = nx.null_graph(create_using=nx.DiGraph)
    with pytest.raises(
        nx.NetworkXPointlessConcept, match="LCA meaningless on null graphs"
    ):
        nx.lowest_common_ancestor(G, 0, 0)


def test_lca_on_cycle_graph():
    G = nx.cycle_graph(6, create_using=nx.DiGraph)
    with pytest.raises(
        nx.NetworkXError, match="LCA only defined on directed acyclic graphs"
    ):
        nx.lowest_common_ancestor(G, 0, 3)


def test_lca_multiple_valid_solutions():
    G = nx.DiGraph()
    G.add_nodes_from(range(4))
    G.add_edges_from([(2, 0), (3, 0), (2, 1), (3, 1)])
    assert nx.lowest_common_ancestor(G, 0, 1) in {2, 3}


def test_lca_dont_rely_on_single_successor():
    # Nodes 0 and 1 have nodes 2 and 3 as immediate ancestors,
    # and node 2 also has node 3 as an immediate ancestor.
    G = nx.DiGraph()
    G.add_nodes_from(range(4))
    G.add_edges_from([(2, 0), (2, 1), (3, 1), (3, 0), (3, 2)])
    assert nx.lowest_common_ancestor(G, 0, 1) == 2


class _Unorderable:
    """Hashable node that intentionally cannot be compared or sorted."""

    def __init__(self, name):
        self.name = name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, _Unorderable) and self.name == other.name

    def __lt__(self, other):
        raise TypeError("nodes are not orderable")

    __le__ = __gt__ = __ge__ = __lt__

    def __repr__(self):
        return f"_Unorderable({self.name!r})"


class TestAllPairsLCASets:
    """Complete sets of lowest common ancestors on general DAGs."""

    @staticmethod
    def diamond_diamond():
        # r is a higher common ancestor; a and b are two incomparable,
        # mutually unreachable lowest common ancestors of x and y.
        G = nx.DiGraph(
            [("r", "a"), ("r", "b"), ("a", "x"), ("b", "x"),
             ("a", "y"), ("b", "y")]
        )
        return G

    def test_acceptance_two_incomparable_lcas_plus_higher_ancestor(self):
        G = self.diamond_diamond()
        result = dict(nx.all_pairs_lowest_common_ancestors(G, [("x", "y")]))
        assert result == {("x", "y"): {"a", "b"}}
        index = nx.DAGLCAIndex(G)
        assert index.lowest_common_ancestors("x", "y") == {"a", "b"}
        # The higher common ancestor must not survive.
        assert "r" not in index.lowest_common_ancestors("x", "y")

    def test_self_pair_contains_node_itself(self):
        G = self.diamond_diamond()
        index = nx.DAGLCAIndex(G)
        for node in G:
            assert index.lowest_common_ancestors(node, node) == {node}
        result = dict(nx.all_pairs_lowest_common_ancestors(G, [("x", "x")]))
        assert result == {("x", "x"): {"x"}}

    def test_no_common_ancestor_returns_empty_set(self):
        G = nx.DiGraph([(1, 2), (3, 4)])
        index = nx.DAGLCAIndex(G)
        assert index.lowest_common_ancestors(2, 4) == set()
        assert dict(index([(2, 4), (2, 2)])) == {
            (2, 4): set(),
            (2, 2): {2},
        }
        result = dict(nx.all_pairs_lowest_common_ancestors(G, [(2, 4)]))
        assert result == {(2, 4): set()}
        # An isolated node paired with an unrelated node has no LCA,
        # but its self-pair still contains itself.
        G.add_node(9)
        index.rebuild()
        assert index.lowest_common_ancestors(9, 2) == set()
        assert index.lowest_common_ancestors(9, 9) == {9}

    def test_default_pairs_and_duplicates(self):
        G = nx.DiGraph([(0, 1), (1, 2)])
        result = dict(nx.all_pairs_lowest_common_ancestors(G))
        assert len(result) == 6
        assert result[(0, 2)] == {0}
        index = nx.DAGLCAIndex(G)
        subset = dict(index([(0, 2), (0, 2)]))
        assert subset == {(0, 2): {0}}

    def test_set_agrees_with_single_lca_on_unique_dag(self):
        # On the existing test DAG the set-valued answer must be a singleton
        # containing whatever lowest_common_ancestor returns.
        G = nx.DiGraph()
        nx.add_path(G, (0, 1, 2, 3))
        nx.add_path(G, (0, 4, 3))
        nx.add_path(G, (0, 5, 6, 8, 3))
        nx.add_path(G, (5, 7, 8))
        G.add_edge(6, 2)
        G.add_edge(7, 2)
        index = nx.DAGLCAIndex(G)
        for u, v in product(G, repeat=2):
            lcas = index.lowest_common_ancestors(u, v)
            single = nx.lowest_common_ancestor(G, u, v)
            if single is None:
                assert lcas == set()
            else:
                # The single-LCA routine returns one arbitrary member of
                # the full set of lowest common ancestors.
                assert single in lcas
                # None of the other lowest ancestors is reachable from it.
                for other in lcas - {single}:
                    assert other not in nx.descendants(G, single)

    def test_multiple_valid_solutions_returns_both(self):
        G = nx.DiGraph(
            [(2, 0), (3, 0), (2, 1), (3, 1)]
        )
        index = nx.DAGLCAIndex(G)
        assert index.lowest_common_ancestors(0, 1) == {2, 3}

    def test_nodes_need_not_be_orderable(self):
        r, a, b, x, y = (_Unorderable(c) for c in "rabxy")
        G = nx.DiGraph(
            [(r, a), (r, b), (a, x), (b, x), (a, y), (b, y)]
        )
        index = nx.DAGLCAIndex(G)
        assert index.lowest_common_ancestors(x, y) == {a, b}
        result = dict(nx.all_pairs_lowest_common_ancestors(G, [(x, y)]))
        assert result == {(x, y): {a, b}}

    def test_multigraph(self):
        G = nx.MultiDiGraph(
            [("r", "a"), ("r", "b"), ("a", "x"), ("b", "x"),
             ("a", "y"), ("b", "y")]
        )
        G.add_edge("r", "a")
        index = nx.DAGLCAIndex(G)
        assert index.lowest_common_ancestors("x", "y") == {"a", "b"}

    def test_invalid_nodes(self):
        G = self.diamond_diamond()
        index = nx.DAGLCAIndex(G)
        with pytest.raises(nx.NodeNotFound):
            index.lowest_common_ancestors("x", "nope")
        with pytest.raises(nx.NodeNotFound):
            list(index([("x", "nope")]))
        with pytest.raises(nx.NodeNotFound):
            list(nx.all_pairs_lowest_common_ancestors(G, [("x", "nope")]))

    def test_null_cyclic_and_undirected(self):
        with pytest.raises(nx.NetworkXPointlessConcept):
            nx.DAGLCAIndex(nx.DiGraph())
        with pytest.raises(nx.NetworkXError):
            nx.DAGLCAIndex(nx.DiGraph([(1, 2), (2, 1)]))
        with pytest.raises(nx.NetworkXNotImplemented):
            nx.DAGLCAIndex(nx.Graph([(1, 2)]))
        with pytest.raises(nx.NetworkXNotImplemented):
            next(nx.all_pairs_lowest_common_ancestors(nx.Graph([(1, 2)])))
        # Rebuilding on a graph that became cyclic is an error too.
        G = nx.DiGraph([(1, 2)])
        index = nx.DAGLCAIndex(G)
        G.add_edge(2, 1)
        with pytest.raises(nx.NetworkXError):
            index.rebuild()

    def test_index_is_reused_across_queries(self):
        G = self.diamond_diamond()
        index = nx.DAGLCAIndex(G)
        before = index.ancestor_sets
        assert index.lowest_common_ancestors("x", "y") == {"a", "b"}
        assert index.lowest_common_ancestors("a", "b") == {"r"}
        # Same precomputed structure is reused, not rebuilt.
        assert index.ancestor_sets is before

    def test_staleness_after_mutation_and_explicit_rebuild(self):
        G = self.diamond_diamond()
        index = nx.DAGLCAIndex(G)

        G.add_edge("z", "x")
        G.add_edge("z", "y")
        with pytest.raises(nx.NetworkXError, match="stale LCA index"):
            index.lowest_common_ancestors("x", "y")
        with pytest.raises(nx.NetworkXError, match="stale LCA index"):
            list(index([("x", "y")]))

        index.rebuild()
        assert index.lowest_common_ancestors("x", "y") == {"a", "b", "z"}

        # Same node/edge counts but a different edge set must still be
        # detected: version invalidation cannot rely on counts alone.
        G.remove_edge("z", "y")
        G.add_edge("z", "a")
        with pytest.raises(nx.NetworkXError):
            index.lowest_common_ancestors("x", "y")
        index.rebuild()
        # z now reaches only y, so it is no longer a common ancestor.
        assert index.lowest_common_ancestors("x", "y") == {"a", "b"}

    def test_attribute_change_does_not_invalidate(self):
        # LCA depends only on the node/edge sets, not on attributes.
        G = self.diamond_diamond()
        index = nx.DAGLCAIndex(G)
        G.nodes["x"]["color"] = "red"
        G["r"]["a"]["weight"] = 3
        assert index.lowest_common_ancestors("x", "y") == {"a", "b"}
