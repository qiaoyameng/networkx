from collections import deque
from itertools import combinations, permutations

import pytest

import networkx as nx
from networkx.utils import edges_equal, pairwise


# Recipe from the itertools documentation.
def _consume(iterator):
    "Consume the iterator entirely."
    # Feed the entire iterator into a zero-length deque.
    deque(iterator, maxlen=0)


class TestDagLongestPath:
    """Unit tests computing the longest path in a directed acyclic graph."""

    def test_empty(self):
        G = nx.DiGraph()
        assert nx.dag_longest_path(G) == []

    def test_unweighted1(self):
        edges = [(1, 2), (2, 3), (2, 4), (3, 5), (5, 6), (3, 7)]
        G = nx.DiGraph(edges)
        assert nx.dag_longest_path(G) == [1, 2, 3, 5, 6]

    def test_unweighted2(self):
        edges = [(1, 2), (2, 3), (3, 4), (4, 5), (1, 3), (1, 5), (3, 5)]
        G = nx.DiGraph(edges)
        assert nx.dag_longest_path(G) == [1, 2, 3, 4, 5]

    def test_weighted(self):
        G = nx.DiGraph()
        edges = [(1, 2, -5), (2, 3, 1), (3, 4, 1), (4, 5, 0), (3, 5, 4), (1, 6, 2)]
        G.add_weighted_edges_from(edges)
        assert nx.dag_longest_path(G) == [2, 3, 5]

    def test_undirected_not_implemented(self):
        G = nx.Graph()
        pytest.raises(nx.NetworkXNotImplemented, nx.dag_longest_path, G)

    def test_unorderable_nodes(self):
        """Tests that computing the longest path does not depend on
        nodes being orderable.

        For more information, see issue #1989.

        """
        # Create the directed path graph on four nodes in a diamond shape,
        # with nodes represented as (unorderable) Python objects.
        nodes = [object() for n in range(4)]
        G = nx.DiGraph()
        G.add_edge(nodes[0], nodes[1])
        G.add_edge(nodes[0], nodes[2])
        G.add_edge(nodes[2], nodes[3])
        G.add_edge(nodes[1], nodes[3])

        # this will raise NotImplementedError when nodes need to be ordered
        nx.dag_longest_path(G)

    def test_multigraph_unweighted(self):
        edges = [(1, 2), (2, 3), (2, 3), (3, 4), (4, 5), (1, 3), (1, 5), (3, 5)]
        G = nx.MultiDiGraph(edges)
        assert nx.dag_longest_path(G) == [1, 2, 3, 4, 5]

    def test_multigraph_weighted(self):
        G = nx.MultiDiGraph()
        edges = [
            (1, 2, 2),
            (2, 3, 2),
            (1, 3, 1),
            (1, 3, 5),
            (1, 3, 2),
        ]
        G.add_weighted_edges_from(edges)
        assert nx.dag_longest_path(G) == [1, 3]

    def test_multigraph_weighted_default_weight(self):
        G = nx.MultiDiGraph([(1, 2), (2, 3)])  # Unweighted edges
        G.add_weighted_edges_from([(1, 3, 1), (1, 3, 5), (1, 3, 2)])

        # Default value for default weight is 1
        assert nx.dag_longest_path(G) == [1, 3]
        assert nx.dag_longest_path(G, default_weight=3) == [1, 2, 3]


class TestDagLongestPathLength:
    """Unit tests for computing the length of a longest path in a
    directed acyclic graph.

    """

    def test_unweighted(self):
        edges = [(1, 2), (2, 3), (2, 4), (3, 5), (5, 6), (5, 7)]
        G = nx.DiGraph(edges)
        assert nx.dag_longest_path_length(G) == 4

        edges = [(1, 2), (2, 3), (3, 4), (4, 5), (1, 3), (1, 5), (3, 5)]
        G = nx.DiGraph(edges)
        assert nx.dag_longest_path_length(G) == 4

        # test degenerate graphs
        G = nx.DiGraph()
        G.add_node(1)
        assert nx.dag_longest_path_length(G) == 0

    def test_undirected_not_implemented(self):
        G = nx.Graph()
        pytest.raises(nx.NetworkXNotImplemented, nx.dag_longest_path_length, G)

    def test_weighted(self):
        edges = [(1, 2, -5), (2, 3, 1), (3, 4, 1), (4, 5, 0), (3, 5, 4), (1, 6, 2)]
        G = nx.DiGraph()
        G.add_weighted_edges_from(edges)
        assert nx.dag_longest_path_length(G) == 5

    def test_multigraph_unweighted(self):
        edges = [(1, 2), (2, 3), (2, 3), (3, 4), (4, 5), (1, 3), (1, 5), (3, 5)]
        G = nx.MultiDiGraph(edges)
        assert nx.dag_longest_path_length(G) == 4

    def test_multigraph_weighted(self):
        G = nx.MultiDiGraph()
        edges = [
            (1, 2, 2),
            (2, 3, 2),
            (1, 3, 1),
            (1, 3, 5),
            (1, 3, 2),
        ]
        G.add_weighted_edges_from(edges)
        assert nx.dag_longest_path_length(G) == 5


class TestDAG:
    @classmethod
    def setup_class(cls):
        pass

    def test_topological_sort1(self):
        DG = nx.DiGraph([(1, 2), (1, 3), (2, 3)])

        for algorithm in [nx.topological_sort, nx.lexicographical_topological_sort]:
            assert tuple(algorithm(DG)) == (1, 2, 3)

        DG.add_edge(3, 2)

        for algorithm in [nx.topological_sort, nx.lexicographical_topological_sort]:
            pytest.raises(nx.NetworkXUnfeasible, _consume, algorithm(DG))

        DG.remove_edge(2, 3)

        for algorithm in [nx.topological_sort, nx.lexicographical_topological_sort]:
            assert tuple(algorithm(DG)) == (1, 3, 2)

        DG.remove_edge(3, 2)

        assert tuple(nx.topological_sort(DG)) in {(1, 2, 3), (1, 3, 2)}
        assert tuple(nx.lexicographical_topological_sort(DG)) == (1, 2, 3)

    def test_is_directed_acyclic_graph(self):
        G = nx.generators.complete_graph(2)
        assert not nx.is_directed_acyclic_graph(G)
        assert not nx.is_directed_acyclic_graph(G.to_directed())
        assert not nx.is_directed_acyclic_graph(nx.Graph([(3, 4), (4, 5)]))
        assert nx.is_directed_acyclic_graph(nx.DiGraph([(3, 4), (4, 5)]))

    def test_topological_sort2(self):
        DG = nx.DiGraph(
            {
                1: [2],
                2: [3],
                3: [4],
                4: [5],
                5: [1],
                11: [12],
                12: [13],
                13: [14],
                14: [15],
            }
        )
        pytest.raises(nx.NetworkXUnfeasible, _consume, nx.topological_sort(DG))

        assert not nx.is_directed_acyclic_graph(DG)

        DG.remove_edge(1, 2)
        _consume(nx.topological_sort(DG))
        assert nx.is_directed_acyclic_graph(DG)

    def test_topological_sort3(self):
        DG = nx.DiGraph()
        DG.add_edges_from([(1, i) for i in range(2, 5)])
        DG.add_edges_from([(2, i) for i in range(5, 9)])
        DG.add_edges_from([(6, i) for i in range(9, 12)])
        DG.add_edges_from([(4, i) for i in range(12, 15)])

        def validate(order):
            assert isinstance(order, list)
            assert set(order) == set(DG)
            for u, v in combinations(order, 2):
                assert not nx.has_path(DG, v, u)

        validate(list(nx.topological_sort(DG)))

        DG.add_edge(14, 1)
        pytest.raises(nx.NetworkXUnfeasible, _consume, nx.topological_sort(DG))

    def test_topological_sort4(self):
        G = nx.Graph()
        G.add_edge(1, 2)
        # Only directed graphs can be topologically sorted.
        pytest.raises(nx.NetworkXError, _consume, nx.topological_sort(G))

    def test_topological_sort5(self):
        G = nx.DiGraph()
        G.add_edge(0, 1)
        assert list(nx.topological_sort(G)) == [0, 1]

    def test_topological_sort6(self):
        for algorithm in [nx.topological_sort, nx.lexicographical_topological_sort]:

            def runtime_error():
                DG = nx.DiGraph([(1, 2), (2, 3), (3, 4)])
                first = True
                for x in algorithm(DG):
                    if first:
                        first = False
                        DG.add_edge(5 - x, 5)

            def unfeasible_error():
                DG = nx.DiGraph([(1, 2), (2, 3), (3, 4)])
                first = True
                for x in algorithm(DG):
                    if first:
                        first = False
                        DG.remove_node(4)

            def runtime_error2():
                DG = nx.DiGraph([(1, 2), (2, 3), (3, 4)])
                first = True
                for x in algorithm(DG):
                    if first:
                        first = False
                        DG.remove_node(2)

            pytest.raises(RuntimeError, runtime_error)
            pytest.raises(RuntimeError, runtime_error2)
            pytest.raises(nx.NetworkXUnfeasible, unfeasible_error)

    def test_all_topological_sorts_1(self):
        DG = nx.DiGraph([(1, 2), (2, 3), (3, 4), (4, 5)])
        assert list(nx.all_topological_sorts(DG)) == [[1, 2, 3, 4, 5]]

    def test_all_topological_sorts_2(self):
        DG = nx.DiGraph([(1, 3), (2, 1), (2, 4), (4, 3), (4, 5)])
        assert sorted(nx.all_topological_sorts(DG)) == [
            [2, 1, 4, 3, 5],
            [2, 1, 4, 5, 3],
            [2, 4, 1, 3, 5],
            [2, 4, 1, 5, 3],
            [2, 4, 5, 1, 3],
        ]

    def test_all_topological_sorts_3(self):
        def unfeasible():
            DG = nx.DiGraph([(1, 2), (2, 3), (3, 4), (4, 2), (4, 5)])
            # convert to list to execute generator
            list(nx.all_topological_sorts(DG))

        def not_implemented():
            G = nx.Graph([(1, 2), (2, 3)])
            # convert to list to execute generator
            list(nx.all_topological_sorts(G))

        def not_implemented_2():
            G = nx.MultiGraph([(1, 2), (1, 2), (2, 3)])
            list(nx.all_topological_sorts(G))

        pytest.raises(nx.NetworkXUnfeasible, unfeasible)
        pytest.raises(nx.NetworkXNotImplemented, not_implemented)
        pytest.raises(nx.NetworkXNotImplemented, not_implemented_2)

    def test_all_topological_sorts_4(self):
        DG = nx.DiGraph()
        for i in range(7):
            DG.add_node(i)
        assert sorted(map(list, permutations(DG.nodes))) == sorted(
            nx.all_topological_sorts(DG)
        )

    def test_all_topological_sorts_multigraph_1(self):
        DG = nx.MultiDiGraph([(1, 2), (1, 2), (2, 3), (3, 4), (3, 5), (3, 5), (3, 5)])
        assert sorted(nx.all_topological_sorts(DG)) == sorted(
            [[1, 2, 3, 4, 5], [1, 2, 3, 5, 4]]
        )

    def test_all_topological_sorts_multigraph_2(self):
        N = 9
        edges = []
        for i in range(1, N):
            edges.extend([(i, i + 1)] * i)
        DG = nx.MultiDiGraph(edges)
        assert list(nx.all_topological_sorts(DG)) == [list(range(1, N + 1))]

    def test_ancestors(self):
        G = nx.DiGraph()
        ancestors = nx.algorithms.dag.ancestors
        G.add_edges_from([(1, 2), (1, 3), (4, 2), (4, 3), (4, 5), (2, 6), (5, 6)])
        assert ancestors(G, 6) == {1, 2, 4, 5}
        assert ancestors(G, 3) == {1, 4}
        assert ancestors(G, 1) == set()
        pytest.raises(nx.NetworkXError, ancestors, G, 8)

    def test_descendants(self):
        G = nx.DiGraph()
        descendants = nx.algorithms.dag.descendants
        G.add_edges_from([(1, 2), (1, 3), (4, 2), (4, 3), (4, 5), (2, 6), (5, 6)])
        assert descendants(G, 1) == {2, 3, 6}
        assert descendants(G, 4) == {2, 3, 5, 6}
        assert descendants(G, 3) == set()
        pytest.raises(nx.NetworkXError, descendants, G, 8)

    def test_transitive_closure(self):
        G = nx.DiGraph([(1, 2), (2, 3), (3, 4)])
        solution = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
        assert edges_equal(nx.transitive_closure(G).edges(), solution, directed=True)
        G = nx.DiGraph([(1, 2), (2, 3), (2, 4)])
        solution = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4)]
        assert edges_equal(nx.transitive_closure(G).edges(), solution, directed=True)
        G = nx.DiGraph([(1, 2), (2, 3), (3, 1)])
        solution = [(1, 2), (2, 1), (2, 3), (3, 2), (1, 3), (3, 1)]
        soln = sorted(solution + [(n, n) for n in G])
        assert edges_equal(
            sorted(nx.transitive_closure(G).edges()), soln, directed=True
        )

        G = nx.Graph([(1, 2), (2, 3), (3, 4)])
        solution = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
        assert edges_equal(sorted(nx.transitive_closure(G).edges()), solution)

        G = nx.MultiGraph([(1, 2), (2, 3), (3, 4)])
        solution = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
        assert edges_equal(sorted(nx.transitive_closure(G).edges()), solution)

        G = nx.MultiDiGraph([(1, 2), (2, 3), (3, 4)])
        solution = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
        assert edges_equal(
            sorted(nx.transitive_closure(G).edges()), solution, directed=True
        )

        # test if edge data is copied
        G = nx.DiGraph([(1, 2, {"a": 3}), (2, 3, {"b": 0}), (3, 4)])
        H = nx.transitive_closure(G)
        for u, v in G.edges():
            assert G.get_edge_data(u, v) == H.get_edge_data(u, v)

        k = 10
        G = nx.DiGraph((i, i + 1, {"f": "b", "weight": i}) for i in range(k))
        H = nx.transitive_closure(G)
        for u, v in G.edges():
            assert G.get_edge_data(u, v) == H.get_edge_data(u, v)

        G = nx.Graph()
        with pytest.raises(nx.NetworkXError):
            nx.transitive_closure(G, reflexive="wrong input")

    def test_reflexive_transitive_closure(self):
        G = nx.DiGraph([(1, 2), (2, 3), (3, 4)])
        solution = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
        soln = sorted(solution + [(n, n) for n in G])
        assert edges_equal(nx.transitive_closure(G).edges(), solution, directed=True)
        assert edges_equal(
            nx.transitive_closure(G, False).edges(), solution, directed=True
        )
        assert edges_equal(nx.transitive_closure(G, True).edges(), soln, directed=True)
        assert edges_equal(
            nx.transitive_closure(G, None).edges(), solution, directed=True
        )

        G = nx.DiGraph([(1, 2), (2, 3), (2, 4)])
        solution = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4)]
        soln = sorted(solution + [(n, n) for n in G])
        assert edges_equal(nx.transitive_closure(G).edges(), solution, directed=True)
        assert edges_equal(
            nx.transitive_closure(G, False).edges(), solution, directed=True
        )
        assert edges_equal(nx.transitive_closure(G, True).edges(), soln, directed=True)
        assert edges_equal(
            nx.transitive_closure(G, None).edges(), solution, directed=True
        )

        G = nx.DiGraph([(1, 2), (2, 3), (3, 1)])
        solution = sorted([(1, 2), (2, 1), (2, 3), (3, 2), (1, 3), (3, 1)])
        soln = sorted(solution + [(n, n) for n in G])
        assert edges_equal(
            sorted(nx.transitive_closure(G).edges()), soln, directed=True
        )
        assert edges_equal(
            sorted(nx.transitive_closure(G, False).edges()), soln, directed=True
        )
        assert edges_equal(
            sorted(nx.transitive_closure(G, None).edges()), solution, directed=True
        )
        assert edges_equal(
            sorted(nx.transitive_closure(G, True).edges()), soln, directed=True
        )

        G = nx.Graph([(1, 2), (2, 3), (3, 4)])
        solution = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
        soln = sorted(solution + [(n, n) for n in G])
        assert edges_equal(nx.transitive_closure(G).edges(), solution)
        assert edges_equal(nx.transitive_closure(G, False).edges(), solution)
        assert edges_equal(nx.transitive_closure(G, True).edges(), soln)
        assert edges_equal(nx.transitive_closure(G, None).edges(), solution)

        G = nx.MultiGraph([(1, 2), (2, 3), (3, 4)])
        solution = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
        soln = sorted(solution + [(n, n) for n in G])
        assert edges_equal(nx.transitive_closure(G).edges(), solution)
        assert edges_equal(nx.transitive_closure(G, False).edges(), solution)
        assert edges_equal(nx.transitive_closure(G, True).edges(), soln)
        assert edges_equal(nx.transitive_closure(G, None).edges(), solution)

        G = nx.MultiDiGraph([(1, 2), (2, 3), (3, 4)])
        solution = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
        soln = sorted(solution + [(n, n) for n in G])
        assert edges_equal(nx.transitive_closure(G).edges(), solution, directed=True)
        assert edges_equal(
            nx.transitive_closure(G, False).edges(), solution, directed=True
        )
        assert edges_equal(nx.transitive_closure(G, True).edges(), soln, directed=True)
        assert edges_equal(
            nx.transitive_closure(G, None).edges(), solution, directed=True
        )

    def test_transitive_closure_dag(self):
        G = nx.DiGraph([(1, 2), (2, 3), (3, 4)])
        transitive_closure = nx.algorithms.dag.transitive_closure_dag
        solution = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
        assert edges_equal(transitive_closure(G).edges(), solution, directed=True)
        G = nx.DiGraph([(1, 2), (2, 3), (2, 4)])
        solution = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4)]
        assert edges_equal(transitive_closure(G).edges(), solution, directed=True)
        G = nx.Graph([(1, 2), (2, 3), (3, 4)])
        pytest.raises(nx.NetworkXNotImplemented, transitive_closure, G)

        # test if edge data is copied
        G = nx.DiGraph([(1, 2, {"a": 3}), (2, 3, {"b": 0}), (3, 4)])
        H = transitive_closure(G)
        for u, v in G.edges():
            assert G.get_edge_data(u, v) == H.get_edge_data(u, v)

        k = 10
        G = nx.DiGraph((i, i + 1, {"foo": "bar", "weight": i}) for i in range(k))
        H = transitive_closure(G)
        for u, v in G.edges():
            assert G.get_edge_data(u, v) == H.get_edge_data(u, v)

    def test_transitive_reduction(self):
        G = nx.DiGraph([(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)])
        transitive_reduction = nx.algorithms.dag.transitive_reduction
        solution = [(1, 2), (2, 3), (3, 4)]
        assert edges_equal(transitive_reduction(G).edges(), solution, directed=True)
        G = nx.DiGraph([(1, 2), (1, 3), (1, 4), (2, 3), (2, 4)])
        transitive_reduction = nx.algorithms.dag.transitive_reduction
        solution = [(1, 2), (2, 3), (2, 4)]
        assert edges_equal(transitive_reduction(G).edges(), solution, directed=True)
        G = nx.Graph([(1, 2), (2, 3), (3, 4)])
        pytest.raises(nx.NetworkXNotImplemented, transitive_reduction, G)

    def test_antichain_width(self):
        G = nx.DiGraph()
        assert nx.dag.antichain_width(G) == 0

        G.add_nodes_from([0, 1, 2])
        assert nx.dag.antichain_width(G) == 3

        G = nx.path_graph(4, create_using=nx.DiGraph)
        assert nx.dag.antichain_width(G) == 1

        G = nx.DiGraph([(1, 2), (1, 3), (2, 4), (3, 4)])
        assert nx.dag.antichain_width(G) == 2

        G = nx.DiGraph([(1, 3), (2, 3), (3, 4), (3, 5)])
        assert nx.dag.antichain_width(G) == 2

    def test_antichain_width_multidigraph(self):
        G = nx.MultiDiGraph([(1, 2), (1, 2), (1, 3), (2, 4), (3, 4)])
        assert nx.dag.antichain_width(G) == 2

    def test_antichain_width_exceptions(self):
        G = nx.Graph([(1, 2), (2, 3)])
        with pytest.raises(nx.NetworkXNotImplemented):
            nx.dag.antichain_width(G)

        G = nx.DiGraph([(1, 2), (2, 3), (3, 1)])
        with pytest.raises(nx.NetworkXUnfeasible, match="Graph contains a cycle"):
            nx.dag.antichain_width(G)

    def _check_antichains(self, solution, result):
        sol = [frozenset(a) for a in solution]
        res = [frozenset(a) for a in result]
        assert set(sol) == set(res)

    def test_antichains(self):
        antichains = nx.algorithms.dag.antichains
        G = nx.DiGraph([(1, 2), (2, 3), (3, 4)])
        solution = [[], [4], [3], [2], [1]]
        self._check_antichains(list(antichains(G)), solution)
        G = nx.DiGraph([(1, 2), (2, 3), (2, 4), (3, 5), (5, 6), (5, 7)])
        solution = [
            [],
            [4],
            [7],
            [7, 4],
            [6],
            [6, 4],
            [6, 7],
            [6, 7, 4],
            [5],
            [5, 4],
            [3],
            [3, 4],
            [2],
            [1],
        ]
        self._check_antichains(list(antichains(G)), solution)
        G = nx.DiGraph([(1, 2), (1, 3), (3, 4), (3, 5), (5, 6)])
        solution = [
            [],
            [6],
            [5],
            [4],
            [4, 6],
            [4, 5],
            [3],
            [2],
            [2, 6],
            [2, 5],
            [2, 4],
            [2, 4, 6],
            [2, 4, 5],
            [2, 3],
            [1],
        ]
        self._check_antichains(list(antichains(G)), solution)
        G = nx.DiGraph({0: [1, 2], 1: [4], 2: [3], 3: [4]})
        solution = [[], [4], [3], [2], [1], [1, 3], [1, 2], [0]]
        self._check_antichains(list(antichains(G)), solution)
        G = nx.DiGraph()
        self._check_antichains(list(antichains(G)), [[]])
        G = nx.DiGraph()
        G.add_nodes_from([0, 1, 2])
        solution = [[], [0], [1], [1, 0], [2], [2, 0], [2, 1], [2, 1, 0]]
        self._check_antichains(list(antichains(G)), solution)

        def f(x):
            return list(antichains(x))

        G = nx.Graph([(1, 2), (2, 3), (3, 4)])
        pytest.raises(nx.NetworkXNotImplemented, f, G)
        G = nx.DiGraph([(1, 2), (2, 3), (3, 1)])
        pytest.raises(nx.NetworkXUnfeasible, f, G)

    def test_lexicographical_topological_sort(self):
        G = nx.DiGraph([(1, 2), (2, 3), (1, 4), (1, 5), (2, 6)])
        assert list(nx.lexicographical_topological_sort(G)) == [1, 2, 3, 4, 5, 6]
        assert list(nx.lexicographical_topological_sort(G, key=lambda x: x)) == [
            1,
            2,
            3,
            4,
            5,
            6,
        ]
        assert list(nx.lexicographical_topological_sort(G, key=lambda x: -x)) == [
            1,
            5,
            4,
            2,
            6,
            3,
        ]

    def test_lexicographical_topological_sort2(self):
        """
        Check the case of two or more nodes with same key value.
        Want to avoid exception raised due to comparing nodes directly.
        See Issue #3493
        """

        class Test_Node:
            def __init__(self, n):
                self.label = n
                self.priority = 1

            def __repr__(self):
                return f"Node({self.label})"

        def sorting_key(node):
            return node.priority

        test_nodes = [Test_Node(n) for n in range(4)]
        G = nx.DiGraph()
        edges = [(0, 1), (0, 2), (0, 3), (2, 3)]
        G.add_edges_from((test_nodes[a], test_nodes[b]) for a, b in edges)

        sorting = list(nx.lexicographical_topological_sort(G, key=sorting_key))
        assert sorting == test_nodes


def test_topological_generations():
    G = nx.DiGraph(
        {1: [2, 3], 2: [4, 5], 3: [7], 4: [], 5: [6, 7], 6: [], 7: []}
    ).reverse()
    # order within each generation is inconsequential
    generations = [sorted(gen) for gen in nx.topological_generations(G)]
    expected = [[4, 6, 7], [3, 5], [2], [1]]
    assert generations == expected

    MG = nx.MultiDiGraph(G.edges)
    MG.add_edge(2, 1)
    generations = [sorted(gen) for gen in nx.topological_generations(MG)]
    assert generations == expected


def test_topological_generations_empty():
    G = nx.DiGraph()
    assert list(nx.topological_generations(G)) == []


def test_topological_generations_cycle():
    G = nx.DiGraph([[2, 1], [3, 1], [1, 2]])
    with pytest.raises(nx.NetworkXUnfeasible):
        list(nx.topological_generations(G))


def test_is_aperiodic_cycle():
    G = nx.DiGraph()
    nx.add_cycle(G, [1, 2, 3, 4])
    assert not nx.is_aperiodic(G)


def test_is_aperiodic_cycle2():
    G = nx.DiGraph()
    nx.add_cycle(G, [1, 2, 3, 4])
    nx.add_cycle(G, [3, 4, 5, 6, 7])
    assert nx.is_aperiodic(G)


def test_is_aperiodic_cycle3():
    G = nx.DiGraph()
    nx.add_cycle(G, [1, 2, 3, 4])
    nx.add_cycle(G, [3, 4, 5, 6])
    assert not nx.is_aperiodic(G)


def test_is_aperiodic_cycle4():
    G = nx.DiGraph()
    nx.add_cycle(G, [1, 2, 3, 4])
    G.add_edge(1, 3)
    assert nx.is_aperiodic(G)


def test_is_aperiodic_selfloop():
    G = nx.DiGraph()
    nx.add_cycle(G, [1, 2, 3, 4])
    G.add_edge(1, 1)
    assert nx.is_aperiodic(G)


def test_is_aperiodic_forward_edge():
    G = nx.DiGraph([(0, 1), (0, 2), (1, 3), (2, 3), (3, 0)])
    assert not nx.is_aperiodic(G)


def test_is_aperiodic_multiedge():
    G = nx.MultiDiGraph()
    nx.add_cycle(G, [0, 1, 2])
    G.add_edge(0, 1)
    assert not nx.is_aperiodic(G)
    G.add_edge(1, 0)
    assert nx.is_aperiodic(G)


def test_is_aperiodic_two_cycles1():
    G = nx.DiGraph()
    nx.add_cycle(G, [0, 1, 2, 3])
    nx.add_cycle(G, [0, 4, 2])
    assert nx.is_aperiodic(G)


def test_is_aperiodic_two_cycles2():
    G = nx.DiGraph()
    nx.add_cycle(G, [0, 1, 2])
    nx.add_cycle(G, [0, 4, 2])
    assert not nx.is_aperiodic(G)


@pytest.mark.parametrize("graph_type", (nx.Graph, nx.MultiGraph))
def test_is_aperiodic_undirected_raises(graph_type):
    G = graph_type([(1, 2), (2, 3), (3, 1)])
    with pytest.raises(
        nx.NetworkXNotImplemented, match="not implemented for undirected"
    ):
        nx.is_aperiodic(G)


@pytest.mark.parametrize(
    "G",
    [
        nx.DiGraph([(1, 2), (2, 3)]),
        nx.DiGraph([(0, 1), (1, 0), (2, 2)]),
        nx.DiGraph([(0, 1), (1, 2), (0, 3), (3, 4), (4, 0)]),
        nx.DiGraph([(0, 1), (1, 2), (2, 1)]),
        nx.empty_graph(3, create_using=nx.DiGraph),
    ],
)
def test_is_aperiodic_not_sc_raises(G):
    with pytest.raises(nx.NetworkXError, match="not strongly connected"):
        nx.is_aperiodic(G)


def test_is_aperiodic_null_graph():
    G = nx.DiGraph()
    with pytest.raises(nx.NetworkXPointlessConcept, match="Graph has no nodes."):
        nx.is_aperiodic(G)


def test_is_aperiodic_bipartite():
    # Bipartite graph
    G = nx.DiGraph(nx.davis_southern_women_graph())
    assert not nx.is_aperiodic(G)


def test_is_aperiodic_single_node():
    G = nx.DiGraph()
    G.add_node(0)
    assert not nx.is_aperiodic(G)
    G.add_edge(0, 0)
    assert nx.is_aperiodic(G)


class TestDagToBranching:
    """Unit tests for the :func:`networkx.dag_to_branching` function."""

    def test_single_root(self):
        """Tests that a directed acyclic graph with a single degree
        zero node produces an arborescence.

        """
        G = nx.DiGraph([(0, 1), (0, 2), (1, 3), (2, 3)])
        B = nx.dag_to_branching(G)
        expected = nx.DiGraph([(0, 1), (1, 3), (0, 2), (2, 4)])
        assert nx.is_arborescence(B)
        assert nx.is_isomorphic(B, expected)

    def test_multiple_roots(self):
        """Tests that a directed acyclic graph with multiple degree zero
        nodes creates an arborescence with multiple (weakly) connected
        components.

        """
        G = nx.DiGraph([(0, 1), (0, 2), (1, 3), (2, 3), (5, 2)])
        B = nx.dag_to_branching(G)
        expected = nx.DiGraph([(0, 1), (1, 3), (0, 2), (2, 4), (5, 6), (6, 7)])
        assert nx.is_branching(B)
        assert not nx.is_arborescence(B)
        assert nx.is_isomorphic(B, expected)

    # # Attributes are not copied by this function. If they were, this would
    # # be a good test to uncomment.
    # def test_copy_attributes(self):
    #     """Tests that node attributes are copied in the branching."""
    #     G = nx.DiGraph([(0, 1), (0, 2), (1, 3), (2, 3)])
    #     for v in G:
    #         G.node[v]['label'] = str(v)
    #     B = nx.dag_to_branching(G)
    #     # Determine the root node of the branching.
    #     root = next(v for v, d in B.in_degree() if d == 0)
    #     assert_equal(B.node[root]['label'], '0')
    #     children = B[root]
    #     # Get the left and right children, nodes 1 and 2, respectively.
    #     left, right = sorted(children, key=lambda v: B.node[v]['label'])
    #     assert_equal(B.node[left]['label'], '1')
    #     assert_equal(B.node[right]['label'], '2')
    #     # Get the left grandchild.
    #     children = B[left]
    #     assert_equal(len(children), 1)
    #     left_grandchild = arbitrary_element(children)
    #     assert_equal(B.node[left_grandchild]['label'], '3')
    #     # Get the right grandchild.
    #     children = B[right]
    #     assert_equal(len(children), 1)
    #     right_grandchild = arbitrary_element(children)
    #     assert_equal(B.node[right_grandchild]['label'], '3')

    def test_already_arborescence(self):
        """Tests that a directed acyclic graph that is already an
        arborescence produces an isomorphic arborescence as output.

        """
        A = nx.balanced_tree(2, 2, create_using=nx.DiGraph())
        B = nx.dag_to_branching(A)
        assert nx.is_isomorphic(A, B)

    def test_already_branching(self):
        """Tests that a directed acyclic graph that is already a
        branching produces an isomorphic branching as output.

        """
        T1 = nx.balanced_tree(2, 2, create_using=nx.DiGraph())
        T2 = nx.balanced_tree(2, 2, create_using=nx.DiGraph())
        G = nx.disjoint_union(T1, T2)
        B = nx.dag_to_branching(G)
        assert nx.is_isomorphic(G, B)

    def test_not_acyclic(self):
        """Tests that a non-acyclic graph causes an exception."""
        with pytest.raises(nx.HasACycle):
            G = nx.DiGraph(pairwise("abc", cyclic=True))
            nx.dag_to_branching(G)

    def test_undirected(self):
        with pytest.raises(nx.NetworkXNotImplemented):
            nx.dag_to_branching(nx.Graph())

    def test_multigraph(self):
        with pytest.raises(nx.NetworkXNotImplemented):
            nx.dag_to_branching(nx.MultiGraph())

    def test_multidigraph(self):
        with pytest.raises(nx.NetworkXNotImplemented):
            nx.dag_to_branching(nx.MultiDiGraph())


def test_ancestors_descendants_undirected():
    """Regression test to ensure ancestors and descendants work as expected on
    undirected graphs."""
    G = nx.path_graph(5)
    nx.ancestors(G, 2) == nx.descendants(G, 2) == {0, 1, 3, 4}


def test_v_structures_raise():
    G = nx.Graph()
    with pytest.raises(nx.NetworkXNotImplemented, match="for undirected type"):
        nx.dag.v_structures(G)


@pytest.mark.parametrize(
    ("edgelist", "expected"),
    (
        (
            [(0, 1), (0, 2), (3, 2)],
            {(0, 2, 3)},
        ),
        (
            [("A", "B"), ("C", "B"), ("D", "G"), ("D", "E"), ("G", "E")],
            {("A", "B", "C")},
        ),
        ([(0, 1), (2, 1), (0, 2)], set()),  # adjacent parents case: see gh-7385
    ),
)
def test_v_structures(edgelist, expected):
    G = nx.DiGraph(edgelist)
    v_structs = set(nx.dag.v_structures(G))
    assert v_structs == expected


def test_colliders_raise():
    G = nx.Graph()
    with pytest.raises(nx.NetworkXNotImplemented, match="for undirected type"):
        nx.dag.colliders(G)


@pytest.mark.parametrize(
    ("edgelist", "expected"),
    (
        (
            [(0, 1), (0, 2), (3, 2)],
            {(0, 2, 3)},
        ),
        (
            [("A", "B"), ("C", "B"), ("D", "G"), ("D", "E"), ("G", "E")],
            {("A", "B", "C"), ("D", "E", "G")},
        ),
    ),
)
def test_colliders(edgelist, expected):
    G = nx.DiGraph(edgelist)
    colliders = set(nx.dag.colliders(G))
    assert colliders == expected



class TestIncrementalTopologicalOrder:
    def test_empty_init(self):
        ito = nx.IncrementalTopologicalOrder()
        assert len(ito) == 0
        assert ito.order == []
        assert ito.is_valid()
        assert isinstance(ito.graph, nx.DiGraph)

    def test_init_from_graph(self):
        G = nx.DiGraph([(0, 1), (1, 2), (0, 2)])
        ito = nx.IncrementalTopologicalOrder(G)
        assert ito.graph is G
        assert sorted(ito.order) == [0, 1, 2]
        assert ito.is_valid()

    def test_init_cyclic_graph_raises_with_cycle(self):
        G = nx.DiGraph([(0, 1), (1, 2), (2, 0)])
        with pytest.raises(nx.HasACycle) as excinfo:
            nx.IncrementalTopologicalOrder(G)
        cycle = excinfo.value.cycle
        assert cycle[0] == cycle[-1]
        assert set(cycle) == {0, 1, 2}

    def test_undirected_rejected(self):
        with pytest.raises(nx.NetworkXError):
            nx.IncrementalTopologicalOrder(nx.Graph())

    def test_multigraph_rejected(self):
        with pytest.raises(nx.NetworkXNotImplemented):
            nx.IncrementalTopologicalOrder(nx.MultiDiGraph())

    def test_add_node_and_nodes_from(self):
        ito = nx.IncrementalTopologicalOrder()
        ito.add_node("a", color="red")
        ito.add_nodes_from(["b", ("c", {"size": 3})], color="blue")
        assert ito.order == ["a", "b", "c"]
        assert ito.graph.nodes["a"]["color"] == "red"
        assert ito.graph.nodes["c"]["size"] == 3
        assert ito.graph.nodes["c"]["color"] == "blue"
        # Re-adding an existing node only updates attributes.
        ito.add_node("a", color="green")
        assert ito.order == ["a", "b", "c"]
        assert ito.graph.nodes["a"]["color"] == "green"
        assert ito.nodes_visited == 0

    def test_forward_edge_needs_no_search(self):
        ito = nx.IncrementalTopologicalOrder(nx.DiGraph([(0, 1)]))
        ito.add_edge(0, 2)  # new node appended at the end
        assert ito.order == [0, 1, 2]
        ito.add_edge(1, 2)  # already consistent with the order
        assert ito.nodes_visited == 0
        assert ito.order == [0, 1, 2]
        assert ito.is_valid()

    def test_duplicate_edge_is_noop(self):
        ito = nx.IncrementalTopologicalOrder()
        ito.add_edge(0, 1, weight=1)
        order_before = ito.order
        ito.add_edge(0, 1, weight=5)
        assert ito.order == order_before
        assert ito.graph.number_of_edges() == 1
        assert ito.graph[0][1]["weight"] == 5
        assert ito.nodes_visited == 0


    def test_remove_edge_keeps_order(self):
        ito = nx.IncrementalTopologicalOrder()
        ito.add_edges_from([(0, 1), (1, 2), (0, 2)])
        order_before = ito.order
        ito.remove_edge(0, 2)
        assert ito.order == order_before
        assert ito.is_valid()
        ito.remove_edges_from([(0, 1), (9, 9)])  # missing edges ignored
        assert ito.is_valid()
        with pytest.raises(nx.NetworkXError):
            ito.remove_edge(0, 1)
    def test_backward_edge_reorders_interval(self):
        ito = nx.IncrementalTopologicalOrder()
        ito.add_edges_from([(0, 1), (1, 2), (2, 3), (0, 4), (4, 3)])
        assert ito.order == [0, 1, 2, 4, 3]
        ito.add_edge(4, 1)  # backward edge, but no path 1 ~> 4 exists
        assert ito.order == [0, 4, 1, 2, 3]
        assert ito.is_valid()
        # Only the interval between the endpoints was searched.
        assert 0 < ito.nodes_visited <= 4

    def test_backward_edge_with_new_source_node(self):
        ito = nx.IncrementalTopologicalOrder(nx.DiGraph([(0, 1)]))
        ito.add_edge(5, 0)  # new node appended, then moved before 0
        assert ito.order == [5, 0, 1]
        assert ito.is_valid()

    def test_cycle_edge_raises_with_actual_cycle(self):
        ito = nx.IncrementalTopologicalOrder()
        ito.add_edges_from([(0, 1), (1, 2), (2, 3)])
        edges_before = set(ito.graph.edges)
        order_before = ito.order
        with pytest.raises(nx.HasACycle) as excinfo:
            ito.add_edge(3, 1)
        cycle = excinfo.value.cycle
        assert cycle == [3, 1, 2, 3]
        # Every consecutive pair is an edge of the graph plus the new edge.
        for x, y in pairwise(cycle):
            assert (x, y) in edges_before or (x, y) == (3, 1)
        # Graph and order are left in their pre-commit state.
        assert set(ito.graph.edges) == edges_before
        assert ito.order == order_before
        assert ito.is_valid()

    def test_self_loop_rejected(self):
        ito = nx.IncrementalTopologicalOrder()
        ito.add_node(0)
        with pytest.raises(nx.HasACycle) as excinfo:
            ito.add_edge(0, 0)
        assert excinfo.value.cycle == [0, 0]
        assert ito.graph.number_of_edges() == 0
        # A self-loop on a missing node must not add the node.
        with pytest.raises(nx.HasACycle):
            ito.add_edge(7, 7)
        assert 7 not in ito.graph

    def test_add_edges_from_batch_success(self):
        ito = nx.IncrementalTopologicalOrder()
        ito.add_edges_from([(0, 1), (1, 2)], weight=3)
        ito.add_edges_from([(2, 3, {"weight": 7}), (0, 4), "ab"])
        assert ito.is_valid()
        assert ito.graph[1][2]["weight"] == 3
        assert ito.graph[2][3]["weight"] == 7
        assert ito.graph.has_edge("a", "b")

    def test_add_edges_from_batch_is_atomic(self):
        ito = nx.IncrementalTopologicalOrder()
        ito.add_edges_from([(0, 1), (1, 2)])
        edges_before = set(ito.graph.edges)
        nodes_before = set(ito.graph.nodes)
        order_before = ito.order
        with pytest.raises(nx.HasACycle):
            ito.add_edges_from([(2, 3), (3, 0), (4, 5)])  # (3, 0) closes a cycle
        assert set(ito.graph.edges) == edges_before
        assert set(ito.graph.nodes) == nodes_before
        assert ito.order == order_before
        assert ito.is_valid()

    def test_external_modification_detected(self):
        G = nx.DiGraph([(0, 1)])
        ito = nx.IncrementalTopologicalOrder(G)
        G.add_edge(1, 2)  # mutation behind the maintainer's back
        with pytest.raises(nx.NetworkXError, match="modified outside"):
            ito.add_edge(2, 3)
        with pytest.raises(nx.NetworkXError, match="modified outside"):
            ito.order

    def test_nodes_need_not_be_comparable(self):
        class Token:
            def __init__(self, name):
                self.name = name

            def __repr__(self):
                return f"Token({self.name})"

        tokens = [Token(str(i)) for i in range(6)]
        ito = nx.IncrementalTopologicalOrder()
        ito.add_nodes_from(tokens)
        ito.add_edge(tokens[0], tokens[3])
        ito.add_edge(tokens[3], tokens[5])
        ito.add_edge(tokens[5], tokens[1])  # backward edge triggers reorder
        assert ito.is_valid()
        pos = ito.position
        assert pos(tokens[0]) < pos(tokens[3]) < pos(tokens[5]) < pos(tokens[1])

    def test_local_insert_does_not_scan_whole_graph(self):
        # 980 isolated nodes followed by two chains of 10 nodes each.
        n = 1000
        edges = [(i, i + 1) for i in range(980, 989)]
        edges += [(i, i + 1) for i in range(990, 999)]
        ito = nx.IncrementalTopologicalOrder(nx.DiGraph(edges))
        ito.add_nodes_from(range(980))
        # Forward edge: accepted directly, no search at all.
        ito.add_edge(0, 500)
        assert ito.nodes_visited == 0
        # Backward edge within the two chains: only the affected interval
        # (the 20 chain nodes) is searched, not the whole 1000-node graph.
        ito.add_edge(989, 990)
        assert ito.is_valid()
        assert 0 < ito.nodes_visited <= 20

    def test_randomized_cross_validation(self):
        import random
        rng = random.Random(42)
        ito = nx.IncrementalTopologicalOrder()
        nodes = list(range(30))
        ito.add_nodes_from(nodes)
        for step in range(500):
            op = rng.randrange(4)
            if op == 0:
                u, v = rng.sample(nodes, 2)
                edges_before = set(ito.graph.edges)
                order_before = ito.order
                try:
                    ito.add_edge(u, v)
                except nx.HasACycle as err:
                    # The reported cycle must be real in graph + new edge.
                    for x, y in pairwise(err.cycle):
                        assert (x, y) in edges_before or (x, y) == (u, v)
                    assert err.cycle[0] == err.cycle[-1]
                    # State is unchanged.
                    assert set(ito.graph.edges) == edges_before
                    assert ito.order == order_before
                    # The edge really does close a cycle.
                    H = nx.DiGraph(edges_before)
                    H.add_edge(u, v)
                    assert not nx.is_directed_acyclic_graph(H)
                else:
                    assert nx.is_directed_acyclic_graph(ito.graph)
            elif op == 1 and ito.graph.number_of_edges() > 0:
                ito.remove_edge(*rng.choice(list(ito.graph.edges)))
            elif op == 2:
                ito.add_node(f"extra-{step}")
            else:
                batch = [tuple(rng.sample(nodes, 2)) for _ in range(5)]
                try:
                    ito.add_edges_from(batch)
                except nx.HasACycle:
                    pass
            # Cross-validate the maintained order against a full check.
            assert ito.is_valid()
            assert set(ito.order) == set(ito.graph.nodes)
