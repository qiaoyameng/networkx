"""Tests for single-edge replacement paths."""

from math import inf

import pytest

import networkx as nx
from networkx.utils import pairwise


def _slow_reference(G, s, t, path, keys=None, weight="weight"):
    """Replacement distances by deleting each path edge and rerunning Dijkstra."""
    result = {}
    multi = G.is_multigraph()
    if multi and keys is None:
        keys = []
        for u, v in pairwise(path):
            keys.append(min(G[u][v], key=lambda k: G[u][v][k].get(weight, 1)))
    for i, (u, v) in enumerate(pairwise(path)):
        H = G.copy()
        key = keys[i] if multi else None
        if multi:
            H.remove_edge(u, v, key)
        else:
            H.remove_edge(u, v)
        try:
            length = nx.dijkstra_path_length(H, s, t, weight=weight)
        except nx.NetworkXNoPath:
            length = inf
        failed = (u, v) if not multi else (u, v, key)
        result[failed] = length
    return result


def _assert_replacement_valid(G, s, t, failed, length, route, weight="weight"):
    """Check ``route`` avoids the concrete failed edge and realizes ``length``."""
    if route is None:
        assert length == inf
        return
    assert route[0] == s and route[-1] == t
    H = G.copy()
    if G.is_multigraph():
        H.remove_edge(failed[0], failed[1], failed[2])
    else:
        H.remove_edge(failed[0], failed[1])
    for u, v in pairwise(route):
        assert H.has_edge(u, v)
    if callable(weight):
        wf = weight
    elif H.is_multigraph():
        wf = lambda u, v, d: min(data.get("weight", 1) for data in d.values())
    else:
        wf = lambda u, v, d: d.get(weight, 1)
    walked = sum(wf(u, v, H[u][v]) for u, v in pairwise(route))
    assert walked == length


class TestReplacementPaths:
    def test_simple_graph(self):
        G = nx.Graph()
        G.add_weighted_edges_from([(0, 1, 1), (1, 2, 1), (0, 2, 4)])
        result = nx.replacement_paths(G, 0, 2)
        assert result[(0, 1)] == (4, [0, 2])
        assert result[(1, 2)] == (4, [0, 2])
        assert nx.replacement_path_lengths(G, 0, 2) == {
            (0, 1): 4,
            (1, 2): 4,
        }

    def test_bridge_is_irreplaceable(self):
        G = nx.path_graph(4)
        result = nx.replacement_paths(G, 0, 3)
        for edge in pairwise(range(4)):
            assert result[edge] == (inf, None)
        assert all(v == inf for v in nx.replacement_path_lengths(G, 0, 3).values())

    def test_mixed_bridges_and_detours(self):
        G = nx.Graph()
        G.add_weighted_edges_from(
            [
                (0, 1, 1), (1, 2, 1), (2, 3, 1),  # reference path, length 3
                (0, 4, 1), (4, 3, 7),            # slow outer detour, length 8
                (1, 5, 1), (5, 2, 1),            # parallel to edge (1, 2)
            ]
        )
        result = nx.replacement_path_lengths(G, 0, 3)
        assert result[(0, 1)] == 8
        assert result[(1, 2)] == 4  # 0-1-5-2-3
        assert result[(2, 3)] == 8
        full = nx.replacement_paths(G, 0, 3)
        assert full[(1, 2)][1] == [0, 1, 5, 2, 3]

    def test_equal_weight_shortest_paths(self):
        # Diamond with two equal shortest paths 0-1-2 and 0-3-2.
        G = nx.Graph()
        G.add_weighted_edges_from(
            [(0, 1, 1), (1, 2, 1), (0, 3, 1), (3, 2, 1)]
        )
        for path in ([0, 1, 2], [0, 3, 2]):
            result = nx.replacement_paths(G, 0, 2, path=path)
            assert {length for length, _ in result.values()} == {2}
            for edge, (length, route) in result.items():
                _assert_replacement_valid(G, 0, 2, edge, length, route)

    def test_zero_weight_edges(self):
        G = nx.Graph()
        G.add_weighted_edges_from(
            [(0, 1, 0), (1, 2, 0), (0, 2, 0), (0, 3, 3), (3, 2, 4)]
        )
        result = nx.replacement_paths(G, 0, 2)
        for edge, (length, route) in result.items():
            _assert_replacement_valid(G, 0, 2, edge, length, route)
        expected = _slow_reference(G, 0, 2, nx.dijkstra_path(G, 0, 2))
        assert {e: d for e, (d, _) in result.items()} == expected

    def test_multigraph_removes_one_key_only(self):
        G = nx.MultiGraph()
        G.add_edge(0, 1, key=0, weight=1)
        G.add_edge(0, 1, key=1, weight=5)
        G.add_edge(1, 2, key=0, weight=1)
        result = nx.replacement_paths(G, 0, 2)
        # Losing the cheap parallel key: cheap path 1->2 plus parallel 5 = 6.
        assert result[(0, 1, 0)][0] == 6
        # The parallel key is still usable; the returned node walk uses it.
        _assert_replacement_valid(
            G, 0, 2, (0, 1, 0), *result[(0, 1, 0)]
        )
        # (1,2) is a bridge between the pair, regardless of keys.
        assert result[(1, 2, 0)] == (inf, None)

    def test_multigraph_explicit_keys(self):
        G = nx.MultiGraph()
        G.add_edge(0, 1, key="a", weight=1)
        G.add_edge(0, 1, key="b", weight=4)
        G.add_edge(1, 2, key="x", weight=1)
        result = nx.replacement_paths(
            G, 0, 2, path=[0, 1, 2], path_keys=["a", "x"]
        )
        assert set(result) == {(0, 1, "a"), (1, 2, "x")}
        assert result[(0, 1, "a")][0] == 5  # key "b" (4) + 1
        # A reference path on an even more expensive key is not shortest.
        G.add_edge(0, 1, key="c", weight=9)
        with pytest.raises(nx.NetworkXError, match="shortest"):
            nx.replacement_paths(
                G, 0, 2, path=[0, 1, 2], path_keys=["c", "x"]
            )

    def test_callable_weight_with_hidden_edges(self):
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (0, 2), (2, 3)])
        values = {(0, 1): 2, (1, 2): 2, (0, 2): 1, (2, 3): 3}

        def weight(u, v, d):
            return values[(min(u, v), max(u, v))]

        def hidden_weight(u, v, d):
            return None if (min(u, v), max(u, v)) == (0, 2) else weight(u, v, d)

        result = nx.replacement_paths(G, 0, 3, weight=weight)
        assert result[(0, 2)][0] == 7
        hidden = nx.replacement_paths(G, 0, 3, weight=hidden_weight)
        # With the direct edge hidden, the reference path is via node 1.
        assert set(hidden) == {(0, 1), (1, 2), (2, 3)}

    def test_directed_graph_rejected(self):
        G = nx.path_graph(3, create_using=nx.DiGraph)
        with pytest.raises(nx.NetworkXNotImplemented):
            nx.replacement_paths(G, 0, 2)
        with pytest.raises(nx.NetworkXNotImplemented):
            nx.replacement_path_lengths(G, 0, 2)

    def test_input_graph_not_modified(self):
        G = nx.MultiGraph()
        G.add_edge(0, 1, key=0, weight=1)
        G.add_edge(0, 1, key=1, weight=2)
        G.add_edge(1, 2, weight=1)
        before = (G.number_of_edges(), list(G.edges(keys=True, data=True)))
        nx.replacement_paths(G, 0, 2)
        nx.replacement_path_lengths(G, 0, 2)
        after = (G.number_of_edges(), list(G.edges(keys=True, data=True)))
        assert before == after

    def test_source_equals_target(self):
        G = nx.Graph([(0, 1)])
        assert nx.replacement_paths(G, 0, 0) == {}
        assert nx.replacement_path_lengths(G, 0, 0) == {}

    @pytest.mark.parametrize("bad_path", [[], [1, 0], [0, 0]])
    def test_bad_path(self, bad_path):
        G = nx.Graph()
        G.add_edge(0, 1, weight=1)
        with pytest.raises(nx.NetworkXError):
            nx.replacement_paths(G, 0, 1, path=bad_path)

    def test_nonexistent_edge_in_path(self):
        G = nx.Graph([(0, 1), (1, 2)])
        with pytest.raises(nx.NetworkXError, match="not in G"):
            nx.replacement_paths(G, 0, 2, path=[0, 9, 2])

    def test_missing_key(self):
        G = nx.MultiGraph([(0, 1), (1, 2)])
        with pytest.raises(nx.NetworkXError, match="not in G"):
            nx.replacement_paths(
                G, 0, 2, path=[0, 1, 2], path_keys=["nope", 0]
            )

    def test_path_keys_only_for_multigraph(self):
        G = nx.Graph([(0, 1)])
        with pytest.raises(nx.NetworkXError, match="MultiGraph"):
            nx.replacement_paths(G, 0, 1, path=[0, 1], path_keys=[0])

    def test_node_not_found(self):
        G = nx.Graph([(0, 1)])
        with pytest.raises(nx.NodeNotFound):
            nx.replacement_paths(G, 0, 9)
        with pytest.raises(nx.NodeNotFound):
            nx.replacement_paths(G, 9, 0)

    def test_missing_weight_attribute_defaults_to_one(self):
        G = nx.Graph([(0, 1), (1, 2), (0, 2)])
        result = nx.replacement_path_lengths(G, 0, 2)
        assert result == {(0, 2): 2}

    def test_lengths_interface_returns_only_distances(self):
        G = nx.cycle_graph(4)
        lengths = nx.replacement_path_lengths(G, 0, 2)
        assert all(not isinstance(v, tuple) for v in lengths.values())
        assert lengths == {(0, 1): 2, (1, 2): 2}


class TestAgainstSlowReference:
    """Validate the O(m log n) routine against per-edge Dijkstra reruns."""

    @staticmethod
    def _check(G, s, t, path=None, keys=None, weight="weight"):
        chosen = path if path is not None else nx.dijkstra_path(
            G, s, t, weight=weight
        )
        result = nx.replacement_paths(
            G, s, t, path=path, path_keys=keys, weight=weight
        )
        expected = _slow_reference(G, s, t, chosen, keys, weight=weight)
        assert set(result) == set(expected)
        for failed, (length, route) in result.items():
            assert length == expected[failed], (failed, length, expected[failed])
            _assert_replacement_valid(G, s, t, failed, length, route, weight)

    def test_small_gnm_graphs(self):
        for seed in range(120):
            n = 2 + seed % 6
            import random

            rng = random.Random(seed)
            G = nx.gnm_random_graph(n, rng.randint(n - 1, n * (n - 1) // 2), seed)
            for u, v in G.edges():
                G[u][v]["weight"] = rng.randint(0, 5)
            s, t = 0, n - 1
            if nx.has_path(G, s, t):
                self._check(G, s, t)
                self._check(G, s, t, path=nx.dijkstra_path(G, s, t))

    def test_trees_are_all_bridges(self):
        for seed in range(20):
            G = nx.random_labeled_tree(7, seed=seed)
            for u, v in G.edges():
                G[u][v]["weight"] = seed % 4
            self._check(G, 0, 6)

    def test_multigraphs(self):
        import random

        rng = random.Random(42)
        for seed in range(80):
            base = nx.gnm_random_graph(5, rng.randint(4, 9), seed=seed)
            G = nx.MultiGraph()
            G.add_nodes_from(range(5))
            for u, v in base.edges():
                for _ in range(rng.randint(1, 3)):
                    G.add_edge(u, v, weight=rng.randint(0, 4))
            if not nx.has_path(G, 0, 4):
                continue
            path = nx.dijkstra_path(G, 0, 4)
            keys = [
                min(G[u][v], key=lambda k: G[u][v][k].get("weight", 1))
                for u, v in pairwise(path)
            ]
            self._check(G, 0, 4)
            self._check(G, 0, 4, path=path, keys=keys)

    def test_grid_with_zero_weight_ties(self):
        import random

        rng = random.Random(1)
        G = nx.grid_2d_graph(3, 3)
        for u, v in G.edges():
            G[u][v]["weight"] = rng.choice([0, 0, 1, 2])
        self._check(G, (0, 0), (2, 2))

    def test_callable_weight_random(self):
        import random

        rng = random.Random(7)
        for seed in range(40):
            G = nx.gnm_random_graph(5, rng.randint(4, 9), seed=seed)
            values = {}
            for u, v in G.edges():
                values[(min(u, v), max(u, v))] = rng.randint(0, 5)

            def weight(u, v, d, values=values):
                return values[(min(u, v), max(u, v))]

            if nx.has_path(G, 0, 4):
                self._check(G, 0, 4, weight=weight)
