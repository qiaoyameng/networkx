import math
import random

import pytest

import networkx as nx


def slow_replacement_paths(G, source, target, path, keys=None, weight="weight"):
    """Reference: delete each path edge individually and re-run Dijkstra."""
    result = {}
    for j in range(len(path) - 1):
        u, v = path[j], path[j + 1]
        H = G.copy()
        if G.is_multigraph():
            H.remove_edge(u, v, key=keys[j])
            edge_id = (u, v, keys[j])
        else:
            H.remove_edge(u, v)
            edge_id = (u, v)
        try:
            dist = nx.dijkstra_path_length(H, source, target, weight=weight)
            nodes = nx.dijkstra_path(H, source, target, weight=weight)
        except nx.NetworkXNoPath:
            dist, nodes = math.inf, None
        result[edge_id] = (dist, nodes)
    return result


def path_cost(G, nodes, weight="weight"):
    total = 0
    for u, v in zip(nodes, nodes[1:]):
        if G.is_multigraph():
            total += min(d.get(weight, 1) for d in G[u][v].values())
        else:
            total += G[u][v].get(weight, 1)
    return total


def auto_path_keys(G, result):
    """Recover the path edges (with keys) used as result keys."""
    return list(result)


class TestReplacementPaths:
    def test_path_graph_bridge(self):
        # Every edge of a path graph is a bridge: no replacement exists.
        G = nx.path_graph(5)
        result = nx.replacement_paths(G, 0, 4)
        assert len(result) == 4
        for dist in result.values():
            assert dist == math.inf

    def test_cycle(self):
        G = nx.cycle_graph(5)
        result = nx.replacement_paths(G, 0, 2)
        # shortest path is 0-1-2; detour around either edge is the long way
        assert result == {(0, 1): 3, (1, 2): 3}

    def test_example(self):
        G = nx.Graph()
        G.add_weighted_edges_from(
            [(0, 1, 1), (1, 2, 1), (2, 3, 1), (0, 2, 3), (1, 3, 5)]
        )
        result = nx.replacement_paths(G, 0, 3)
        assert result == {(0, 1): 4, (1, 2): 4, (2, 3): 6}
        result = nx.replacement_paths(G, 0, 3, return_paths=True)
        for edge, (dist, nodes) in result.items():
            H = G.copy()
            H.remove_edge(*edge)
            assert path_cost(H, nodes) == dist

    def test_zero_weight_edges(self):
        G = nx.Graph()
        G.add_weighted_edges_from(
            [(0, 1, 0), (1, 2, 0), (2, 3, 0), (0, 3, 1), (0, 2, 2)]
        )
        result = nx.replacement_paths(G, 0, 3)
        assert result == {(0, 1): 1, (1, 2): 1, (2, 3): 1}

    def test_tied_shortest_paths(self):
        # Two equal shortest paths; each edge's replacement uses the other.
        G = nx.Graph()
        G.add_weighted_edges_from(
            [("s", "a", 1), ("a", "t", 1), ("s", "b", 1), ("b", "t", 1)]
        )
        result = nx.replacement_paths(G, "s", "t")
        assert all(dist == 2 for dist in result.values())

    def test_multigraph_parallel_edges(self):
        G = nx.MultiGraph()
        G.add_edge(0, 1, key="a", weight=1)
        G.add_edge(0, 1, key="b", weight=2)
        G.add_edge(1, 2, key="c", weight=1)
        result = nx.replacement_paths(G, 0, 2)
        # failing (0,1,'a') leaves the parallel edge (0,1,'b')
        assert result[(0, 1, "a")] == 3
        assert result[(1, 2, "c")] == math.inf

    def test_multigraph_only_single_edge_removed(self):
        G = nx.MultiGraph()
        G.add_edge(0, 1, key="x", weight=1)
        G.add_edge(0, 1, key="y", weight=5)
        result = nx.replacement_paths(G, 0, 1)
        # removing the cheap edge leaves the expensive parallel edge
        assert result[(0, 1, "x")] == 5

    def test_user_supplied_path(self):
        G = nx.Graph()
        G.add_weighted_edges_from(
            [(0, 1, 1), (1, 2, 1), (0, 2, 2), (2, 3, 1), (0, 3, 10)]
        )
        # two shortest 0-3 paths of weight 3: 0-1-2-3 and 0-2-3
        result = nx.replacement_paths(G, 0, 3, path=[0, 1, 2, 3])
        assert set(result) == {(0, 1), (1, 2), (2, 3)}
        assert result[(0, 1)] == 3  # via 0-2-3
        assert result[(2, 3)] == 10  # via 0-3

    def test_user_supplied_path_multigraph_keys(self):
        G = nx.MultiGraph()
        G.add_edge(0, 1, key="a", weight=1)
        G.add_edge(0, 1, key="b", weight=1)
        G.add_edge(1, 2, key="c", weight=1)
        result = nx.replacement_paths(G, 0, 2, path=[0, 1, 2], path_keys=["b", "c"])
        assert set(result) == {(0, 1, "b"), (1, 2, "c")}
        assert result[(0, 1, "b")] == 2  # parallel edge (0,1,'a') + (1,2,'c')

    def test_invalid_path_raises(self):
        G = nx.Graph()
        G.add_weighted_edges_from([(0, 1, 1), (1, 2, 1), (0, 2, 1)])
        with pytest.raises(nx.NetworkXError):  # not a shortest path
            nx.replacement_paths(G, 0, 2, path=[0, 1, 2])
        with pytest.raises(nx.NetworkXError):  # wrong endpoints
            nx.replacement_paths(G, 0, 2, path=[1, 0, 2])
        with pytest.raises(nx.NetworkXError):  # non-existent edge
            nx.replacement_paths(G, 0, 2, path=[0, 3, 2])
        with pytest.raises(nx.NetworkXError):  # repeated nodes
            nx.replacement_paths(G, 0, 2, path=[0, 1, 0, 2])

    def test_invalid_path_keys_raise(self):
        G = nx.MultiGraph()
        G.add_edge(0, 1, key="a", weight=1)
        G.add_edge(1, 2, key="b", weight=1)
        with pytest.raises(nx.NetworkXError):
            nx.replacement_paths(G, 0, 2, path=[0, 1, 2], path_keys=["a", "zzz"])
        with pytest.raises(nx.NetworkXError):
            nx.replacement_paths(G, 0, 2, path=[0, 1, 2], path_keys=["a"])
        H = nx.Graph()
        H.add_edge(0, 1)
        with pytest.raises(nx.NetworkXError):  # keys on a non-multigraph
            nx.replacement_paths(H, 0, 1, path=[0, 1], path_keys=[None])

    def test_directed_not_implemented(self):
        G = nx.DiGraph()
        G.add_edge(0, 1)
        with pytest.raises(nx.NetworkXNotImplemented):
            nx.replacement_paths(G, 0, 1)

    def test_missing_endpoints(self):
        G = nx.path_graph(3)
        with pytest.raises(nx.NodeNotFound):
            nx.replacement_paths(G, 0, 99)
        with pytest.raises(nx.NodeNotFound):
            nx.replacement_paths(G, 99, 0)

    def test_no_path(self):
        G = nx.Graph()
        G.add_nodes_from([0, 1])
        with pytest.raises(nx.NetworkXNoPath):
            nx.replacement_paths(G, 0, 1)

    def test_source_equals_target(self):
        G = nx.path_graph(3)
        assert nx.replacement_paths(G, 1, 1) == {}

    def test_graph_not_modified(self):
        G = nx.MultiGraph()
        G.add_edge(0, 1, key="a", weight=1)
        G.add_edge(0, 1, key="b", weight=2)
        G.add_edge(1, 2, key="c", weight=1)
        before = list(G.edges(keys=True, data=True))
        nx.replacement_paths(G, 0, 2, return_paths=True)
        assert list(G.edges(keys=True, data=True)) == before

    def test_callable_weight(self):
        G = nx.Graph()
        G.add_edge(0, 1, cost=1)
        G.add_edge(1, 2, cost=1)
        G.add_edge(0, 2, cost=4)
        result = nx.replacement_paths(G, 0, 2, weight=lambda u, v, d: d["cost"])
        assert result == {(0, 1): 4, (1, 2): 4}

    def test_missing_weight_attribute_defaults_to_one(self):
        G = nx.path_graph(4)  # no weight attributes
        result = nx.replacement_paths(G, 0, 3)
        assert all(dist == math.inf for dist in result.values())
        G.add_edge(0, 3)
        result = nx.replacement_paths(G, 0, 3)
        assert result == {(0, 3): 3}

    @pytest.mark.parametrize("multigraph", [False, True])
    @pytest.mark.parametrize("seed", range(60))
    def test_against_edge_deletion_reference(self, multigraph, seed):
        rng = random.Random(seed)
        n = 2 + rng.randrange(8)
        m = rng.randrange(2 * n + 4)
        G = nx.MultiGraph() if multigraph else nx.Graph()
        G.add_nodes_from(range(n))
        for _ in range(m):
            u, v = rng.randrange(n), rng.randrange(n)
            if u != v:
                G.add_edge(u, v, weight=rng.randrange(0, 6))  # incl. zero weights
        source, target = 0, n - 1
        try:
            nx.dijkstra_path_length(G, source, target)
        except nx.NetworkXNoPath:
            pytest.skip("disconnected")
        result = nx.replacement_paths(G, source, target, return_paths=True)
        # recover the path the function used, from the result keys
        edges = list(result)
        path = [edges[0][0]] + [e[1] for e in edges]
        keys = [e[2] for e in edges] if multigraph else None
        reference = slow_replacement_paths(G, source, target, path, keys)
        assert set(result) == set(reference)
        for edge_id, (dist, _) in reference.items():
            got_dist, got_path = result[edge_id]
            assert got_dist == dist
            if got_path is not None:
                H = G.copy()
                H.remove_edge(*edge_id)
                assert path_cost(H, got_path) == got_dist
                assert got_path[0] == source and got_path[-1] == target

    @pytest.mark.parametrize("seed", range(20))
    def test_user_path_matches_auto(self, seed):
        rng = random.Random(1000 + seed)
        G = nx.gnm_random_graph(8, 16, seed=rng)
        for u, v in G.edges:
            G[u][v]["weight"] = rng.randrange(1, 5)
        try:
            path = nx.dijkstra_path(G, 0, 7)
        except nx.NetworkXNoPath:
            pytest.skip("disconnected")
        auto = nx.replacement_paths(G, 0, 7)
        manual = nx.replacement_paths(G, 0, 7, path=path)
        reference = slow_replacement_paths(G, 0, 7, path)
        for edge_id, (dist, _) in reference.items():
            assert manual[edge_id] == dist
        # same shortest path -> identical results
        assert manual == auto
