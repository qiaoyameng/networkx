"""Regression tests for scheduled-transit earliest-arrival routing."""

from datetime import datetime, timedelta
from itertools import product

import pytest

import networkx as nx
from networkx.algorithms.shortest_paths.transit import (
    TransitLeg,
    TransitRoute,
    TransitTimetable,
    _build_frontier,
    earliest_arrival,
)


def no_wait_wrong_choice_graph():
    """Fast "as-the-crow-flies" service loses to a slower one with a wait.

    Taking the direct departure at t=0 is tempting for a plain shortest-path
    heuristic on in-vehicle time, but waiting for the later fast connection
    arrives earlier.
    """
    G = nx.DiGraph()
    # Direct: boards immediately, arrives 20.
    G.add_edge("A", "C", trips=[(0, 20)])
    # Two-leg alternative: depart later, arrive 15.
    G.add_edge("A", "B", trips=[(5, 7)])
    G.add_edge("B", "C", trips=[(8, 15)])
    return G


def first_service_not_optimal_graph():
    """The first feasible service is overtaken by a later one.

    Boarding the earliest departure (0 -> 10) misses the transfer; waiting for
    the later departure (7 -> 8) catches it and wins overall.
    """
    G = nx.MultiDiGraph()
    G.add_edge("A", "B", trips=[(0, 10), (7, 8)])
    G.add_edge("B", "C", trips=[(9, 12)])
    G.nodes["B"]["transfer_time"] = 1
    return G


def multiedge_graph():
    """Two parallel services on A -> B, selected per starting time."""
    G = nx.MultiDiGraph()
    # At ready=0 boarding "slow" arrives at 3; waiting for "fast" arrives 7.
    # At ready>=4 the slow service is gone and the fast one is the only ride.
    slow = G.add_edge("A", "B", key="slow", trips=[(0, 3)])
    fast = G.add_edge("A", "B", key="fast", trips=[(6, 7)])
    return G, slow, fast


class TestFrontier:
    def test_keeps_suffix_minimum_holders(self):
        d, a = _build_frontier([0, 5, 7], [10, 15, 8])
        # (0, 10) and (5, 15) are dominated by the later (7, 8).
        assert d == [7]
        assert a == [8]

    def test_pareto_ties_equal_departure(self):
        d, a = _build_frontier([3, 3, 9], [6, 5, 12])
        assert d == [3, 9]
        assert a == [5, 12]

    def test_strictly_increasing_outputs(self):
        d, a = _build_frontier([0, 4, 8], [10, 6, 12])
        assert d == [4, 8]
        assert a == [6, 12]
        assert all(x < y for x, y in zip(d, d[1:]))
        assert all(x < y for x, y in zip(a, a[1:]))

    def test_fifo_link_arrival(self):
        d, a = _build_frontier([0, 4, 8], [5, 6, 12])
        from bisect import bisect_left

        def arrival(ready):
            idx = bisect_left(d, ready)
            return a[idx] if idx < len(a) else None

        prev = None
        for ready in range(-2, 12):
            cur = arrival(ready)
            if cur is not None and prev is not None:
                assert cur >= prev
            if cur is not None:
                prev = cur


class TestEarliestArrival:
    def test_no_wait_wrong_choice(self):
        G = no_wait_wrong_choice_graph()
        route = earliest_arrival(G, "A", "C", 0)
        assert route.path == ["A", "B", "C"]
        assert route.arrival_time == 15
        assert [(leg.departure, leg.arrival) for leg in route.legs] == [
            (5, 7),
            (8, 15),
        ]

    def test_direct_choice_when_starting_late(self):
        G = no_wait_wrong_choice_graph()
        # At t=4 the fast A->B at 5 is still reachable; at t=6 only the
        # direct trip already departed, so the alternative wins once more...
        # but starting at t=9 nothing from A is feasible.
        with pytest.raises(nx.NetworkXNoPath):
            earliest_arrival(G, "A", "C", 9)

    def test_first_service_not_optimal(self):
        G = first_service_not_optimal_graph()
        route = earliest_arrival(G, "A", "C", 0)
        assert route.path == ["A", "B", "C"]
        assert route.arrival_time == 12
        assert route.legs[0].departure == 7  # not the (0, 10) service
        assert route.legs[0].arrival == 8
        # minimum transfer time at B: board at >= 8 + 1
        assert route.legs[1].departure == 9

    def test_transfer_time_blocks_connection(self):
        G = first_service_not_optimal_graph()
        G.nodes["B"]["transfer_time"] = 2
        with pytest.raises(nx.NetworkXNoPath):
            earliest_arrival(G, "A", "C", 0)

    def test_multiedge_early_takes_slow(self):
        G, slow, fast = multiedge_graph()
        route = earliest_arrival(G, "A", "B", 0)
        assert route.legs[0].key == "slow"
        assert route.arrival_time == 3

    def test_multiedge_late_takes_fast(self):
        G, slow, fast = multiedge_graph()
        route = earliest_arrival(G, "A", "B", 5)
        assert route.legs[0].key == "fast"
        assert route.arrival_time == 7

    def test_multiedge_path_with_key_sequence(self):
        G, _, _ = multiedge_graph()
        G.add_edge("B", "C", key="local", trips=[(4, 5)])
        G.add_edge("B", "C", key="night", trips=[(8, 9)])
        # From t=0: "slow" arrives B at 3 in time for "local" (4 -> 5).
        route = earliest_arrival(G, "A", "C", 0)
        assert [leg.key for leg in route.legs] == ["slow", "local"]
        assert route.arrival_time == 5
        # From t=4: "slow" already left, so take "fast" then "night".
        route = earliest_arrival(G, "A", "C", 4)
        assert [leg.key for leg in route.legs] == ["fast", "night"]
        assert route.arrival_time == 9

    def test_source_equals_target(self):
        G = first_service_not_optimal_graph()
        route = earliest_arrival(G, "A", "A", 3)
        assert route.path == ["A"]
        assert route.legs == []
        assert route.arrival_time == 3

    def test_unreachable(self):
        G = nx.DiGraph()
        G.add_edge("A", "B", trips=[(0, 5)])
        G.add_edge("C", "B", trips=[(0, 5)])
        with pytest.raises(nx.NetworkXNoPath):
            earliest_arrival(G, "A", "C", 0)

    def test_no_service_after_ready_time(self):
        G = nx.DiGraph()
        G.add_edge("A", "B", trips=[(1, 2)])
        with pytest.raises(nx.NetworkXNoPath):
            earliest_arrival(G, "A", "B", 2)

    def test_undirected_rejected(self):
        G = nx.Graph()
        G.add_edge("A", "B", trips=[(0, 1)])
        with pytest.raises(nx.NetworkXNotImplemented):
            earliest_arrival(G, "A", "B", 0)

    def test_missing_nodes(self):
        G = nx.DiGraph()
        with pytest.raises(nx.NodeNotFound):
            earliest_arrival(G, "A", "B", 0)

    def test_malformed_service(self):
        G = nx.DiGraph()
        G.add_edge("A", "B", trips=[(0, 1, 2)])
        with pytest.raises(nx.NetworkXError):
            earliest_arrival(G, "A", "B", 0)

    def test_arrival_before_departure(self):
        G = nx.DiGraph()
        G.add_edge("A", "B", trips=[(5, 4)])
        with pytest.raises(nx.NetworkXError):
            earliest_arrival(G, "A", "B", 0)

    def test_edge_without_trips_ignored(self):
        G = nx.DiGraph()
        G.add_edge("A", "B", weight=3)
        G.add_edge("A", "B", trips=[(0, 1)]) if G.is_multigraph() else None
        with pytest.raises(nx.NetworkXNoPath):
            earliest_arrival(G, "A", "B", 0)

    def test_route_is_dataclass_with_unpacking(self):
        G = nx.DiGraph()
        G.add_edge("A", "B", trips=[(0, 5)])
        route = earliest_arrival(G, "A", "B", 0)
        assert isinstance(route, TransitRoute)
        assert isinstance(route.legs[0], TransitLeg)
        path, legs, arrival = route
        assert path == ["A", "B"] and arrival == 5 and len(legs) == 1


class TestDatetime:
    def test_datetime_timetable(self):
        t0 = datetime(2024, 1, 1, 8, 0)
        G = nx.DiGraph()
        G.add_edge(
            "A",
            "B",
            trips=[
                (t0, t0 + timedelta(minutes=30)),
                (t0 + timedelta(minutes=20), t0 + timedelta(minutes=25)),
            ],
        )
        G.add_edge(
            "B",
            "C",
            trips=[(t0 + timedelta(minutes=26), t0 + timedelta(minutes=40))],
        )
        G.nodes["B"]["transfer_time"] = timedelta(minutes=1)
        route = earliest_arrival(G, "A", "C", t0)
        assert route.arrival_time == t0 + timedelta(minutes=40)
        assert route.legs[0].departure == t0 + timedelta(minutes=20)


class TestPreprocessedTimetable:
    def test_reuse_timetable(self):
        G, _, _ = multiedge_graph()
        tt = TransitTimetable(G)
        assert tt.route_to("A", "B", 0).arrival_time == 3
        assert tt.route_to("A", "B", 5).arrival_time == 7

    def test_board_returns_none_after_last_service(self):
        tt = TransitTimetable(first_service_not_optimal_graph())
        assert tt.board("A", "B", 0, 100) is None
        leg = tt.board("A", "B", 0, 0)
        assert (leg.departure, leg.arrival) == (7, 8)

    def test_timetable_undirected_rejected(self):
        G = nx.Graph()
        G.add_edge("A", "B")
        with pytest.raises(nx.NetworkXNotImplemented):
            TransitTimetable(G)

    def test_wait_at_source(self):
        G = nx.MultiDiGraph()
        G.add_edge("A", "B", trips=[(0, 2), (3, 5)])
        G.nodes["A"]["transfer_time"] = 2
        # Without waiting at the source the (0, 2) service is boardable.
        assert earliest_arrival(G, "A", "B", 0).arrival_time == 2
        route = earliest_arrival(G, "A", "B", 0, wait_at_source=True)
        assert route.legs[0].departure == 3


class TestRandomizedAgainstEnumeration:
    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_matches_brute_force(self, seed):
        import random

        rng = random.Random(seed)
        G = nx.MultiDiGraph()
        nodes = list(range(5))
        G.add_nodes_from(nodes)
        for u, v in product(nodes, nodes):
            if u == v:
                continue
            if rng.random() < 0.35:
                for key in range(rng.randint(1, 2)):
                    services = []
                    for _ in range(rng.randint(0, 4)):
                        dep = rng.randint(0, 12)
                        arr = dep + rng.randint(0, 6)
                        services.append((dep, arr))
                    if services:
                        G.add_edge(u, v, key=key, trips=services)
        for node in nodes:
            if rng.random() < 0.3:
                G.nodes[node]["transfer_time"] = rng.randint(0, 2)

        start = 0
        for target in nodes:
            for ready in range(0, 14):
                expected = _brute_force_earliest(G, start, target, ready)
                if expected is None:
                    with pytest.raises(nx.NetworkXNoPath):
                        earliest_arrival(G, start, target, ready)
                else:
                    route = earliest_arrival(G, start, target, ready)
                    assert route.arrival_time == expected


def _brute_force_earliest(G, source, target, ready0):
    """Earliest arrival via exhaustive simple-path enumeration."""
    if source == target:
        return ready0
    best = None

    def edges_from(u):
        if G.is_multigraph():
            for v, keydict in G.adj[u].items():
                for key, data in keydict.items():
                    yield v, key, data.get("trips") or []
        else:
            for v, data in G.adj[u].items():
                yield v, None, data.get("trips") or []

    def walk(u, ready, visited, first):
        nonlocal best
        if u == target:
            if best is None or ready < best:
                best = ready
            return
        board_ready = ready + (0 if first else G.nodes[u].get("transfer_time", 0))
        for v, key, services in edges_from(u):
            if v in visited:
                continue
            chosen = None
            for dep, arr in services:
                if dep >= board_ready and (chosen is None or arr < chosen[1]):
                    chosen = (dep, arr)
            if chosen is None:
                continue
            if best is not None and chosen[1] >= best:
                continue
            walk(v, chosen[1], visited | {v}, False)

    walk(source, ready0, {source}, True)
    return best
