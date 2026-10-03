"""Unit tests for timetable (scheduled-service) routing."""

from datetime import datetime, timedelta

import pytest

import networkx as nx
from networkx.algorithms.timetable import Timetable, Trip


def no_wait_shortest_path(G, source, target, departure_time):
    """Static shortest path that ignores schedules and waiting.

    Every edge is weighted by its shortest possible service duration and
    Dijkstra is run as if boarding were immediate, so unavailable fast
    services are optimistically counted. This is the mistake the scheduled
    label-setting search avoids.
    """
    return nx.shortest_path_length(
        G,
        source,
        target,
        weight=lambda u, v, data: min(
            arrival - departure for departure, arrival, *rest in data["trips"]
        ),
    )


def make_no_wait_mistake_graph():
    """Graph where static travel-duration shortest path is wrong.

    The direct edge has a fast service early in the day (duration 3), which
    makes the static path pick it, but at the requested departure time that
    service is gone and the remaining direct service is slow. The two-hop
    route, despite larger static duration, arrives earlier.
    """
    G = nx.DiGraph()
    G.add_edge("A", "B", trips=[(10, 12)])
    G.add_edge("B", "C", trips=[(12, 14)])
    G.add_edge("A", "C", trips=[(0, 3), (10, 30)])
    return G


def make_first_service_graph():
    """Single edge whose first feasible service is not the best.

    Binary search finds the 5 o'clock service first, but the 7 o'clock
    service arrives earlier.
    """
    G = nx.DiGraph()
    G.add_edge("A", "B", trips=[(0, 20), (5, 18), (7, 9), (10, 15)])
    return G


def make_multiedge_graph():
    """Parallel edges between the same nodes, keyed by service line."""
    G = nx.MultiDiGraph()
    G.add_edge("A", "B", key="local", trips=[(0, 30), (20, 45)])
    G.add_edge("A", "B", key="express", trips=[(2, 25), (15, 22)])
    return G


class TestEarliestArrival:
    def test_no_wait_shortest_path_picks_wrongly(self):
        G = make_no_wait_mistake_graph()
        # The static length 3 comes from the fast (0, 3) direct service,
        # which has already left by the requested departure time 10.
        assert no_wait_shortest_path(G, "A", "C", 10) == 3
        result = nx.earliest_arrival_path(G, "A", "C", 10)
        assert result.arrival_time == 14
        assert result.nodes == ["A", "B", "C"]
        assert result.edges == [("A", "B", None), ("B", "C", None)]
        assert result.trips == [Trip(10, 12, None), Trip(12, 14, None)]

    def test_first_bisected_service_is_not_optimal(self):
        G = make_first_service_graph()
        result = nx.earliest_arrival_path(G, "A", "B", 5)
        assert result.trips == [Trip(7, 9, None)]
        assert result.arrival_time == 9
        assert result.edges == [("A", "B", None)]

    def test_first_service_when_it_is_best(self):
        G = nx.DiGraph()
        G.add_edge("A", "B", trips=[(0, 5), (7, 9)])
        result = nx.earliest_arrival_path(G, "A", "B", 0)
        assert result.trips == [Trip(0, 5, None)]

    def test_parallel_edges_respect_keys(self):
        G = make_multiedge_graph()
        result = nx.earliest_arrival_path(G, "A", "B", 1)
        assert result.arrival_time == 22
        assert result.edges == [("A", "B", "express")]
        assert result.trips == [Trip(15, 22, None)]

        earliest = nx.earliest_arrival_path(G, "A", "B", 0)
        assert earliest.edges == [("A", "B", "express")]
        assert earliest.trips == [Trip(15, 22, None)]

        late = nx.earliest_arrival_path(G, "A", "B", 16)
        assert late.edges == [("A", "B", "local")]
        assert late.trips == [Trip(20, 45, None)]

    def test_multiedge_keys_survive_over_path(self):
        G = nx.MultiDiGraph()
        G.add_edge("A", "B", key="slow", trips=[(0, 100)])
        G.add_edge("A", "B", key="fast", trips=[(1, 5)])
        G.add_edge("B", "C", key="night", trips=[(5, 9)])
        result = nx.earliest_arrival_path(G, "A", "C", 0)
        assert result.edges == [("A", "B", "fast"), ("B", "C", "night")]
        assert result.arrival_time == 9

    def test_transfer_time_at_intermediate_node(self):
        G = nx.DiGraph()
        G.add_edge("A", "B", trips=[(0, 10)])
        G.add_edge("B", "C", trips=[(11, 20), (15, 25)])
        G.nodes["B"]["transfer_time"] = 5
        result = nx.earliest_arrival_path(G, "A", "C", 0)
        assert result.trips == [Trip(0, 10, None), Trip(15, 25, None)]
        assert result.arrival_time == 25

    def test_transfer_time_not_applied_at_source(self):
        G = nx.DiGraph()
        G.add_edge("A", "B", trips=[(0, 5)])
        G.nodes["A"]["transfer_time"] = 100
        result = nx.earliest_arrival_path(G, "A", "B", 0)
        assert result.trips == [Trip(0, 5, None)]

    def test_transfer_time_not_applied_at_target(self):
        G = nx.DiGraph()
        G.add_edge("A", "B", trips=[(0, 5)])
        G.nodes["B"]["transfer_time"] = 100
        result = nx.earliest_arrival_path(G, "A", "B", 0)
        assert result.arrival_time == 5

    def test_unreachable_after_last_service(self):
        G = make_first_service_graph()
        with pytest.raises(nx.NetworkXNoPath):
            nx.earliest_arrival_path(G, "A", "B", 11)

    def test_unreachable_destination(self):
        G = nx.DiGraph()
        G.add_edge("A", "B", trips=[(0, 1)])
        G.add_node("C")
        with pytest.raises(nx.NetworkXNoPath):
            nx.earliest_arrival_path(G, "A", "C", 0)

    def test_source_equals_target_needs_no_ride(self):
        G = make_no_wait_mistake_graph()
        result = nx.earliest_arrival_path(G, "A", "A", 7)
        assert result.nodes == ["A"]
        assert result.edges == []
        assert result.trips == []
        assert result.arrival_time == 7

    def test_missing_nodes(self):
        G = make_first_service_graph()
        with pytest.raises(nx.NodeNotFound):
            nx.earliest_arrival_path(G, "Z", "B", 0)
        with pytest.raises(nx.NodeNotFound):
            nx.earliest_arrival_path(G, "A", "Z", 0)

    def test_undirected_graph_rejected(self):
        G = nx.Graph()
        G.add_edge("A", "B", trips=[(0, 1)])
        with pytest.raises(nx.NetworkXNotImplemented):
            nx.earliest_arrival_path(G, "A", "B", 0)
        with pytest.raises(nx.NetworkXNotImplemented):
            nx.build_timetable(G)

    def test_datetime_times(self):
        G = nx.DiGraph()
        t0 = datetime(2024, 1, 1, 8, 0)
        G.add_edge(
            "A",
            "B",
            trips=[
                (t0, t0 + timedelta(hours=3)),
                (t0 + timedelta(hours=2), t0 + timedelta(hours=2, minutes=30)),
            ],
        )
        result = nx.earliest_arrival_path(G, "A", "B", t0)
        assert result.arrival_time == t0 + timedelta(hours=2, minutes=30)
        assert result.trips[0].departure_time == t0 + timedelta(hours=2)

    def test_trip_ids_passthrough(self):
        G = nx.MultiDiGraph()
        G.add_edge("A", "B", key="line", trips=[(0, 10, "S1"), (2, 6, "S2")])
        result = nx.earliest_arrival_path(G, "A", "B", 0)
        assert result.trips == [Trip(2, 6, "S2")]

    def test_reused_preprocessed_timetable(self):
        G = make_no_wait_mistake_graph()
        timetable = nx.build_timetable(G)
        assert isinstance(timetable, Timetable)
        assert nx.earliest_arrival_path(timetable, "A", "C", 0).arrival_time == 3
        assert nx.earliest_arrival_path(timetable, "A", "C", 10).arrival_time == 14
        assert nx.earliest_arrival_path(G, "A", "C", 0).arrival_time == 3

    def test_custom_attribute_names(self):
        G = nx.MultiDiGraph()
        G.add_edge(0, 1, key="k", services=[(0, 5, "X"), (4, 3, "Y")])
        G.nodes[1]["wait"] = 2
        G.add_edge(1, 2, key="k", services=[(5, 8, "Z"), (6, 7, "W")])
        result = nx.earliest_arrival_path(
            G, 0, 2, 0, trips="services", transfer_time="wait"
        )
        assert result.trips == [Trip(4, 3, "Y"), Trip(6, 7, "W")]
        assert result.arrival_time == 7

    def test_edges_without_or_empty_schedules_ignored(self):
        G = nx.MultiDiGraph()
        G.add_edge("A", "B", key="walk")
        G.add_edge("A", "B", key="bus", trips=[])
        G.add_edge("A", "B", key="night", trips=[(0, 4)])
        result = nx.earliest_arrival_path(G, "A", "B", 0)
        assert result.edges == [("A", "B", "night")]

    def test_malformed_service_raises(self):
        G = nx.DiGraph()
        G.add_edge("A", "B", trips=[42])
        with pytest.raises(nx.NetworkXError):
            nx.earliest_arrival_path(G, "A", "B", 0)

    def test_fifo_wait_function_is_non_decreasing(self):
        from networkx.algorithms.timetable import _EdgeSchedule

        schedule = _EdgeSchedule([(0, 20), (5, 18), (7, 9), (10, 15)])
        arrivals = [
            schedule.earliest(t).arrival_time for t in range(11) if schedule.earliest(t)
        ]
        assert arrivals == sorted(arrivals)
        assert arrivals[0] == 9
        assert arrivals[7] == 9
        assert arrivals[8] == 15
        assert schedule.earliest(11) is None

    def test_earliest_departure_boarded_at_exact_ready_time(self):
        G = nx.DiGraph()
        G.add_edge("A", "B", trips=[(5, 9)])
        result = nx.earliest_arrival_path(G, "A", "B", 5)
        assert result.trips == [Trip(5, 9, None)]
