"""Timetable routing on directed graphs with scheduled edge departures.

Edges carry a list of scheduled services (``(departure, arrival)`` pairs or
``(departure, arrival, trip_id)`` triples). A traveller arriving at a node at
time ``t`` may board the earliest feasible service of an outgoing edge, where
"feasible" means a service departing at or after ``t`` (plus any transfer
time at the node). The earliest-arrival problem asks for the path and the
specific services taken that minimise the arrival time at a target.

All time values come from the caller; the algorithms only compare them and
add transfer times. They are agnostic to units, calendars and time zones.
"""

from bisect import bisect_left
from collections import namedtuple
import heapq

import networkx as nx

__all__ = [
    "Timetable",
    "Trip",
    "EarliestArrivalPath",
    "build_timetable",
    "earliest_arrival_path",
]

Trip = namedtuple("Trip", ["departure_time", "arrival_time", "trip_id"])
"""A single scheduled service taken by a traveller.

``departure_time`` and ``arrival_time`` are the caller-provided ordered time
values; ``trip_id`` is the optional third element stored on an edge and is
``None`` when no identifier was given.
"""

EarliestArrivalPath = namedtuple(
    "EarliestArrivalPath", ["nodes", "edges", "trips", "arrival_time"]
)
"""Result of an earliest-arrival query.

``nodes`` is the sequence of nodes from source to target. ``edges`` and
``trips`` are aligned sequences, one entry per traversed edge: ``edges``
holds ``(u, v, key)`` triples (the key is ``None`` for a non-multigraph) and
``trips`` holds the :class:`Trip` actually boarded. ``arrival_time`` is the
time at which the target is reached.
"""


class _EdgeSchedule:
    """Preprocessed services of one edge.

    Stores departure times in sorted order and, aligned with each departure
    index, the suffix-minimum arrival of every service departing at or after
    that time. The suffix front is non-decreasing as the feasible departure
    time grows, i.e. it is a FIFO link function: leaving a node later can
    never produce an earlier arrival on this edge.
    """

    __slots__ = ("departures", "front")

    def __init__(self, raw_trips):
        parsed = []
        for entry in raw_trips:
            try:
                departure_time = entry[0]
                arrival_time = entry[1]
            except (TypeError, IndexError, KeyError) as err:
                raise nx.NetworkXError(
                    "Each scheduled service must be a (departure, arrival) "
                    "pair or a (departure, arrival, trip_id) triple."
                ) from err
            trip_id = entry[2] if len(entry) > 2 else None
            parsed.append(Trip(departure_time, arrival_time, trip_id))
        parsed.sort(key=lambda trip: trip.departure_time)

        departures = [trip.departure_time for trip in parsed]
        front = [None] * len(parsed)
        best = None
        for index in range(len(parsed) - 1, -1, -1):
            trip = parsed[index]
            if (
                best is None
                or trip.arrival_time < best.arrival_time
                or (
                    trip.arrival_time == best.arrival_time
                    and trip.departure_time > best.departure_time
                )
            ):
                best = trip
            front[index] = best
        self.departures = departures
        self.front = front

    def earliest(self, ready_time):
        """Return the best boardable trip at or after ``ready_time``.

        The first service departing at or after ``ready_time`` is found with
        binary search, then the suffix-minimum front is used so a later
        departure with an earlier arrival is preferred. Returns ``None`` when
        no later service exists.
        """
        index = bisect_left(self.departures, ready_time)
        if index == len(self.departures):
            return None
        return self.front[index]


class Timetable:
    """Preprocessed schedules used to answer earliest-arrival queries.

    Instances are created with :func:`build_timetable` and can be reused for
    any number of queries against the same graph. The instance keeps a
    reference to the graph, which must not be modified afterwards.
    """

    __slots__ = ("graph", "schedules", "adjacency", "trips", "transfer_time")

    def __init__(self, graph, schedules, adjacency, trips, transfer_time):
        self.graph = graph
        self.schedules = schedules
        self.adjacency = adjacency
        self.trips = trips
        self.transfer_time = transfer_time


def build_timetable(G, *, trips="trips", transfer_time="transfer_time"):
    """Preprocess the scheduled services of every edge.

    Parameters
    ----------
    G : NetworkX directed graph
        A ``DiGraph`` or ``MultiDiGraph``. Parallel edges are distinguished
        by their edge key and each keeps its own schedule.
    trips : string (default: ``"trips"``)
        Name of the edge attribute holding the services. Every service is a
        ``(departure_time, arrival_time)`` pair or a
        ``(departure_time, arrival_time, trip_id)`` triple. Edges without the
        attribute (or with an empty schedule) can never be boarded.
    transfer_time : string (default: ``"transfer_time"``)
        Name of the optional node attribute holding the minimum time that
        must pass after arriving at a node before another service may be
        boarded there. Nodes without the attribute have zero transfer time.
        The attribute is not applied at the query source (the journey has not
        begun) or at the target (no onward service is boarded).

    Returns
    -------
    :class:`Timetable`
        A reusable preprocessed timetable.

    Raises
    ------
    NetworkXNotImplemented
        If ``G`` is undirected.
    NetworkXError
        If a service is not a pair or triple of time values.
    """
    if not G.is_directed():
        raise nx.NetworkXNotImplemented("Timetable routing requires a directed graph.")

    schedules = {}
    adjacency = {node: [] for node in G.nodes}
    multigraph = G.is_multigraph()
    if multigraph:
        edges = G.edges(keys=True, data=True)
    else:
        edges = ((u, v, None, data) for u, v, data in G.edges(data=True))
    for u, v, key, data in edges:
        if trips not in data:
            continue
        schedule = _EdgeSchedule(list(data[trips]))
        if not schedule.departures:
            continue
        schedules[(u, v, key)] = schedule
        adjacency[u].append((v, key, schedule))

    return Timetable(G, schedules, adjacency, trips, transfer_time)


def earliest_arrival_path(
    G, source, target, departure_time, *, trips="trips", transfer_time="transfer_time"
):
    r"""Find the earliest-arriving path and the services taken.

    Uses a label-setting search over the preprocessed FIFO edge schedules,
    analogous to Dijkstra's algorithm: the label of a node is the earliest
    time it can be reached, and settling nodes in non-decreasing label order
    is correct because every edge link function is non-decreasing (a later
    feasible departure cannot yield an earlier arrival).

    Parameters
    ----------
    G : NetworkX directed graph or :class:`Timetable`
        A ``DiGraph``/``MultiDiGraph`` with scheduled edges, or a
        :class:`Timetable` returned by :func:`build_timetable` for repeated
        queries.
    source : node
        Node at which the journey starts at ``departure_time``.
    target : node
        Node to reach.
    departure_time : ordered value
        Time from which services may be boarded at ``source``.
    trips, transfer_time : string
        Edge and node attribute names, as in :func:`build_timetable`; ignored
        when ``G`` is already a :class:`Timetable`.

    Returns
    -------
    :class:`EarliestArrivalPath`
        The path as nodes, the traversed ``(u, v, key)`` edges, the aligned
        :class:`Trip` objects boarded on each edge, and the arrival time.

    Raises
    ------
    NodeNotFound
        If ``source`` or ``target`` is not in the graph.
    NetworkXNoPath
        If no sequence of services connects ``source`` to ``target`` at or
        after ``departure_time``.
    NetworkXNotImplemented
        If ``G`` is undirected.

    Examples
    --------
    On the edge ``A -> B`` the service leaving at 5 arrives before the one
    leaving at 0, so the later service is taken:

    >>> G = nx.MultiDiGraph()
    >>> G.add_edges_from(
    ...     [
    ...         ("A", "B", "bus", {"trips": [(0, 10), (5, 7)]}),
    ...         ("B", "C", "bus", {"trips": [(8, 20), (10, 11)]}),
    ...     ]
    ... )
    ['bus', 'bus']
    >>> result = nx.earliest_arrival_path(G, "A", "C", 0)
    >>> result.arrival_time
    11
    >>> result.edges
    [('A', 'B', 'bus'), ('B', 'C', 'bus')]
    >>> [trip.departure_time for trip in result.trips]
    [5, 10]

    Preprocessing once and reusing it for many queries avoids rebuilding the
    FIFO schedules:

    >>> timetable = nx.build_timetable(G)
    >>> nx.earliest_arrival_path(timetable, "A", "C", 0).arrival_time
    11

    Notes
    -----
    Time values are treated as an ordered domain supplied by the caller;
    only comparisons and addition of transfer times are used. No time zone,
    calendar or unit conversion is performed.
    """
    timetable = (
        G
        if isinstance(G, Timetable)
        else build_timetable(G, trips=trips, transfer_time=transfer_time)
    )
    graph = timetable.graph
    if source not in graph:
        raise nx.NodeNotFound(f"Source {source} is not in the graph.")
    if target not in graph:
        raise nx.NodeNotFound(f"Target {target} is not in the graph.")

    if source == target:
        return EarliestArrivalPath([source], [], [], departure_time)

    earliest = {source: departure_time}
    predecessor = {}
    heap = [(departure_time, source)]
    while heap:
        arrival, node = heapq.heappop(heap)
        if arrival != earliest[node]:
            continue
        if node == target:
            break

        node_attr = graph.nodes[node]
        transfer_attr = timetable.transfer_time
        if node != source and transfer_attr in node_attr:
            ready_time = arrival + node_attr[transfer_attr]
        else:
            ready_time = arrival

        for neighbour, key, schedule in timetable.adjacency[node]:
            ride = schedule.earliest(ready_time)
            if ride is None:
                continue
            if neighbour not in earliest or ride.arrival_time < earliest[neighbour]:
                earliest[neighbour] = ride.arrival_time
                predecessor[neighbour] = (node, key, ride)
                heapq.heappush(heap, (ride.arrival_time, neighbour))

    if target not in earliest:
        raise nx.NetworkXNoPath(
            f"No scheduled path from {source} to {target} at or after "
            f"{departure_time}."
        )

    nodes = [target]
    edges = []
    rides = []
    node = target
    while node != source:
        previous, key, ride = predecessor[node]
        edges.append((previous, node, key))
        rides.append(ride)
        nodes.append(previous)
        node = previous
    nodes.reverse()
    edges.reverse()
    rides.reverse()
    return EarliestArrivalPath(nodes, edges, rides, earliest[target])
