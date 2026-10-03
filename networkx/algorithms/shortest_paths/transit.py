"""Earliest-arrival routing on scheduled-transit (timetable) networks.

Each edge carries a list of services.  A service is an ordered pair
``(departure_time, arrival_time)`` of values drawn from any totally ordered
type chosen by the caller (integers, floats, :class:`datetime.datetime`, ...).
Arriving at a node at time ``t`` lets a traveller wait for any service with
``departure_time >= t`` (plus an optional minimum transfer time).

The timetable of every edge is reduced once, during preprocessing, to its
non-dominated service frontier.  A service is dominated when boarding some
later-or-equal departure reaches the destination no later, because waiting is
always allowed.  On the resulting frontier both departure and arrival times
are strictly increasing, so the first service departing at or after the ready
time already minimises arrival.  The link arrival function is therefore
non-decreasing in the ready time -- it satisfies the FIFO property -- and a
Dijkstra-style label-setting search is exact.

Only comparison and addition are performed on time values; the algorithm never
makes assumptions about their units or time zones.
"""

from bisect import bisect_left
from dataclasses import dataclass, field
from heapq import heappop, heappush
from itertools import count

import networkx as nx
from networkx.utils import not_implemented_for

__all__ = ["TransitLeg", "TransitRoute", "earliest_arrival", "TransitTimetable"]


@dataclass(frozen=True)
class TransitLeg:
    """One edge of a :class:`TransitRoute`.

    Attributes
    ----------
    source, target : node
        Endpoints of the edge used by this leg.
    key : hashable or None
        Edge key; ``None`` for simple (non-multi) directed graphs.
    departure, arrival : time value
        Departure and arrival times of the service actually boarded.
    """

    source: object
    target: object
    key: object
    departure: object
    arrival: object


@dataclass(frozen=True)
class TransitRoute:
    """An earliest-arrival itinerary.

    Attributes
    ----------
    path : list of nodes
        Nodes visited, including source and target.
    legs : list of TransitLeg
        One entry per edge of the path, recording the service boarded.
        Empty when source equals target.
    arrival_time : time value
        Arrival time at the target; the supplied ``departure_time`` when
        source equals target.
    """

    path: list
    legs: list = field(default_factory=list)
    arrival_time: object = None

    def __iter__(self):
        # Allow tuple-style unpacking: route, legs, arrival_time.
        yield self.path
        yield self.legs
        yield self.arrival_time


def _iter_edges(G):
    """Yield ``(u, v, key, data)`` for every edge with a stable ``key``."""
    if G.is_multigraph():
        for u, nbrs in G.adj.items():
            for v, keydict in nbrs.items():
                for key, data in keydict.items():
                    yield u, v, key, data
    else:
        for u, nbrs in G.adj.items():
            for v, data in nbrs.items():
                yield u, v, None, data


def _coerce_services(raw):
    """Validate one edge's service list and return ``(departures, arrivals)``."""
    if raw is None:
        return None
    try:
        services = list(raw)
    except TypeError as exc:
        raise nx.NetworkXError(
            "edge timetable must be an iterable of (departure, arrival) pairs"
        ) from exc
    if not services:
        return None
    departures = []
    arrivals = []
    for service in services:
        try:
            departure, arrival = service
        except (TypeError, ValueError) as exc:
            raise nx.NetworkXError(
                "every service must be a (departure, arrival) pair"
            ) from exc
        if arrival < departure:
            raise nx.NetworkXError(
                f"service {(departure, arrival)} arrives before it departs"
            )
        departures.append(departure)
        arrivals.append(arrival)
    return departures, arrivals


def _build_frontier(departures, arrivals):
    """Return the Pareto frontier ``(f_departures, f_arrivals)``.

    Services are sorted by ascending departure (ties broken by ascending
    arrival) and scanned from latest to earliest.  A service is retained only
    when its arrival is strictly earlier than the best (smallest) arrival of
    every service departing at least as late: boarding an earlier service that
    arrives no earlier than a later one can never be useful, because waiting
    for the later service is always allowed.  Reversed back into ascending
    order, both departure and arrival are strictly increasing, so the first
    service departing at or after a ready time already minimises arrival.
    """
    # Best (smallest) arrival available at each departure time; services with
    # the same departure are interchangeable, so only the fastest survives.
    best_at_departure = {}
    for departure, arrival in zip(departures, arrivals):
        if departure not in best_at_departure or arrival < best_at_departure[departure]:
            best_at_departure[departure] = arrival

    # Scan departure times from latest to earliest and keep a service only
    # when waiting for everything departing later cannot arrive earlier.
    chosen = []
    best = None
    for departure in sorted(best_at_departure, reverse=True):
        arrival = best_at_departure[departure]
        if best is not None and arrival >= best:
            continue
        best = arrival
        chosen.append((departure, arrival))
    chosen.reverse()
    return [d for d, _ in chosen], [a for _, a in chosen]


class TransitTimetable:
    """Preprocessed scheduled-transit timetable.

    Building the object once preprocesses every edge; it can then answer many
    earliest-arrival queries on the same graph without repeating the work.

    Parameters
    ----------
    G : DiGraph or MultiDiGraph
        Directed graph.  Parallel edges are supported and distinguished by
        their edge keys.
    trips : string (default: ``"trips"``)
        Name of the edge attribute holding an iterable of
        ``(departure_time, arrival_time)`` pairs.  Edges without a non-empty
        timetable are ignored.
    transfer_time : string (default: ``"transfer_time"``)
        Name of the node attribute holding a minimum waiting/transfer time at
        that node.  Missing attributes default to ``zero``.  The value type
        must be addable to the time values, e.g. ``0`` for numeric times.
    zero : number (default: 0)
        Zero value of the time type, used as the default transfer time.

    Raises
    ------
    NetworkXNotImplemented
        If ``G`` is undirected.
    NetworkXError
        If a service is malformed or arrives before it departs.
    """

    def __init__(self, G, *, trips="trips", transfer_time="transfer_time", zero=0):
        if not G.is_directed():
            raise nx.NetworkXNotImplemented(
                "earliest-arrival routing requires a directed graph"
            )
        self.G = G
        self._trips_attr = trips
        self._transfer_attr = transfer_time
        self._zero = zero
        # frontier[(u, v, key)] = (sorted_departures, sorted_arrivals)
        self._frontier = {}
        for u, v, key, data in _iter_edges(G):
            raw = data.get(trips)
            services = _coerce_services(raw)
            if services is None:
                continue
            self._frontier[(u, v, key)] = _build_frontier(*services)
        self._transfer = {
            node: G.nodes[node].get(transfer_time, zero) for node in G.nodes
        }

    def wait_at(self, node):
        """Minimum transfer/ waiting time configured at ``node``."""
        return self._transfer.get(node, self._zero)

    def board(self, u, v, key, ready_time):
        """Return the ``TransitLeg`` available at the earliest at ``ready_time``.

        ``None`` when the edge has no service departing at or after
        ``ready_time``.
        """
        frontier = self._frontier.get((u, v, key))
        if frontier is None:
            return None
        departures, arrivals = frontier
        idx = bisect_left(departures, ready_time)
        if idx == len(departures):
            return None
        return TransitLeg(u, v, key, departures[idx], arrivals[idx])

    def route_to(self, source, target, departure_time, *, wait_at_source=False):
        """Run an earliest-arrival query against this preprocessed timetable."""
        return earliest_arrival(
            self.G,
            source,
            target,
            departure_time,
            trips=self._trips_attr,
            transfer_time=self._transfer_attr,
            timetable=self,
            wait_at_source=wait_at_source,
        )


def _validate_node(G, node, name):
    if node not in G:
        raise nx.NodeNotFound(f"{name} {node} is not in G")


def _zero_of(value):
    """Additive zero compatible with ``value``'s time type."""
    try:
        return value - value
    except TypeError:
        return 0


@not_implemented_for("undirected")
@nx._dispatchable(edge_attrs={"trips": None}, node_attrs={"transfer_time": 0})
def earliest_arrival(
    G,
    source,
    target,
    departure_time,
    *,
    trips="trips",
    transfer_time="transfer_time",
    wait_at_source=False,
    timetable=None,
):
    r"""Find the earliest-arriving itinerary in a scheduled-transit network.

    The traveller starts at ``source`` at ``departure_time`` and may wait at
    nodes for later services.  A :class:`TransitRoute` is returned describing
    the node path and, for every leg, the edge key and the actual service
    boarded.

    Parameters
    ----------
    G : DiGraph or MultiDiGraph
        Directed graph whose edges carry a timetable.
    source : node
        Starting node.
    target : node
        Destination node.
    departure_time : time value
        Time at which the traveller is ready at ``source``.  Any ordered,
        addable type is accepted; the algorithm performs no time-zone or unit
        conversion.
    trips : string (default: ``"trips"``)
        Edge attribute holding an iterable of ``(departure, arrival)`` pairs.
        Edges without a non-empty timetable are ignored.
    transfer_time : string (default: ``"transfer_time"``)
        Node attribute giving a minimum waiting time between arriving at the
        node and boarding the next service.  Nodes without the attribute have
        zero transfer time.
    wait_at_source : bool (default: False)
        If True, the ``source`` node's transfer time is also applied before the
        first boarding.  By default boarding at the source only requires the
        service to depart at or after ``departure_time``.
    timetable : TransitTimetable, optional
        A preprocessed :class:`TransitTimetable` for ``G``; built on the fly
        when omitted.  Reuse it for repeated queries on the same graph.

    Returns
    -------
    TransitRoute
        The earliest-arriving itinerary.  When ``source == target`` the path is
        ``[source]`` with no legs and ``arrival_time == departure_time``.

    Raises
    ------
    NodeNotFound
        If ``source`` or ``target`` is not in the graph.
    NetworkXNoPath
        If no sequence of feasible services reaches ``target``.
    NetworkXNotImplemented
        If ``G`` is undirected.
    NetworkXError
        If a timetable entry is malformed.

    Examples
    --------
    >>> G = nx.MultiDiGraph()
    >>> key = G.add_edge("A", "B", trips=[(0, 10), (7, 8)])
    >>> _ = G.add_edge("B", "C", trips=[(9, 12), (11, 15)])
    >>> G.nodes["B"]["transfer_time"] = 1
    >>> route = nx.earliest_arrival(G, "A", "C", 0)
    >>> route.path
    ['A', 'B', 'C']
    >>> [(leg.key, leg.departure, leg.arrival) for leg in route.legs]
    [(0, 7, 8), (0, 9, 12)]
    >>> route.arrival_time
    12

    Notes
    -----
    Each edge timetable is preprocessed into a non-dominated service frontier.
    A service is discarded when some *later-or-equal* departure reaches no
    later, since waiting for that later service is always allowed.  This
    matters when a later departure overtakes an earlier one: the overtaking
    service is kept, and boarding the first feasible departure blindly would be
    wrong.  On the frontier the first service departing at or after the ready
    time already has the earliest arrival, giving a FIFO (non decreasing) link
    arrival function -- exactly what a label-setting Dijkstra search needs.

    Parallel edges of a ``MultiDiGraph`` are kept distinct by their keys, so a
    slower and a faster connection between the same node pair can coexist and
    the faster one is selected per boarding.
    """
    _validate_node(G, source, "source")
    _validate_node(G, target, "target")

    if timetable is None:
        zero = _zero_of(departure_time)
        timetable = TransitTimetable(
            G, trips=trips, transfer_time=transfer_time, zero=zero
        )

    if source == target:
        return TransitRoute([source], [], departure_time)

    dist = {source: departure_time}
    parent = {}  # node -> (parent_node, TransitLeg)
    settled = set()
    counter = count()
    heap = [(departure_time, next(counter), source)]

    while heap:
        arrival_at_u, _, u = heappop(heap)
        if u in settled:
            continue
        if u == target:
            path = [target]
            legs = []
            node = target
            while node != source:
                prev, leg = parent[node]
                legs.append(leg)
                path.append(prev)
                node = prev
            path.reverse()
            legs.reverse()
            return TransitRoute(path, legs, arrival_at_u)
        settled.add(u)

        ready_time = arrival_at_u
        if u != source or wait_at_source:
            ready_time = ready_time + timetable.wait_at(u)

        for v, keydict in G.adj[u].items():
            keys = keydict if G.is_multigraph() else {None: None}
            for key in keys:
                leg = timetable.board(u, v, key, ready_time)
                if leg is None:
                    continue
                candidate = leg.arrival
                if v not in settled and (v not in dist or candidate < dist[v]):
                    dist[v] = candidate
                    parent[v] = (u, leg)
                    heappush(heap, (candidate, next(counter), v))

    raise nx.NetworkXNoPath(f"No scheduled-transit path to {target}")
