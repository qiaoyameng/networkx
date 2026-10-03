"""Shortest paths constrained by a finite automaton."""

from heapq import heappop, heappush
from inspect import signature
from itertools import count

import networkx as nx

__all__ = ["constrained_shortest_path"]


def _as_state_set(states, name):
    if isinstance(states, (str, bytes)) or not isinstance(
        states, (set, list, tuple, frozenset)
    ):
        states = [states]
    try:
        return set(states)
    except TypeError as exc:
        raise TypeError(f"{name} must contain hashable automaton states") from exc


def _edge_function(G, value, default):
    if callable(value):
        if G.is_multigraph():
            try:
                positional = [
                    parameter
                    for parameter in signature(value).parameters.values()
                    if parameter.kind
                    in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
                ]
                use_three_arguments = len(positional) == 3
            except (TypeError, ValueError):
                use_three_arguments = False

            if use_three_arguments:
                return lambda u, v, key, data: value(u, v, data)
            return lambda u, v, key, data: value(u, v, key, data)
        return lambda u, v, key, data: value(u, v, data)

    return lambda u, v, key, data: data.get(value, default)


def _normalize_transitions(states, transitions):
    if not isinstance(transitions, dict):
        raise TypeError("transitions must be a dictionary keyed by automaton state")

    normalized = {}
    for state, symbol_transitions in transitions.items():
        if state not in states:
            raise ValueError(f"transition source state {state!r} is not in states")
        if not isinstance(symbol_transitions, dict):
            raise TypeError(
                f"transitions for state {state!r} must be a dictionary keyed by symbol"
            )

        normalized_state = {}
        for symbol, destinations in symbol_transitions.items():
            destination_set = set()
            for destination in _as_state_set(destinations, "transition destination"):
                if destination not in states:
                    raise ValueError(
                        f"transition {state!r} --{symbol!r}--> {destination!r} "
                        "references a state not in states"
                    )
                destination_set.add(destination)
            normalized_state[symbol] = destination_set
        normalized[state] = normalized_state

    for state in states:
        normalized.setdefault(state, {})
    return normalized


@nx._dispatchable(preserve_edge_attrs=True)
def constrained_shortest_path(
    G,
    source,
    target,
    states,
    transitions,
    initial_state,
    accepting_states,
    label="label",
    weight="weight",
    epsilon=None,
    max_states=None,
):
    r"""Find the shortest path accepted by a finite automaton.

    The search is performed on pairs ``(node, automaton_state)`` as they are
    reached. The complete Cartesian product is never materialized.

    Parameters
    ----------
    G : NetworkX graph
        A graph with non-negative edge weights.

    source, target : node
        Start and end nodes in ``G``.

    states : iterable
        All states of the finite automaton. A tuple contains multiple states;
        a state that is itself a tuple must be wrapped in a list or set.

    transitions : dict
        Nondeterministic transition table of the form
        ``{state: {symbol: destination_or_destinations}}``. Destinations may be
        a single state or a collection of states. Tuples, lists, sets, and
        frozensets contain multiple states. Transitions keyed by the
        ``epsilon`` symbol do not consume an edge label.

    initial_state : state
        Automaton state before reading any edge label.

    accepting_states : state or iterable of states
        Automaton states for which a walk from ``source`` to ``target`` is
        accepted.

    label : string or callable, optional (default: "label")
        Edge attribute containing the symbol consumed by the automaton. If
        callable, it is called as ``label(u, v, data)`` for simple graphs and
        ``label(u, v, key, data)`` for multigraphs. A three-argument callable
        is also accepted for multigraphs. Missing attributes default to
        ``None``.

    weight : string or callable, optional (default: "weight")
        Edge weight attribute or callable, using the same calling convention as
        ``label``. Missing attributes default to one. A callable may return
        ``None`` to hide an edge.

    epsilon : hashable, optional (default: None)
        Symbol used for automaton epsilon transitions. Since missing edge
        labels also default to ``None``, unlabeled edges are epsilon edges
        unless another ``label`` attribute or symbol is used.

    max_states : int, optional
        Maximum number of distinct ``(node, automaton_state)`` configurations
        explored.

    Returns
    -------
    length : number
        Total weight of the shortest accepted walk.

    nodes : list
        Nodes in the walk. The same original node may occur more than once.

    edges : list
        Edges in the walk. Entries are ``(u, v, key)`` for multigraphs and
        ``(u, v)`` for simple graphs.

    Raises
    ------
    NodeNotFound
        If ``source`` or ``target`` is not in ``G``.

    NetworkXNoPath
        If no accepted walk exists.

    ValueError
        If an automaton reference is invalid, an edge weight is negative, or
        ``max_states`` is not positive.

    NetworkXError
        If exploring the next configuration would exceed ``max_states``.

    Examples
    --------
    >>> G = nx.MultiDiGraph()
    >>> key = G.add_edge("client", "server", label="login", weight=1)
    >>> key = G.add_edge("server", "server", key=key, label="read", weight=2)
    >>> key = G.add_edge("server", "client", key=key, label="logout", weight=1)
    >>> states = {"start", "out", "in", "read"}
    >>> transitions = {
    ...     "start": {"login": "in"},
    ...     "in": {"read": "read"},
    ...     "read": {"read": "read", "logout": "out"},
    ...     "out": {},
    ... }
    >>> length, nodes, edges = nx.constrained_shortest_path(
    ...     G, "client", "client", states, transitions, "start", "out"
    ... )
    >>> length
    4
    >>> nodes
    ['client', 'server', 'server', 'client']
    """
    if source not in G:
        raise nx.NodeNotFound(f"source node {source!r} not found in graph")
    if target not in G:
        raise nx.NodeNotFound(f"target node {target!r} not found in graph")
    if max_states is not None:
        if (
            isinstance(max_states, bool)
            or not isinstance(max_states, int)
            or max_states <= 0
        ):
            raise ValueError("max_states must be a positive integer")

    state_set = _as_state_set(states, "states")
    if not state_set:
        raise ValueError("states must not be empty")
    try:
        hash(epsilon)
    except TypeError as exc:
        raise TypeError("epsilon must be hashable") from exc
    if initial_state not in state_set:
        raise ValueError(f"initial state {initial_state!r} is not in states")

    accepting_set = _as_state_set(accepting_states, "accepting_states")
    if not accepting_set:
        raise ValueError("accepting_states must not be empty")
    for state in accepting_set:
        if state not in state_set:
            raise ValueError(f"accepting state {state!r} is not in states")

    normalized_transitions = _normalize_transitions(state_set, transitions)

    label_of = _edge_function(G, label, None)
    weight_of = _edge_function(G, weight, 1)

    predecessors = {}
    distances = {}
    queue = []
    unique = count()
    explored = 0

    def add_configuration(configuration, distance, predecessor):
        nonlocal explored
        old_distance = distances.get(configuration)
        if old_distance is not None and old_distance <= distance:
            return
        if old_distance is None:
            if max_states is not None and explored >= max_states:
                raise nx.NetworkXError(
                    f"constrained shortest path search exceeded max_states={max_states}"
                )
            explored += 1
        distances[configuration] = distance
        predecessors[configuration] = predecessor
        heappush(queue, (distance, next(unique), configuration))

    add_configuration((source, initial_state), 0, None)

    final_configuration = None
    while queue:
        current_distance, _, current_configuration = heappop(queue)
        if current_distance > distances[current_configuration]:
            continue
        current_node, automaton_state = current_configuration
        if current_node == target and automaton_state in accepting_set:
            final_configuration = current_configuration
            break

        epsilon_destinations = normalized_transitions[automaton_state].get(
            epsilon, ()
        )
        for next_automaton_state in epsilon_destinations:
            add_configuration(
                (current_node, next_automaton_state),
                current_distance,
                (current_configuration, None),
            )

        adjacency = G._adj[current_node]
        if G.is_multigraph():
            outgoing_edges = (
                (next_node, edge_key, data)
                for next_node, edge_data in adjacency.items()
                for edge_key, data in edge_data.items()
            )
        else:
            outgoing_edges = (
                (next_node, None, data) for next_node, data in adjacency.items()
            )

        for next_node, edge_key, data in outgoing_edges:
            edge_label = label_of(current_node, next_node, edge_key, data)
            edge_weight = weight_of(current_node, next_node, edge_key, data)
            if edge_weight is None:
                continue
            if edge_weight < 0:
                raise ValueError(
                    f"negative edge weight {edge_weight!r} on edge "
                    f"({current_node!r}, {next_node!r}, {edge_key!r})"
                )

            if edge_label == epsilon:
                next_automaton_states = {automaton_state}
            else:
                next_automaton_states = normalized_transitions[automaton_state].get(
                    edge_label, ()
                )

            path_edge = (
                (current_node, next_node, edge_key)
                if G.is_multigraph()
                else (current_node, next_node)
            )
            next_distance = current_distance + edge_weight
            for next_automaton_state in next_automaton_states:
                next_configuration = (next_node, next_automaton_state)
                if next_distance < distances.get(next_configuration, float("inf")):
                    add_configuration(
                        next_configuration,
                        next_distance,
                        (current_configuration, path_edge),
                    )

    if final_configuration is None:
        raise nx.NetworkXNoPath(
            f"No accepted path from {source!r} to {target!r}."
        )

    nodes = [target]
    edges = []
    configuration = final_configuration
    while (predecessor := predecessors[configuration]) is not None:
        configuration, edge = predecessor
        if edge is not None:
            edges.append(edge)
            nodes.append(configuration[0])
    nodes.reverse()
    edges.reverse()
    return distances[final_configuration], nodes, edges
