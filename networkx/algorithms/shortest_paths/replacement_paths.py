"""Replacement paths for a fixed shortest s-t path.

Given a shortest path between a source ``s`` and a target ``t`` in an
undirected, non-negatively weighted graph, the algorithms here compute,
for every edge of that path assumed to fail, the best detour: the shortest
``s``-``t`` path that does not use the failed edge.  This is the
single-edge replacement paths (edge-bypass) problem.

Only undirected graphs are supported.  The correctness argument relies on
the symmetry of undirected edge weights and the cut structure of an
undirected path; it does not extend to directed graphs.
"""

from heapq import heappop, heappush
from itertools import count
from math import inf

import networkx as nx

__all__ = [
    "replacement_paths",
    "replacement_path_lengths",
]


def _make_weight(G, weight):
    """Build a ``(u, v, edges_dict) -> cost`` function.

    Mirrors the weight convention of the Dijkstra functions: a callable
    receives the full adjacency entry (the single edge dict for ``Graph``,
    the ``{key: data}`` mapping for ``MultiGraph``), and a string attribute
    defaults to ``1`` and, for multigraphs, is minimised over parallel
    edges.
    """
    if callable(weight):
        pair_cost = weight
        key_cost = lambda u, v, edges, key: weight(u, v, edges)
        return pair_cost, key_cost
    if G.is_multigraph():
        pair_cost = lambda u, v, d: min(a.get(weight, 1) for a in d.values())
        key_cost = lambda u, v, edges, key: edges[key].get(weight, 1)
        return pair_cost, key_cost
    pair_cost = lambda u, v, d: d.get(weight, 1)
    key_cost = lambda u, v, edges, key: edges.get(weight, 1)
    return pair_cost, key_cost


def _path_edges(G, path, path_keys, pair_cost, key_cost):
    """Resolve the concrete edge key and weight of every hop of ``path``."""
    keys = []
    weights = []
    for i, (u, v) in enumerate(zip(path[:-1], path[1:])):
        if not G.has_edge(u, v):
            rep = f"({u}, {v})" if not G.is_multigraph() else f"({u}, {v}, ...)"
            raise nx.NetworkXError(f"Edge {rep} of the given path is not in G.")
        edges = G[u][v]
        if G.is_multigraph():
            if path_keys is None:
                key = min(edges, key=lambda k: key_cost(u, v, edges, k))
            else:
                key = path_keys[i]
                if key not in edges:
                    raise nx.NetworkXError(f"Edge ({u}, {v}, {key}) is not in G.")
            keys.append(key)
            cost = key_cost(u, v, edges, key)
        else:
            if path_keys is not None:
                raise nx.NetworkXError(
                    "path_keys is only supported for MultiGraph instances."
                )
            keys.append(None)
            cost = pair_cost(u, v, edges)
        if cost is None:
            raise nx.NetworkXError("The given path uses a hidden (None-weight) edge.")
        if cost < 0:
            raise nx.NetworkXError("Replacement paths require non-negative weights.")
        weights.append(cost)
    return keys, weights


def _seeded_dijkstra(G, path, hop_weights, side, edge_costs):
    """Dijkstra from the source/target, forced to keep ``path`` in the tree.

    Since ``path`` is a shortest path, seeding the fringe with every path
    node at its prefix (``side="s"``) or suffix (``side="t"``) distance and
    never relaxing into path nodes is equivalent to Dijkstra with ties
    broken in favour of path edges.

    Returns
    -------
    dist : dict
        Distance from the root to every reachable node.
    parent : dict
        Parent of every reachable node in the shortest-path tree.
    label : dict
        Path index at which the tree path of a node meets ``path``: the
        index of the last path node before leaving ``path`` from ``s``
        (``side="s"``), or the first path node when heading towards ``t``
        (``side="t"``).
    """
    n = len(path)
    index = dict.fromkeys(path)
    for i, v in enumerate(path):
        index[v] = i

    totals = [0] * n
    if side == "s":
        running = 0
        for i, w in enumerate(hop_weights):
            running += w
            totals[i + 1] = running
    else:
        running = 0
        for i in range(n - 1, 0, -1):
            running += hop_weights[i - 1]
            totals[i - 1] = running

    dist = {}
    seen = {}
    parent = {}
    label = {}
    fringe = []
    c = count()
    for i, v in enumerate(path):
        d = totals[i]
        dist[v] = seen[v] = d
        label[v] = i
        if side == "s":
            parent[v] = path[i - 1] if i > 0 else None
        else:
            parent[v] = path[i + 1] if i < n - 1 else None
        tie = i if side == "s" else n - 1 - i
        heappush(fringe, (d, tie, v))

    while fringe:
        d, _, v = heappop(fringe)
        if seen.get(v) != d:
            continue  # stale heap entry
        if v not in dist:
            dist[v] = d
        for u, cost in edge_costs(v):
            if u in index:
                # Path nodes are seeded at their exact shortest distance;
                # relaxing into one would detach the tree from the path.
                continue
            if cost is None:
                continue
            nd = d + cost
            if nd < 0:
                raise nx.NetworkXError(
                    "Replacement paths require non-negative weights."
                )
            if u not in seen or nd < seen[u]:
                seen[u] = nd
                parent[u] = v
                label[u] = label[v]
                heappush(fringe, (nd, next(c), u))
            elif u in dist and nd < dist[u]:
                raise nx.NetworkXError(
                    "Contradictory paths found: negative weights?"
                )
    return dist, parent, label


def _tree_path(parent, node, root, toward_root):
    """Walk the tree from ``node`` to ``root``.

    The source-tree parents point away from the root (children -> parent
    walks towards the source); the target-tree parents point towards the
    root, so the walk has to be reversed there.
    """
    seq = [node]
    while node != root:
        node = parent[node]
        seq.append(node)
    # ``seq`` walks node -> root (towards the root).  For the s-tree the
    # desired head is source -> node, i.e. root -> node, so reverse there.
    return seq[::-1] if not toward_root else seq


def _splice(head, tail):
    """Join an s-side and a t-side tree walk into a loop-free path.

    The two walks may share a prefix of path nodes (or, with zero-weight
    edges, other nodes) when the bypass is degenerate.  The last shared
    node is the meeting point; everything between duplicate occurrences is
    a loop and is removed.
    """
    head_index = {v: i for i, v in enumerate(head)}
    cut = None
    for i, v in enumerate(tail):
        if v in head_index:
            cut = (head_index[v], i)
    if cut is not None:
        hi, ti = cut
        route = head[: hi + 1] + tail[ti + 1:]
    else:
        route = head + tail
    # With arbitrary zero-weight ties the two tree walks can share more
    # than one node; remove any residual loop from the concatenation.
    out = []
    for v in route:
        if v in out:
            out = out[: out.index(v) + 1]
        else:
            out.append(v)
    return out


def _realizes(G, failed, route, length, weight):
    """Check that ``route`` avoids the failed edge and realises ``length``.

    Length is measured with the NetworkX Dijkstra weight convention
    (minimum over parallel edges in a MultiGraph).  A node walk cannot pin
    a specific parallel key, so a failed key only forbids the walk when it
    is the sole edge between its endpoints.
    """
    u, v = failed[0], failed[1]
    key = failed[2] if len(failed) == 3 else None
    if callable(weight):
        wf = weight
    elif G.is_multigraph():
        attr = weight
        wf = lambda a, b, d: min(e.get(attr, 1) for e in d.values())
    else:
        attr = weight
        wf = lambda a, b, d: d.get(attr, 1)
    total = 0
    for a, b in zip(route[:-1], route[1:]):
        if not G.has_edge(a, b):
            return False
        if key is not None:
            if {a, b} == {u, v} and len(G[a][b]) == 1 and key in G[a][b]:
                return False
        elif {a, b} == {u, v}:
            return False
        cost = wf(a, b, G[a][b])
        if cost is None:
            return False
        total += cost
    return total == length


def _replacement_paths_core(G, source, target, path, path_keys, weight, compute_path):
    if G.is_directed():
        raise nx.NetworkXNotImplemented(
            "replacement_paths is not implemented for directed graphs; "
            "the undirected cut argument does not apply to directed graphs."
        )
    if source not in G:
        raise nx.NodeNotFound(f"source {source} is not in G")
    if target not in G:
        raise nx.NodeNotFound(f"target {target} is not in G")
    if source == target:
        return {}, []

    pair_cost, key_cost = _make_weight(G, weight)

    if path is None:
        path = nx.dijkstra_path(G, source, target, weight=weight)
        path_keys = None
    else:
        path = list(path)
        if not path or path[0] != source or path[-1] != target:
            raise nx.NetworkXError("path must start at source and end at target.")
        if path_keys is not None and len(path_keys) != len(path) - 1:
            raise nx.NetworkXError(
                "path_keys must have one key for every edge of path."
            )
    keys, hop_weights = _path_edges(G, path, path_keys, pair_cost, key_cost)
    shortest_length = nx.dijkstra_path_length(G, source, target, weight=weight)
    if sum(hop_weights) != shortest_length:
        raise nx.NetworkXError("The given path is not a shortest s-t path.")

    # Shortest-path trees follow the NetworkX Dijkstra convention: between
    # an endpoint pair only the cheapest parallel edge is usable.  The
    # candidate enumeration below treats every parallel key separately.
    def tree_edges(v):
        for u, edges in G._adj[v].items():
            yield u, pair_cost(v, u, edges)

    def key_edges(v):
        if G.is_multigraph():
            for u, edges in G._adj[v].items():
                for key in edges:
                    yield u, key, key_cost(v, u, edges, key)
        else:
            for u, edges in G._adj[v].items():
                yield u, None, pair_cost(v, u, edges)

    n = len(path)
    removed_edges = [
        (path[i - 1], path[i]) if not G.is_multigraph()
        else (path[i - 1], path[i], keys[i - 1])
        for i in range(1, n)
    ]

    ds, ps, S = _seeded_dijkstra(G, path, hop_weights, "s", tree_edges)
    dt, pt, T = _seeded_dijkstra(G, path, hop_weights, "t", tree_edges)

    # Bypass candidate edges.  A non-path edge {x, y} covers every failing
    # path edge index l with S(x) < l <= T(y); the induced detour has
    # length ds[x] + w + dt[y].  For each l the best such candidate is the
    # answer (Malik, Mittal and Gupta, 1989).  Both orientations are
    # considered, and an undirected edge is visited once.
    path_edge_ids = {
        (frozenset((u, v)), k)
        for (u, v), k in zip(zip(path[:-1], path[1:]), keys)
    }
    events = []
    seen_edge = set()
    multigraph = G.is_multigraph()
    # Iterate every concrete edge exactly once by canonicalising the
    # endpoint pair; both traversal orientations are tested below.
    for x in G:
        for y, edge_key, w in key_edges(x):
            pair = frozenset((x, y))
            edge_id = (pair, edge_key)
            if edge_id in seen_edge:
                continue
            seen_edge.add(edge_id)
            if edge_id in path_edge_ids:
                continue
            if w is None or w < 0:
                continue
            for orient, (u, v) in enumerate(((x, y), (y, x)), start=1):
                if u in S and v in T and S[u] < T[v]:
                    events.append((S[u] + 1, T[v], ds[u] + w + dt[v],
                                   u, v, edge_key, w, orient))

    events.sort(key=lambda e: (e[0], e[2]))

    results = {}
    event_heap = []
    ei = 0
    for l in range(1, n):
        while ei < len(events) and events[ei][0] <= l:
            _start, end, val, u, v, key, w, orient = events[ei]
            heappush(event_heap, (val, end, u, v, key, w, orient))
            ei += 1
        while event_heap and event_heap[0][1] < l:
            heappop(event_heap)
        failed = removed_edges[l - 1]
        if not event_heap:
            if compute_path:
                results[failed] = (inf, None)
            else:
                results[failed] = inf
            continue
        val, _end, u, v, key, _w, orient = event_heap[0]
        if not compute_path:
            results[failed] = val
            continue
        # Fast reconstruction from the two shared shortest-path trees.  The
        # bypass edge is crossed in the stored orientation.
        first, second = (u, v) if orient == 1 else (v, u)
        head = _tree_path(ps, first, source, toward_root=False)
        tail = _tree_path(pt, second, target, toward_root=True)
        route = _splice(head, tail)
        # Zero-weight ties or the "cheapest parallel key" convention can
        # make the tree walk revisit the failed edge or realise a different
        # length; verify and, only then, fall back to one exact search on G
        # minus the concrete failed edge.
        if not _realizes(G, failed, route, val, weight):
            H = G.copy()
            if multigraph:
                H.remove_edge(failed[0], failed[1], failed[2])
            else:
                H.remove_edge(failed[0], failed[1])
            route = nx.dijkstra_path(H, source, target, weight=weight)
        results[failed] = (val, route)

    return results, removed_edges


@nx._dispatchable(edge_attrs="weight")
def replacement_paths(G, source, target, path=None, *, path_keys=None,
                      weight="weight"):
    """Best detour for every edge failure on a chosen shortest s-t path.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph with non-negative edge weights.  Both
        ``Graph`` and ``MultiGraph`` are supported.
    source, target : node
        Endpoints of the reference shortest path.
    path : list of nodes, optional
        A shortest path from ``source`` to ``target``.  If not given, one is
        computed with Dijkstra's algorithm.  In a ``MultiGraph`` a node list
        uses the minimum-weight parallel edge on each hop; supply
        ``path_keys`` to choose specific edges.
    path_keys : list, optional
        Edge keys, one per hop of ``path``; only valid for ``MultiGraph``.
        This selects the *concrete* edges of the reference path: removing a
        path edge removes one key only, not every parallel edge between the
        endpoint pair.
    weight : string or function
        Edge weight attribute name, or a callable
        ``weight(u, v, data) -> number | None`` as used by the Dijkstra
        functions.  ``None`` hides an edge.

    Returns
    -------
    dict
        Mapping from each edge of the reference path to a 2-tuple
        ``(distance, replacement_path)``:

        - ``distance`` is the length of the shortest ``source``-``target``
          path avoiding that concrete edge.  It is ``math.inf`` when the
          edge is a bridge separating ``source`` from ``target``.
        - ``replacement_path`` is the corresponding node list, or ``None``
          when no such path exists.  In a ``MultiGraph`` the node list is
          returned; the bypass edge key between two consecutive nodes can be
          looked up in the graph.

        Edge keys are 2-tuples ``(u, v)`` for ``Graph`` and 3-tuples
        ``(u, v, key)`` for ``MultiGraph``.

    Raises
    ------
    NetworkXNotImplemented
        If ``G`` is directed.  Directed graphs are explicitly unsupported;
        the undirected cut proof does not apply.
    NodeNotFound
        If ``source`` or ``target`` is not in the graph.
    NetworkXError
        If ``path`` is not a shortest ``source``-``target`` path, is
        malformed, uses hidden edges, or if weights are negative.

    Notes
    -----
    Two shortest-path trees (from ``source`` and ``target``) that contain
    the reference path are built once, and every other edge contributes a
    candidate that covers the contiguous range of failing path edges for
    which it bypasses the failure.  A sweep over the path edges takes the
    minimum covering candidate per edge, giving ``O(m log n)`` time rather
    than one full Dijkstra run per edge.

    Examples
    --------
    >>> G = nx.Graph()
    >>> G.add_weighted_edges_from(
    ...     [(0, 1, 1), (1, 2, 1), (0, 2, 4)]
    ... )
    >>> nx.replacement_paths(G, 0, 2)
    {(0, 1): (4, [0, 2]), (1, 2): (4, [0, 2])}

    A bridge has no replacement:

    >>> T = nx.path_graph(3)
    >>> nx.replacement_paths(T, 0, 2)
    {(0, 1): (inf, None), (1, 2): (inf, None)}

    See Also
    --------
    replacement_path_lengths

    References
    ----------
    .. [1] K. Malik, A. K. Mittal, S. K. Gupta,
       "The k most vital arcs in the shortest path problem",
       Operations Research Letters 8 (1989) 223-227.
    """
    results, _ = _replacement_paths_core(
        G, source, target, path, path_keys, weight, compute_path=True
    )
    return results


@nx._dispatchable(edge_attrs="weight")
def replacement_path_lengths(G, source, target, path=None, *, path_keys=None,
                             weight="weight"):
    """Length of the best detour per edge failure on a shortest s-t path.

    Same as :func:`replacement_paths` but returns only the replacement
    distances: a mapping from each edge of the reference path to the
    shortest ``source``-``target`` distance that avoids that concrete edge
    (``math.inf`` when the edge is an s-t bridge).

    See :func:`replacement_paths` for parameters, exceptions and the
    supported graph classes.

    Examples
    --------
    >>> G = nx.cycle_graph(4)
    >>> nx.replacement_path_lengths(G, 0, 2)
    {(0, 1): 2, (1, 2): 2}
    """
    results, _ = _replacement_paths_core(
        G, source, target, path, path_keys, weight, compute_path=False
    )
    return results
