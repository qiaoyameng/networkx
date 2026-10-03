"""
Replacement paths for undirected graphs.

For a chosen shortest path between two nodes, the replacement path problem
asks, for each edge of that path, for the shortest path that avoids that
single edge.
"""

from heapq import heappop, heappush
from itertools import count
from math import inf

import networkx as nx
from networkx.utils import not_implemented_for

__all__ = ["replacement_paths"]


def _single_edge_weight(G, weight):
    """Weight of one individual edge, given its attribute dictionary."""
    if callable(weight):
        return weight
    return lambda u, v, data: data.get(weight, 1)


def _lexicographic_dijkstra(G, source, path_position, prefer_low, edge_weight):
    """Dijkstra from ``source`` with lexicographic ``(distance, position)`` keys.

    Every node ``v`` is labeled with

    - ``dist[v]``: the shortest-path distance from ``source`` to ``v``;
    - ``pos[v]``: an extremal position of the fixed source--target path
      ``P = (v_0, ..., v_k)`` (as given by ``path_position``) over all
      shortest ``source``--``v`` paths.  If ``prefer_low`` is True, this is
      the minimum, over shortest ``source``--``v`` paths, of the largest
      touched path position; otherwise it is the maximum, over shortest
      ``source``--``v`` paths, of the smallest touched path position;
    - ``parent[v]``: ``(predecessor, edge_key)`` on a shortest
      ``source``--``v`` path realizing ``pos[v]``.

    The lexicographic tie-breaking guarantees ``pos_low[v] <= pos_high[v]``
    for every node ``v`` when both variants are computed with the same
    ``path_position``, even in the presence of zero-weight edges.
    """
    sign = 1 if prefer_low else -1
    dist = {source: 0}
    pos = {source: path_position.get(source, 0)}
    parent = {source: None}
    seen = {source: (0, sign * pos[source])}
    settled = set()
    c = count()
    heap = [(0, sign * pos[source], next(c), source)]
    adj = G._adj
    multigraph = G.is_multigraph()
    while heap:
        d, _, _, u = heappop(heap)
        if seen[u] < (d, sign * pos[u]):
            continue
        settled.add(u)
        du, au = dist[u], pos[u]
        for v, edgedata in adj[u].items():
            if multigraph:
                key_best, data_best = min(
                    edgedata.items(), key=lambda kv: edge_weight(u, v, kv[1])
                )
                w = edge_weight(u, v, data_best)
            else:
                key_best = None
                w = edge_weight(u, v, edgedata)
            nd = du + w
            j = path_position.get(v)
            if j is None:
                na = au
            elif prefer_low:
                na = max(au, j)
            else:
                na = min(au, j)
            new_key = (nd, sign * na)
            if v in settled and new_key < seen[v]:
                raise ValueError("Contradictory paths found:", "negative weights?")
            if new_key < seen.get(v, (inf, inf)):
                seen[v] = new_key
                dist[v] = nd
                pos[v] = na
                parent[v] = (u, key_best)
                heappush(heap, (nd, sign * na, next(c), v))
    return dist, pos, parent


def _walk_up(parent, node, root_first=True):
    """Nodes on the parent chain between the root and ``node``."""
    chain = []
    while node is not None:
        chain.append(node)
        link = parent[node]
        node = link[0] if link is not None else None
    if root_first:
        chain.reverse()
    return chain


def _validate_shortest_path(
    G, path, path_keys, source, target, dist_target, edge_weight
):
    """Validate a user-supplied shortest path, returning ``(nodes, keys)``."""
    nodes = list(path)
    if len(nodes) == 0 or nodes[0] != source or nodes[-1] != target:
        raise nx.NetworkXError(
            f"path must start at the source {source!r} and end at {target!r}."
        )
    if len(set(nodes)) != len(nodes):
        raise nx.NetworkXError("path must not contain repeated nodes.")
    multigraph = G.is_multigraph()
    if path_keys is not None:
        if not multigraph:
            raise nx.NetworkXError("path_keys is only meaningful for multigraphs.")
        path_keys = list(path_keys)
        if len(path_keys) != len(nodes) - 1:
            raise nx.NetworkXError("path_keys must have one entry per edge of path.")
    keys = []
    total = 0
    for j, (u, v) in enumerate(zip(nodes, nodes[1:])):
        if not G.has_edge(u, v):
            raise nx.NetworkXError(f"path uses non-existent edge ({u!r}, {v!r}).")
        if multigraph:
            edgedata = G._adj[u][v]
            if path_keys is None:
                key = min(edgedata, key=lambda kk: edge_weight(u, v, edgedata[kk]))
            else:
                key = path_keys[j]
                if key not in edgedata:
                    raise nx.NetworkXError(
                        f"path uses non-existent edge ({u!r}, {v!r}, {key!r})."
                    )
            w = edge_weight(u, v, edgedata[key])
        else:
            key = None
            w = edge_weight(u, v, G._adj[u][v])
        keys.append(key)
        total += w
    if total != dist_target:
        raise nx.NetworkXError(
            f"path has total weight {total}, but the shortest {source!r}--{target!r} "
            f"distance is {dist_target}; path must be a shortest path."
        )
    return nodes, keys


@not_implemented_for("directed")
@nx._dispatchable(edge_attrs="weight")
def replacement_paths(
    G, source, target, weight="weight", path=None, path_keys=None, return_paths=False
):
    r"""Shortest detours around every edge of a shortest path.

    For a shortest ``source``--``target`` path ``P = (v_0, ..., v_k)``, the
    *replacement path* for the path edge ``e_i = (v_i, v_{i+1})`` is a
    shortest ``source``--``target`` path in the graph with only that single
    edge removed.  This function computes the replacement distance for every
    edge of ``P`` in one shot, without re-running a full shortest-path
    computation per edge.

    Parameters
    ----------
    G : NetworkX graph
        Undirected graph.  Both ``Graph`` and ``MultiGraph`` are supported.
        For multigraphs, only the individual path edge (identified by its
        key) is removed, never all parallel edges between its endpoints.
        Edge weights must be nonnegative.  Directed graphs are not
        supported; see Notes.

    source, target : node
        Endpoints of the shortest path.

    weight : string or function, optional (default="weight")
        Edge weight attribute name, or a function ``weight(u, v, data)``
        returning the weight of the edge with attribute dictionary ``data``.
        For multigraphs the function is applied to each individual edge.
        Missing attributes default to 1.

    path : list of nodes, optional
        The shortest path ``P`` to use, as a list of nodes from ``source``
        to ``target``.  It must be simple and have minimum total weight.
        If not given, a shortest path is selected automatically.

    path_keys : list, optional
        Only for multigraphs: the key of the edge used by each step of
        ``path``.  If not given, the minimum-weight edge between each pair
        of consecutive path nodes is used.

    return_paths : bool, optional (default=False)
        If True, each value of the returned dictionary is a
        ``(distance, path)`` pair, where ``path`` is a replacement path as a
        list of nodes (``None`` if no replacement exists).

    Returns
    -------
    dict
        Maps each edge of ``P`` to its replacement distance (a float), or to
        a ``(distance, path)`` pair if ``return_paths`` is True.  Edges are
        identified by ``(u, v)`` 2-tuples for ``Graph`` and by
        ``(u, v, key)`` 3-tuples for ``MultiGraph``, oriented along ``P``
        from ``source`` to ``target``.  The distance is ``math.inf`` (and
        the path ``None``) if removing the edge disconnects ``source`` from
        ``target``, i.e. the edge is an irreplaceable bridge.

    Raises
    ------
    NetworkXNoPath
        If there is no path between ``source`` and ``target``.

    NetworkXError
        If ``path`` (and ``path_keys``) is not a valid shortest
        ``source``--``target`` path.

    NetworkXNotImplemented
        If ``G`` is directed.

    Notes
    -----
    The algorithm shares one shortest-path computation across all path
    edges instead of deleting each edge and re-running Dijkstra's
    algorithm.  Two lexicographic Dijkstra passes label every node ``v``
    with its distance from ``source`` (resp. ``target``) and the
    earliest (resp. latest) position of ``P`` at which a shortest
    ``source``--``v`` (resp. ``v``--``target``) path leaves ``P``.  In an
    undirected graph, every replacement path consists of a shortest
    ``source``--``u`` prefix, a single *detour edge* ``(u, v)`` that is not
    the failed path edge, and a shortest ``v``--``target`` suffix [1]_.
    Each edge of the graph therefore defines one detour candidate per
    orientation that covers a contiguous interval ``[low[u], high[v] - 1]``
    of path positions, and a sweep over these intervals selects the
    cheapest candidate covering each path edge.  The running time is
    ``O(m log n + m log m)`` for ``n`` nodes and ``m`` edges, dominated by
    the two Dijkstra passes and sorting the candidates.

    The lexicographic tie-breaking on the two Dijkstra passes makes the
    result exact in the presence of tied shortest paths and zero-weight
    edges.  The characterization above does not hold for directed graphs
    (a replacement path there may need to re-use the shortest-path
    structure in both directions), so directed graphs are explicitly not
    supported rather than silently answered with the undirected rule.

    The input graph is never modified.

    References
    ----------
    .. [1] K. Malik, A. K. Mittal, S. K. Gupta. "The k most vital arcs in
       the shortest path problem." Operations Research Letters 8(4):223-227,
       1989.

    Examples
    --------
    >>> G = nx.Graph()
    >>> G.add_weighted_edges_from(
    ...     [(0, 1, 1), (1, 2, 1), (2, 3, 1), (0, 2, 3), (1, 3, 5)]
    ... )
    >>> rp = nx.replacement_paths(G, 0, 3)
    >>> for edge, dist in sorted(rp.items()):
    ...     print(edge, dist)
    (0, 1) 4
    (1, 2) 4
    (2, 3) 6

    >>> rp = nx.replacement_paths(G, 0, 3, return_paths=True)
    >>> rp[(0, 1)]
    (4, [0, 2, 3])

    See Also
    --------
    shortest_path, single_source_dijkstra
    """
    if source not in G:
        raise nx.NodeNotFound(f"Source node {source!r} not in graph.")
    if target not in G:
        raise nx.NodeNotFound(f"Target node {target!r} not in graph.")
    edge_weight = _single_edge_weight(G, weight)
    multigraph = G.is_multigraph()

    # One plain Dijkstra pass: distances from source and a shortest path.
    dist_from_source, _, parent_from_source = _lexicographic_dijkstra(
        G, source, {}, True, edge_weight
    )
    if target not in dist_from_source:
        raise nx.NetworkXNoPath(f"No path between {source} and {target}.")
    dist_target = dist_from_source[target]

    if path is None:
        if path_keys is not None:
            raise nx.NetworkXError("path_keys requires an explicit path.")
        nodes = _walk_up(parent_from_source, target)
        keys = []
        if multigraph:
            for u, v in zip(nodes, nodes[1:]):
                edgedata = G._adj[u][v]
                keys.append(
                    min(edgedata, key=lambda kk: edge_weight(u, v, edgedata[kk]))
                )
        else:
            keys = [None] * (len(nodes) - 1)
    else:
        nodes, keys = _validate_shortest_path(
            G, path, path_keys, source, target, dist_target, edge_weight
        )

    k = len(nodes) - 1
    if k == 0:
        return {}

    path_position = {v: j for j, v in enumerate(nodes)}
    dist_s, low, parent_s = _lexicographic_dijkstra(
        G, source, path_position, True, edge_weight
    )
    dist_t, high, parent_t = _lexicographic_dijkstra(
        G, target, path_position, False, edge_weight
    )

    # Every non-path edge (u, v) yields one detour candidate per orientation:
    # a shortest source--u prefix, the edge (u, v), and a shortest v--target
    # suffix.  The candidate bypasses exactly the path edges in the
    # contiguous interval [low[u], high[v] - 1].
    path_edges = set()
    for j in range(k):
        path_edges.add((nodes[j], nodes[j + 1], keys[j]))
        path_edges.add((nodes[j + 1], nodes[j], keys[j]))
    candidates = []
    if multigraph:
        edge_iter = (e for e in G.edges(keys=True, data=True))
    else:
        edge_iter = ((u, v, None, d) for u, v, d in G.edges(data=True))
    for u, v, key, data in edge_iter:
        if u == v or (u, v, key) in path_edges:
            continue
        if u not in dist_s or v not in dist_s:
            continue
        w = edge_weight(u, v, data)
        lo, hi = low[u], high[v] - 1
        if lo <= hi:
            candidates.append((lo, hi, dist_s[u] + w + dist_t[v], (u, v, key)))
        lo, hi = low[v], high[u] - 1
        if lo <= hi:
            candidates.append((lo, hi, dist_s[v] + w + dist_t[u], (v, u, key)))
    candidates.sort()

    # Sweep the path positions, keeping the cheapest candidate covering each.
    best_cost = [inf] * k
    best_edge = [None] * k
    heap = []
    c = count()
    idx = 0
    for i in range(k):
        while idx < len(candidates) and candidates[idx][0] <= i:
            lo, hi, cost, uvk = candidates[idx]
            heappush(heap, (cost, hi, next(c), uvk))
            idx += 1
        while heap and heap[0][1] < i:
            heappop(heap)
        if heap:
            cost, _, _, uvk = heap[0]
            best_cost[i] = cost
            best_edge[i] = uvk

    def replacement_path(i):
        uvk = best_edge[i]
        if uvk is None:
            return None
        u, v, _ = uvk
        return _walk_up(parent_s, u) + _walk_up(parent_t, v, root_first=False)

    result = {}
    for j in range(k):
        if multigraph:
            edge_id = (nodes[j], nodes[j + 1], keys[j])
        else:
            edge_id = (nodes[j], nodes[j + 1])
        if return_paths:
            result[edge_id] = (best_cost[j], replacement_path(j))
        else:
            result[edge_id] = best_cost[j]
    return result
