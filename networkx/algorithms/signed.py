"""Algorithms for signed graphs.

A signed graph is a graph whose edges carry a sign: positive edges
express attraction (endpoints prefer the same group) and negative edges
express repulsion (endpoints prefer different groups).  The *frustration*
of a 2-coloring of the nodes is the total weight of edges whose sign
disagrees with the coloring; minimizing it is the classic frustration
index problem of structural balance theory.
"""

from collections import namedtuple
from math import inf, isfinite, isclose
from numbers import Real

import networkx as nx

__all__ = ["minimum_frustration", "MinimumFrustrationResult"]

MinimumFrustrationResult = namedtuple(
    "MinimumFrustrationResult",
    [
        "partition",
        "frustrated_edges",
        "cost",
        "lower_bound",
        "is_optimal",
        "nodes_explored",
    ],
)
MinimumFrustrationResult.__doc__ = """Result of :func:`minimum_frustration`.

Attributes
----------
partition : dict
    Mapping of each node to its group, ``0`` or ``1``.  One node per
    connected component is fixed to group ``0`` to remove the global
    sign-flip symmetry.
frustrated_edges : list
    Edges whose sign is violated by ``partition``.  For ``Graph`` each
    edge is a ``(u, v)`` tuple; for ``MultiGraph`` a ``(u, v, key)``
    tuple.  Parallel edges are listed (and charged) individually.
    Unsatisfiable negative self-loops are always included.
cost : float
    Total weight of the frustrated edges.
lower_bound : float
    Certified lower bound on the optimal frustration.  Equals ``cost``
    when ``is_optimal`` is True.
is_optimal : bool
    True when the search completed and ``cost`` is proven minimal.
    False when the search-node budget was exhausted first; in that case
    ``partition``/``cost`` are the best solution found so far.
nodes_explored : int
    Number of branch-and-bound search nodes visited.
"""

_VALID_SIGNS = {1: 1, -1: -1, "+": 1, "-": -1}


def _violated(sign, side_u, side_v):
    """Whether an edge with normalized `sign` is violated by the sides."""
    return side_u != side_v if sign > 0 else side_u == side_v


def minimum_frustration(G, sign="sign", weight="weight", max_nodes=None):
    """Exact minimum-frustration 2-coloring of a small signed graph.

    Searches for the partition of the nodes into two groups that
    minimizes the total weight of violated edges: positive edges whose
    endpoints land in different groups and negative edges whose endpoints
    share a group.  Uses branch and bound, so it is intended for small
    graphs; no integer-programming solver is required.

    Parameters
    ----------
    G : NetworkX graph (undirected)
        A ``Graph`` or ``MultiGraph``.  Every edge must carry a `sign`
        attribute.  Parallel edges are charged individually.  A negative
        self-loop can never be satisfied and always contributes its
        weight to the cost.
    sign : string, optional (default="sign")
        Edge attribute holding the sign.  Valid values are ``1``, ``-1``,
        ``"+"`` and ``"-"``.
    weight : string, optional (default="weight")
        Edge attribute holding the cost of violating the edge.  Values
        must be finite, non-negative numbers; missing attributes default
        to 1.
    max_nodes : int or None, optional (default=None)
        Budget on the number of branch-and-bound search nodes.  If the
        budget is exhausted before the search finishes, the best
        solution found so far is returned together with a certified
        lower bound and ``is_optimal`` set to False.  ``None`` means no
        limit.

    Returns
    -------
    MinimumFrustrationResult
        Named tuple with fields ``partition``, ``frustrated_edges``,
        ``cost``, ``lower_bound``, ``is_optimal`` and ``nodes_explored``.

    Raises
    ------
    NetworkXNotImplemented
        If `G` is directed.
    NetworkXError
        If an edge has an invalid sign, an invalid (non-finite or
        negative) weight, or `max_nodes` is negative.  Signs are
        validated before weights.

    Examples
    --------
    >>> G = nx.Graph()
    >>> G.add_edge(0, 1, sign=1, weight=2.0)
    >>> G.add_edge(1, 2, sign=1, weight=2.0)
    >>> G.add_edge(0, 2, sign=-1, weight=5.0)  # frustrated triangle
    >>> res = nx.minimum_frustration(G)
    >>> res.cost, res.is_optimal
    (2.0, True)
    >>> res.partition[0] == res.partition[1] != res.partition[2]
    True

    Notes
    -----
    One vertex of each connected component is fixed to group ``0``,
    which removes the global flip symmetry without changing the optimum.
    The lower bound used for pruning adds, to the cost of edges with
    both endpoints assigned, the cheapest violation each unassigned
    vertex must already incur against the assigned vertices; these edge
    sets are disjoint, so the bound is admissible.

    References
    ----------
    .. [1] F. Harary, "On the measurement of structural balance",
       Behavioral Science, 4(4):316--323, 1959.
    """
    if G.is_directed():
        raise nx.NetworkXNotImplemented(
            "minimum_frustration is not implemented for directed graphs"
        )
    if max_nodes is not None:
        if isinstance(max_nodes, bool) or not isinstance(max_nodes, int):
            raise nx.NetworkXError("max_nodes must be an int or None")
        if max_nodes < 0:
            raise nx.NetworkXError("max_nodes must be non-negative")

    multigraph = G.is_multigraph()
    edge_iter = (
        G.edges(keys=True, data=True) if multigraph else G.edges(data=True)
    )
    raw_edges = list(edge_iter)

    # Validate all signs before any weight so sign errors surface first.
    signs = []
    for e in raw_edges:
        data = e[-1]
        s = data.get(sign)
        if isinstance(s, bool) or s not in _VALID_SIGNS:
            raise nx.NetworkXError(
                f"edge {e[:-1]} has invalid sign {s!r}; "
                "expected one of 1, -1, '+', '-'"
            )
        signs.append(_VALID_SIGNS[s])

    edges = []  # (u, v, sign, weight, edge_id); self-loops excluded
    offset = 0.0  # unavoidable cost of negative self-loops
    always_frustrated = []
    for e, s in zip(raw_edges, signs):
        data = e[-1]
        w = data.get(weight, 1.0)
        if (
            isinstance(w, bool)
            or not isinstance(w, Real)
            or not isfinite(w)
            or w < 0
        ):
            raise nx.NetworkXError(
                f"edge {e[:-1]} has invalid weight {w!r}; "
                "weights must be finite non-negative numbers"
            )
        w = float(w)
        u, v = e[0], e[1]
        edge_id = e[:3] if multigraph else e[:2]
        if u == v:
            if s < 0:
                offset += w
                always_frustrated.append(edge_id)
            continue
        edges.append((u, v, s, w, edge_id))

    # Branch-and-bound state shared across components.
    budget = inf if max_nodes is None else max_nodes
    state = {"nodes": 0, "exhausted": False}

    total_best = offset
    total_lb = offset
    partition = {}

    for comp in nx.connected_components(G):
        comp_edges = [e for e in edges if e[0] in comp]
        # Order vertices by decreasing incident weight for tight bounds.
        weight_degree = {v: 0.0 for v in comp}
        for u, v, s, w, _ in comp_edges:
            weight_degree[u] += w
            weight_degree[v] += w
        order = sorted(comp, key=lambda v: -weight_degree[v])
        k = len(order)
        pos = {v: i for i, v in enumerate(order)}
        adj = [[] for _ in range(k)]
        for u, v, s, w, _ in comp_edges:
            i, j = pos[u], pos[v]
            adj[i].append((j, s, w))
            adj[j].append((i, s, w))

        if k == 1:
            partition[order[0]] = 0
            continue

        # Greedy incumbent: first vertex fixed to 0, each later vertex
        # takes the side cheapest against already-placed vertices.
        greedy = [None] * k
        greedy[0] = 0
        greedy_cost = 0.0
        for i in range(1, k):
            cand = [0.0, 0.0]
            for j, s, w in adj[i]:
                if greedy[j] is not None:
                    for side in (0, 1):
                        if _violated(s, side, greedy[j]):
                            cand[side] += w
            greedy[i] = 0 if cand[0] <= cand[1] else 1
            greedy_cost += cand[greedy[i]]

        best = [greedy_cost, greedy]
        min_live = [inf]
        assign = [None] * k
        c0 = [0.0] * k  # violation cost vs assigned vertices if side 0
        c1 = [0.0] * k  # violation cost vs assigned vertices if side 1

        def dfs(i, incurred):
            state["nodes"] += 1
            lb = incurred
            for t in range(i, k):
                lb += c0[t] if c0[t] < c1[t] else c1[t]
            if state["exhausted"]:
                if lb < min_live[0]:
                    min_live[0] = lb
                return
            if lb >= best[0] and not isclose(
                lb, best[0], rel_tol=1e-9, abs_tol=1e-12
            ):
                return
            if state["nodes"] > budget:
                state["exhausted"] = True
                min_live[0] = lb
                return
            if i == k:
                if incurred < best[0]:
                    best[0] = incurred
                    best[1] = list(assign)
                return
            costs = [0.0, 0.0]
            for j, s, w in adj[i]:
                aj = assign[j]
                if aj is not None:
                    if _violated(s, 0, aj):
                        costs[0] += w
                    if _violated(s, 1, aj):
                        costs[1] += w
            sides = (0,) if i == 0 else ((0, 1) if costs[0] <= costs[1] else (1, 0))
            for side in sides:
                for j, s, w in adj[i]:
                    if assign[j] is None:
                        if _violated(s, 0, side):
                            c0[j] += w
                        if _violated(s, 1, side):
                            c1[j] += w
                assign[i] = side
                dfs(i + 1, incurred + costs[side])
                assign[i] = None
                for j, s, w in adj[i]:
                    if assign[j] is None:
                        if _violated(s, 0, side):
                            c0[j] -= w
                        if _violated(s, 1, side):
                            c1[j] -= w

        dfs(0, 0.0)

        comp_lb = best[0] if not state["exhausted"] else min(min_live[0], best[0])
        for i, v in enumerate(order):
            partition[v] = best[1][i]
        total_best += best[0]
        total_lb += comp_lb
        # Remaining components contribute 0 to the lower bound.

    is_optimal = not state["exhausted"] or total_best <= total_lb or isclose(
        total_best, total_lb, rel_tol=1e-9, abs_tol=1e-12
    )

    frustrated = list(always_frustrated)
    for u, v, s, w, edge_id in edges:
        if _violated(s, partition[u], partition[v]):
            frustrated.append(edge_id)

    return MinimumFrustrationResult(
        partition=partition,
        frustrated_edges=frustrated,
        cost=total_best,
        lower_bound=total_lb,
        is_optimal=is_optimal,
        nodes_explored=state["nodes"],
    )
