"""Minimum frustration index of a signed graph.

A signed graph carries a sign on every edge. A positive edge is satisfied
when its endpoints belong to the same group, while a negative edge is
satisfied when its endpoints belong to different groups. The *minimum
frustration* (also called the *frustration index*) is the minimum total
weight of edges whose relation is violated over all bipartitions of the
nodes.
"""

import sys
from math import isfinite

import networkx as nx
from networkx.utils import not_implemented_for

__all__ = ["FrustrationResult", "minimum_frustration"]


class FrustrationResult(tuple):
    """Result of :func:`minimum_frustration`.

    Behaves as the named tuple ``(groups, frustrated_edges, cost)`` and
    additionally exposes search diagnostics as attributes.
    """


    def __new__(
        cls,
        groups,
        frustrated_edges,
        cost,
        lower_bound=None,
        nodes_explored=0,
        node_budget=None,
        proven_optimal=True,
    ):
        obj = tuple.__new__(cls, (groups, frustrated_edges, cost))
        obj.lower_bound = cost if lower_bound is None else lower_bound
        obj.nodes_explored = nodes_explored
        obj.node_budget = node_budget
        obj.proven_optimal = proven_optimal
        return obj

    def __repr__(self):
        return (
            f"FrustrationResult(groups={self[0]!r}, "
            f"frustrated_edges=<{len(self[1])} edges>, cost={self[2]!r})"
        )

    @property
    def groups(self):
        return self[0]

    @property
    def frustrated_edges(self):
        return self[1]

    @property
    def cost(self):
        return self[2]


@not_implemented_for("directed")
@nx._dispatchable
def minimum_frustration(G, *, sign_attr="sign", weight="weight", node_budget=None):
    r"""Find the minimum frustration of a signed graph exactly.

    Partition the nodes into two groups so that positive edges join nodes
    in the same group and negative edges join nodes in different groups.
    The weight of an edge is paid exactly when its relation is violated.
    The minimum total cost is found with a branch-and-bound search without
    integer programming dependencies.

    Parameters
    ----------
    G : undirected graph
        A signed graph (or signed multigraph). Every edge must carry a
        boolean sign: ``True`` denotes a positive edge and ``False`` a
        negative edge.
    sign_attr : str
        Edge attribute that stores the sign of each edge. The attribute
        must be present on every edge and its value must be ``True`` or
        ``False``.
    weight : str or None
        Edge attribute that stores the cost of violating the edge. Weights
        must be finite and non-negative; edges without the attribute have
        unit weight. Set to ``None`` to give every edge unit weight.
    node_budget : int, optional
        Maximum number of branch-and-bound search nodes to expand. When the
        search exhausts the budget before proving optimality, the best
        feasible partition found so far is returned together with a lower
        bound and ``proven_optimal=False``. By default (``None``) the
        search runs until the optimum is proved.

    Returns
    -------
    FrustrationResult
        A tuple ``(groups, frustrated_edges, cost)`` where:

        - ``groups`` maps each node to ``0`` or ``1``.
        - ``frustrated_edges`` is the set of violated edges. Edges are
          ``(u, v)`` pairs for a :class:`~networkx.Graph` and
          ``(u, v, key)`` triples for a :class:`~networkx.MultiGraph`.
        - ``cost`` is the total weight of the violated edges.

        The result also exposes:

        - ``lower_bound``: proven lower bound on the optimal cost.
        - ``proven_optimal``: whether optimality was proved.
        - ``nodes_explored`` and ``node_budget``: search bookkeeping.

    Raises
    ------
    ValueError
        If an edge has a missing or non-boolean sign, a non-finite or
        negative weight, or if ``node_budget`` is not a positive integer.
    NetworkXNotImplemented
        If ``G`` is directed.

    Notes
    -----
    Solving the frustration index is NP-hard [1]_. One node of each
    connected component is fixed to group ``0``; this removes the global
    flip symmetry of every component without changing the optimum. At each
    search node the partial cost of edges with both endpoints assigned is
    combined with a node-wise relaxation over edges incident to the
    assigned nodes to obtain a lower bound, and the branch is pruned when
    this bound cannot beat the incumbent. Negative self-loops are never
    satisfiable and their weight is always paid, while positive
    self-loops are always satisfied.

    Examples
    --------
    A triangle containing one negative edge is not balanced; one edge must
    be frustrated:

    >>> G = nx.Graph()
    >>> G.add_edge(0, 1, sign=True)
    >>> G.add_edge(1, 2, sign=True)
    >>> G.add_edge(2, 0, sign=False)
    >>> result = nx.minimum_frustration(G)
    >>> result.cost
    1
    >>> groups, frustrated_edges, cost = result
    >>> frustrated_edges
    {(0, 2)}

    References
    ----------
    .. [1] Aref, S., Mason, A. J., Wilson, M. C. (2016).
       "A modeling and computational study of the frustration index in
       signed networks." *Networks and Spatial Economics*,
       18, 429–455. <https://doi.org/10.1007/s11067-017-9365-8>
    """
    if node_budget is not None:
        if isinstance(node_budget, bool) or not isinstance(node_budget, int):
            raise ValueError("node_budget must be a positive integer or None")
        if node_budget < 1:
            raise ValueError("node_budget must be a positive integer or None")

    multigraph = G.is_multigraph()

    # Each adjacency entry is ``(neighbor, positive, weight, edge_id)`` with
    # edge_id (u, v) for graphs and (u, v, key) for multigraphs.
    adj = {u: [] for u in G}
    edges = []
    self_loop_cost = 0
    negative_self_loops = []

    edge_iter = G.edges(keys=True, data=True) if multigraph else G.edges(data=True)
    for edge in edge_iter:
        if multigraph:
            u, v, key, data = edge
            eid = (u, v, key)
        else:
            u, v, data = edge
            eid = (u, v)
        if sign_attr not in data:
            raise ValueError(
                f"edge {eid} is missing the {sign_attr!r} sign attribute"
            )
        sign = data[sign_attr]
        if not isinstance(sign, bool):
            raise ValueError(
                f"edge {eid} has invalid sign {sign!r}; "
                "signs must be True or False"
            )
        w = 1 if weight is None else data.get(weight, 1)
        if isinstance(w, bool) or not isinstance(w, (int, float)) or not isfinite(w):
            raise ValueError(f"weight of edge {eid} must be a finite number")
        if w < 0:
            raise ValueError(f"weight of edge {eid} must be non-negative")
        if u == v:
            if not sign:
                self_loop_cost += w
                negative_self_loops.append(eid)
            continue
        edges.append((u, v, sign, w, eid))
        adj[u].append((v, sign, w, eid))
        adj[v].append((u, sign, w, eid))

    nodes = list(G)

    def added_cost(groups, u, bit):
        """Cost of edges from ``u`` to assigned nodes when ``u`` takes ``bit``."""
        total = 0
        for v, positive, w, _ in adj[u]:
            if v in groups:
                different = bit != groups[v]
                violated = different if positive else (not different)
                if violated:
                    total += w
        return total

    def lower_bound(groups, incurred):
        """Incurred cost plus a node-wise relaxation over boundary edges."""
        bound = incurred + self_loop_cost
        for u in nodes:
            if u in groups:
                continue
            cost_zero = 0
            cost_one = 0
            for v, positive, w, _ in adj[u]:
                if v not in groups:
                    continue
                gv = groups[v]
                # u = 0: positive violated if v = 1; negative violated if v = 0.
                if (positive and gv == 1) or (not positive and gv == 0):
                    cost_zero += w
                # u = 1: positive violated if v = 0; negative violated if v = 1.
                if (positive and gv == 0) or (not positive and gv == 1):
                    cost_one += w
            bound += cost_zero if cost_zero < cost_one else cost_one
        return bound

    # Pin one root of each connected component to 0 to break flip symmetry.
    assignment = {}
    for component in nx.connected_components(G):
        assignment[next(iter(component))] = 0

    state = {
        "best_cost": None,
        "best_groups": None,
        "explored": 1,
        "frontier_floor": lower_bound(assignment, 0),
        "exhausted": False,
    }

    def greedy(groups, incurred):
        """Greedily assign the remaining nodes; return ``(cost, assignment)``."""
        complete = dict(groups)
        remaining = [u for u in nodes if u not in complete]
        cost = incurred + self_loop_cost
        while remaining:
            best_node = None
            best_bit = 0
            best_cost = None
            for u in remaining:
                for bit in (0, 1):
                    c = added_cost(complete, u, bit)
                    if best_cost is None or c < best_cost:
                        best_cost = c
                        best_node = u
                        best_bit = bit
            complete[best_node] = best_bit
            cost += best_cost
            remaining.remove(best_node)
        return cost, complete

    best_cost, best_groups = greedy(assignment, 0)
    state["best_cost"] = best_cost
    state["best_groups"] = best_groups

    if best_cost > state["frontier_floor"]:
        sys.setrecursionlimit(max(sys.getrecursionlimit(), len(nodes) + 100))

        def choose_node(groups):
            """Unassigned node with the largest total weight to assigned nodes."""
            choice = None
            choice_score = None
            for u in nodes:
                if u in groups:
                    continue
                score = 0
                connected = False
                for v, _, w, _ in adj[u]:
                    if v in groups:
                        connected = True
                        score += w
                if connected and (choice_score is None or score > choice_score):
                    choice_score = score
                    choice = u
            return choice

        def dfs(groups, incurred):
            node = choose_node(groups)
            if node is None:
                full_cost = incurred + self_loop_cost
                if full_cost < state["best_cost"]:
                    state["best_cost"] = full_cost
                    state["best_groups"] = dict(groups)
                return
            for bit in (0, 1):
                if state["exhausted"]:
                    return
                over_budget = (
                    node_budget is not None and state["explored"] >= node_budget
                )
                if over_budget:
                    # Cannot expand this child; if its lower bound already
                    # proves it cannot improve the incumbent, it is pruned.
                    groups[node] = bit
                    child_incurred = incurred + added_cost(groups, node, bit)
                    bound = lower_bound(groups, child_incurred)
                    del groups[node]
                    if bound >= state["best_cost"]:
                        continue
                    state["exhausted"] = True
                    return
                groups[node] = bit
                child_incurred = incurred + added_cost(groups, node, bit)
                bound = lower_bound(groups, child_incurred)
                state["explored"] += 1
                if bound < state["frontier_floor"]:
                    state["frontier_floor"] = bound
                if bound < state["best_cost"]:
                    gcost, ggroups = greedy(groups, child_incurred)
                    if gcost < state["best_cost"]:
                        state["best_cost"] = gcost
                        state["best_groups"] = ggroups
                    dfs(groups, child_incurred)
                del groups[node]
                if state["exhausted"]:
                    return

        dfs(assignment, 0)

    proven = not state["exhausted"]
    groups_out = {u: state["best_groups"][u] for u in nodes}
    frustrated = set(negative_self_loops)
    total = self_loop_cost
    for u, v, positive, w, eid in edges:
        different = groups_out[u] != groups_out[v]
        violated = (not different) if not positive else different
        if violated:
            frustrated.add(eid)
            total += w

    return FrustrationResult(
        groups_out,
        frustrated,
        total,
        lower_bound=total if proven else state["frontier_floor"],
        nodes_explored=state["explored"],
        node_budget=node_budget,
        proven_optimal=proven,
    )
