"""
Exact resource-constrained shortest paths.

Finds minimum-cost paths whose cumulative resource consumption does not
exceed a budget, while keeping the full set of non-dominated
``(cost, resource)`` labels at each node.  No external solver is used.
"""

from heapq import heappop, heappush
from itertools import count
from math import isfinite

import networkx as nx

__all__ = ["resource_constrained_path"]


def _is_nonneg_number(value):
    """True for finite real numbers ``>= 0`` (booleans accepted)."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, complex)
        and isfinite(value)
        and value >= 0
    )


def _build_edge_lookup(G, attr):
    """Return ``(u, v, key) -> value`` for an attr name or callable.

    Raises KeyError when a string-named attribute is absent, so that a
    missing attribute is reported instead of silently treated as zero.
    """
    if callable(attr):
        if G.is_multigraph():
            return lambda u, v, key: attr(u, v, G[u][v][key])
        return lambda u, v, key: attr(u, v, G[u][v])
    if G.is_multigraph():
        return lambda u, v, key: G[u][v][key][attr]
    return lambda u, v, key: G[u][v][attr]


def _check_nonneg(value, what):
    if not _is_nonneg_number(value):
        raise ValueError(
            f"{what} must be a finite non-negative number; got {value!r}"
        )


def _dominates(lhs_cost, lhs_res, rhs_cost, rhs_res):
    """True if ``lhs`` is no worse on both criteria and strictly better on
    at least one."""
    return (
        lhs_cost <= rhs_cost
        and lhs_res <= rhs_res
        and (lhs_cost < rhs_cost or lhs_res < rhs_res)
    )


@nx._dispatchable
def resource_constrained_path(
    G,
    source,
    target,
    budget,
    cost="cost",
    resource="resource",
    *,
    return_front=False,
    return_edge_keys=False,
    max_labels=10_000,
):
    r"""Find a minimum-cost path subject to a resource budget.

    Every edge carries two non-negative values: a `cost` to be minimized
    and a `resource` whose path sum must not exceed `budget`.  The
    resource-constrained shortest path problem is NP-hard, so this exact
    algorithm keeps the full set of non-dominated ``(cost, resource)``
    *labels* at each node (a multiobjective label-setting search) rather
    than a single distance per node.  The number of labels can grow
    exponentially; `max_labels` bounds the work and the algorithm reports
    incompleteness instead of returning an uncertified answer.

    Parameters
    ----------
    G : NetworkX graph
        A directed or undirected graph (multigraphs are supported).
    source : node
        Starting node.
    target : node
        Ending node.
    budget : number
        Maximum allowed sum of edge resources.  Must be a finite,
        non-negative real number.
    cost : string or callable (default: "cost")
        Name of the cost edge attribute, or a function ``f(u, v, d)``
        returning a finite non-negative number; returning None hides an
        edge.
    resource : string or callable (default: "resource")
        Name of the resource edge attribute, or a function with the same
        convention as `cost`.
    return_front : bool (default: False)
        If True, also return the Pareto front of non-dominated feasible
        ``(cost, resource, path)`` triples at `target`.
    return_edge_keys : bool (default: False)
        If True and `G` is a multigraph, paths are returned as
        ``(nodes, edge_keys)`` pairs so that the exact parallel edge used
        at each step (and thus the true feasible walk) can be recovered.
        Ignored for non-multigraphs.
    max_labels : int (default: 10000)
        Upper bound on the number of labels created (including the source
        label).  If the bound is hit before an exact answer can be
        certified, :exc:`~networkx.exception.ExceededMaxIterations` is
        raised and no path is returned.

    Returns
    -------
    min_cost, path : tuple
        Minimum achievable cost and the corresponding path, when
        `return_front` is False.  With `return_edge_keys` on a multigraph,
        `path` is a ``(nodes, edge_keys)`` pair, otherwise a node list.
    min_cost, path, front : tuple
        When `return_front` is True, `front` is a list of
        ``(cost, resource, path)`` triples sorted by increasing cost (then
        resource), pairwise non-dominated and all feasible for `budget`.
        `path` is the cheapest front member.

    Raises
    ------
    NodeNotFound
        If `source` or `target` is not in `G`.
    NetworkXNoPath
        If no path from `source` to `target` respects the budget.
    ValueError
        If `budget` is not a finite non-negative number, `max_labels` is
        not positive, an edge lacks a cost/resource attribute, or any edge
        cost/resource is not a finite non-negative number.
    ExceededMaxIterations
        If more than `max_labels` labels would be needed to certify the
        exact optimum; the result is deliberately *not* returned.

    Examples
    --------
    >>> G = nx.DiGraph()
    >>> G.add_edge(0, 1, cost=5, resource=0)
    >>> G.add_edge(0, 2, cost=1, resource=1)
    >>> G.add_edge(2, 1, cost=1, resource=1)
    >>> resource_constrained_path(G, 0, 1, budget=2)
    (2, [0, 2, 1])

    Notes
    -----
    Labels settle in lexicographic ``(cost, resource)`` order; this is a
    label-setting (Dijkstra-like) scheme valid for non-negative edge
    values, so the first settled target label is the cheapest feasible
    path.  Each label stores the id of its predecessor *label*, which lets
    the exact feasible walk be reconstructed even when a node carries many
    labels.  Equal ``(cost, resource)`` labels are merged; zero-cost and
    zero-resource loops therefore cannot spin or multiply labels, and
    parallel multiedges that produce the same label are deduplicated.
    """
    if source not in G:
        raise nx.NodeNotFound(f"source {source} not found in graph")
    if target not in G:
        raise nx.NodeNotFound(f"target {target} not found in graph")
    if not _is_nonneg_number(budget):
        raise ValueError(
            f"budget must be a finite non-negative number; got {budget!r}"
        )
    max_labels = int(max_labels)
    if max_labels < 1:
        raise ValueError("max_labels must be a positive integer")

    get_cost = _build_edge_lookup(G, cost)
    get_resource = _build_edge_lookup(G, resource)

    # Settled labels per node: (cost, resource, label_id), pairwise
    # non-dominated, appended in lexicographic settle order.
    finalized = {source: []}
    # Labels still on the heap per node: (cost, resource) -> label_id.
    # `dead` holds heap label ids superseded by a dominating new label; the
    # pair stays queued until the stale entry is popped.
    queued = {source: {(0, 0): 0}}
    dead = set()
    # info[id] = (node, cost, res, predecessor_label_id, edge_key).  The
    # predecessor points at a specific label rather than just a node.
    label_info = []
    fringe = []  # (cost, res, tie, label_id); tie avoids node comparison
    tie_counter = count()

    def push_label(node, cost_value, res_value, pred_id, edge_key):
        label_id = len(label_info)
        label_info.append((node, cost_value, res_value, pred_id, edge_key))
        queued.setdefault(node, {})[(cost_value, res_value)] = label_id
        heappush(fringe, (cost_value, res_value, next(tie_counter), label_id))
        return label_id

    push_label(source, 0, 0, None, None)
    n_labels = 1

    def build_path(label_id):
        nodes = []
        edge_keys = []
        while label_id is not None:
            node, _, _, pred_id, edge_key = label_info[label_id]
            nodes.append(node)
            if edge_key is not None:
                edge_keys.append(edge_key)
            label_id = pred_id
        nodes.reverse()
        edge_keys.reverse()
        return nodes, edge_keys

    def dominated_label(node, cost_value, res_value):
        for other_cost, other_res, _ in finalized.get(node, ()):
            if _dominates(other_cost, other_res, cost_value, res_value):
                return True
        for other_cost, other_res in queued.get(node, {}):
            if _dominates(other_cost, other_res, cost_value, res_value):
                return True
        return False

    best = None  # (cost, res, label_id)
    adj = G._adj

    while fringe:
        cost_value, res_value, _, label_id = heappop(fringe)
        node = label_info[label_id][0]

        pair = (cost_value, res_value)
        if label_id in dead or queued.get(node, {}).get(pair) != label_id:
            continue  # settled or superseded by a dominating label
        finalized.setdefault(node, []).append(
            (cost_value, res_value, label_id)
        )
        del queued[node][pair]

        if node == target:
            if best is None:
                best = (cost_value, res_value, label_id)
            if not return_front:
                break  # lex order guarantees this is the cheapest feasible
            continue  # no need to extend walks past the target

        neighbors = adj[node]
        if G.is_multigraph():
            edges = (
                (neighbor, key)
                for neighbor, to_neighbor in neighbors.items()
                for key in to_neighbor
            )
        else:
            edges = ((neighbor, None) for neighbor in neighbors)

        for neighbor, edge_key in edges:
            try:
                edge_cost = get_cost(node, neighbor, edge_key)
            except KeyError:
                raise ValueError(
                    f"edge ({node}, {neighbor}) is missing cost attribute "
                    f"{cost!r}"
                ) from None
            try:
                edge_res = get_resource(node, neighbor, edge_key)
            except KeyError:
                raise ValueError(
                    f"edge ({node}, {neighbor}) is missing resource "
                    f"attribute {resource!r}"
                ) from None
            if edge_cost is None or edge_res is None:
                continue  # callable hid this edge
            _check_nonneg(edge_cost, f"edge ({node}, {neighbor}) cost")
            _check_nonneg(edge_res, f"edge ({node}, {neighbor}) resource")

            new_cost = cost_value + edge_cost
            new_res = res_value + edge_res
            if new_res > budget:
                continue

            pair = (new_cost, new_res)
            neighbor_queued = queued.setdefault(neighbor, {})
            if pair in neighbor_queued:
                continue
            if any(
                (other_cost, other_res) == pair
                for other_cost, other_res, _ in finalized.get(neighbor, ())
            ):
                continue  # duplicate: terminates zero-cost/zero-res loops
            if dominated_label(neighbor, new_cost, new_res):
                continue

            # Supersede labels still on the heap that the new label
            # dominates (e.g. a better walk reaching the same pair).
            for other_pair in [
                other_pair
                for other_pair in neighbor_queued
                if _dominates(new_cost, new_res, other_pair[0], other_pair[1])
            ]:
                dead.add(neighbor_queued.pop(other_pair))

            if n_labels >= max_labels:
                raise nx.ExceededMaxIterations(
                    f"label cap of {max_labels} reached before the search "
                    "could certify an exact optimum; increase max_labels. "
                    "No result is returned because optimality is unproven."
                )
            push_label(neighbor, new_cost, new_res, label_id, edge_key)
            n_labels += 1

    if best is None:
        raise nx.NetworkXNoPath(
            f"No path from {source} to {target} with resource <= {budget}."
        )

    def path_for(label_id):
        nodes, edge_keys = build_path(label_id)
        if return_edge_keys and G.is_multigraph():
            return nodes, edge_keys
        return nodes

    best_cost, _, best_id = best
    best_path = path_for(best_id)
    if not return_front:
        return best_cost, best_path

    # Settled target labels are lexicographically ordered and pairwise
    # non-dominated: exactly the feasible Pareto front.
    front = [
        (front_cost, front_res, path_for(label_id))
        for front_cost, front_res, label_id in finalized[target]
    ]
    return best_cost, best_path, front
