"""
Lazy enumeration of all minimum s-t cuts of a capacitated DiGraph.

A single maximum flow is computed and the minimum cuts are enumerated from
the residual network, without re-running a maximum flow algorithm for each
cut and without perturbing edge capacities.
"""

from collections import namedtuple

import networkx as nx

from .preflowpush import preflow_push
from .utils import _capacity_function

__all__ = ["MinimumCut", "AllMinimumCuts", "all_minimum_cuts"]

default_flow_func = preflow_push

MinimumCut = namedtuple("MinimumCut", ["value", "partition", "cut_edges"])
MinimumCut.__doc__ = """A minimum s-t cut.

Each instance is a tuple ``(value, partition, cut_edges)`` where

- ``value`` is the common capacity of every minimum cut (the maximum flow
  value),
- ``partition`` is a tuple ``(reachable, unreachable)`` of the two node sets
  defining the cut; every node of the graph belongs to exactly one side,
- ``cut_edges`` is a set of directed edges ``(u, v)`` with ``u`` in the
  reachable side and ``v`` in the unreachable side.
"""


class AllMinimumCuts:
    """Lazy iterator over every minimum s-t cut of a graph.

    Instances are created with :func:`all_minimum_cuts`; this class is not
    intended to be instantiated directly. Enumeration reuses the residual
    network of a single maximum flow computation and proceeds one cut at a
    time, so iterating over the first few cuts does not enumerate the
    potentially exponentially many remaining cuts.

    Attributes
    ----------
    cut_value : int or float
        The capacity of every enumerated cut (the maximum flow value).
    exhausted : bool
        ``False`` while more minimum cuts may exist. It becomes ``True`` once
        enumeration has produced every minimum cut; if iteration stops because
        a ``limit`` was reached it stays ``False``.

    See Also
    --------
    all_minimum_cuts
    """

    def __init__(self, G, R, condensation, cs, ct, limit):
        self._G = G
        self._C = condensation
        self._cs = cs
        self._ct = ct
        self._limit = limit
        self.cut_value = R.graph["flow_value"]
        self.exhausted = False
        self._produced = 0
        self._iterator = self._enumerate()

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._iterator)

    def _enumerate(self):
        C = self._C
        members = {c: C.nodes[c]["members"] for c in C}
        cs, ct = self._cs, self._ct

        # Nodes reachable from the source SCC in the condensation DAG are
        # forced onto the source side; nodes that can reach the sink SCC are
        # forced onto the sink side. The rest are free.
        forced_S = _descendants(C, cs)
        forced_T = _ancestors(C, ct)

        def materialize(S_components):
            reachable = set()
            for component in S_components:
                reachable |= members[component]
            unreachable = set(self._G) - reachable
            cut_edges = {
                (u, v)
                for u, v, attr in self._G.edges(data=True)
                if u in reachable and v in unreachable
            }
            return MinimumCut(
                self.cut_value, (reachable, unreachable), cut_edges
            )

        # Branch-and-bound enumeration of the closed sets of the condensation
        # DAG. Each state is (source_components, sink_components). Fixing a
        # component on a side forces its descendants to the source side and its
        # ancestors to the sink side. The two child states of every branch are
        # disjoint, so each legal partition is generated exactly once.
        stack = [(forced_S, forced_T)]
        while stack:
            on_source, on_sink = stack.pop()
            pivot = next(
                (
                    c
                    for c in C
                    if c not in on_source and c not in on_sink
                ),
                None,
            )
            if pivot is None:
                if self._limit is not None and self._produced >= self._limit:
                    self.exhausted = False
                    return
                self._produced += 1
                yield materialize(on_source)
                continue
            # Source branch: the pivot and every descendant are on the source
            # side.
            new_source = on_source | _descendants(C, pivot)
            if not (new_source & on_sink):
                stack.append((new_source, on_sink))
            # Sink branch: the pivot and every ancestor are on the sink side.
            new_sink = on_sink | _ancestors(C, pivot)
            if not (new_sink & on_source):
                stack.append((on_source, new_sink))
        self.exhausted = True


def _descendants(C, source):
    """Return the set of components reachable from ``source`` in C."""
    seen = {source}
    stack = [source]
    while stack:
        u = stack.pop()
        for v in C.successors(u):
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return seen


def _ancestors(C, target):
    """Return the components that can reach ``target`` in C."""
    seen = {target}
    stack = [target]
    while stack:
        v = stack.pop()
        for u in C.predecessors(v):
            if u not in seen:
                seen.add(u)
                stack.append(u)
    return seen


@nx._dispatchable(
    graphs="flowG", edge_attrs={"capacity": float("inf")}, preserve_edge_attrs=True
)
def all_minimum_cuts(
    flowG, _s, _t, capacity="capacity", flow_func=None, tol=None, limit=None, **kwargs
):
    """Enumerate all minimum (s, t)-cuts of a capacitated DiGraph.

    A single maximum flow is computed. The residual network is contracted by
    strongly connected components and the minimum cuts are enumerated from the
    closure constraints of the resulting acyclic graph. No maximum flow is
    re-computed for individual cuts and no capacity perturbation is used.

    Parameters
    ----------
    flowG : NetworkX DiGraph
        Graph with finite, non-negative edge capacities. MultiGraphs and
        MultiDiGraphs are not supported.
    _s : node
        Source node.
    _t : node
        Sink node. It must differ from `_s`.
    capacity : string or function (default='capacity')
        Edge attribute, or a function ``(u, v, attrs) -> number``, giving the
        finite non-negative capacity of each edge. Edges without this attribute
        are treated as having infinite capacity, which is not supported and
        raises :exc:`NetworkXUnbounded`.
    flow_func : function (default=None)
        Maximum flow function to use, following the NetworkX residual network
        conventions. Defaults to :meth:`preflow_push`, which is invoked without
        ``value_only`` so that its feasible-flow conversion is performed.
    tol : float (default=None)
        Non-negative residual-capacity tolerance. A residual arc ``(u, v)`` is
        considered to have spare capacity when
        ``capacity - flow > tol``. With ``tol=None``, the tolerance is ``0``
        when every capacity is an integer (giving exact results on integer
        data) and ``1e-12`` times the largest finite capacity otherwise; the
        latter enumeration is approximate because floating-point arithmetic
        is involved. Pass ``0.0`` for strict comparison on floating-point
        data or any explicit non-negative tolerance.
    limit : int (default=None)
        Maximum number of minimum cuts to enumerate. When the limit is reached
        iteration stops with ``exhausted`` set to ``False``; the function never
        claims that a truncated enumeration is complete. With ``limit=None``
        every minimum cut is produced.

    Returns
    -------
    cuts : AllMinimumCuts
        A lazy iterator yielding :class:`MinimumCut` objects, one per distinct
        minimum cut. Each result has the common cut value, a partition of all
        the graph nodes, and the cut edge set. The iterator exposes the common
        cut value as ``cuts.cut_value`` and reports whether enumeration is
        complete through ``cuts.exhausted``.

    Raises
    ------
    NetworkXError
        If the source and sink are the same node, if `limit` is not a positive
        integer, if `tol` is negative, or if the graph is a MultiGraph.
    NetworkXUnbounded
        If the graph has an infinite-capacity s-t path.
    NetworkXNotImplemented
        If `flowG` is not directed.

    See Also
    --------
    minimum_cut
    maximum_flow
    MinimumCut

    Notes
    -----
    By the max-flow/min-cut theorem, a partition ``(S, T)`` with ``_s in S`` and
    ``_t in T`` is a minimum cut exactly when no arc with spare residual
    capacity leaves ``S`` in the residual network of a maximum flow. The arcs
    with spare residual capacity induce a digraph whose strongly connected
    components must stay together. Contracting them leaves a DAG, and the legal
    source sides are precisely the closed sets of this DAG that contain the
    source component but not the sink component. Components disconnected from
    both source and sink are free, producing up to ``2**k`` distinct cuts.

    Enumeration is exact for integer capacities. Floating-point capacities are
    affected by round-off; the `tol` parameter defines which residual arcs are
    considered saturated, and the resulting enumeration should be treated as
    approximate whenever ``tol > 0``.
    """
    if flowG.is_multigraph():
        raise nx.NetworkXError("MultiGraph and MultiDiGraph not supported (yet).")
    if not flowG.is_directed():
        raise nx.NetworkXNotImplemented("only DiGraph is supported.")
    if _s == _t:
        raise nx.NetworkXError("source and sink are the same node")
    if limit is not None and not (isinstance(limit, int) and limit > 0):
        raise nx.NetworkXError("limit must be a positive integer or None")
    if tol is not None and tol < 0:
        raise nx.NetworkXError("tol must be non-negative or None")

    if flow_func is None:
        if kwargs:
            raise nx.NetworkXError(
                "You have to explicitly set a flow_func if"
                " you need to pass parameters via kwargs."
            )
        flow_func = default_flow_func
    if not callable(flow_func):
        raise nx.NetworkXError("flow_func has to be callable.")

    cap_of = _capacity_function(flowG, capacity)
    max_capacity = 0
    all_integer = True
    for u, v, attr in flowG.edges(data=True):
        if u == v:
            continue
        cap = cap_of(u, v, attr)
        if cap is None:
            continue
        if cap == float("inf"):
            # The flow algorithm raises NetworkXUnbounded if this edge lies on
            # an s-t path; otherwise it simply cannot participate in a cut.
            all_integer = False
        elif cap != cap or cap < 0:
            raise nx.NetworkXError(
                "edge capacities must be finite and non-negative"
            )
        else:
            max_capacity = max(max_capacity, cap)
            # bool is an int subclass, so boolean capacities stay exact.
            if not isinstance(cap, int):
                all_integer = False

    if tol is None:
        tol = 0.0 if all_integer else 1e-12 * max_capacity

    # The residual-reachability characterization requires a feasible maximum
    # flow (flow conservation at internal nodes). preflow_push with
    # value_only=True returns a maximum preflow and skips that conversion, so
    # it is rejected here; the other flow functions maintain feasibility while
    # running and may still be used with value_only=True.
    if flow_func is preflow_push and kwargs.get("value_only") is True:
        raise nx.NetworkXError(
            "value_only=True is not supported with preflow_push: the residual"
            " network must describe a feasible maximum flow."
        )
    R = flow_func(flowG, _s, _t, capacity=capacity, **kwargs)

    support = nx.DiGraph()
    support.__networkx_cache__ = None
    support.add_nodes_from(flowG)
    for u, v, attr in R.edges(data=True):
        if attr["capacity"] - attr["flow"] > tol:
            support.add_edge(u, v)

    condensation = nx.condensation(support)
    cs = condensation.graph["mapping"][_s]
    ct = condensation.graph["mapping"][_t]

    return AllMinimumCuts(flowG, R, condensation, cs, ct, limit)
