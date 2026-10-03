"""Incremental (insertion-only) maintenance of strongly connected components.

This module provides :class:`IncrementalSCC`, a maintainer for the strongly
connected components (SCCs) and the condensation (component DAG) of a directed
graph under node and edge insertions only.  Nodes and edges are never removed.

Overview
--------

The maintainer keeps, at all times:

* ``cid_of_node`` -- the current component id of every node;
* ``members`` -- the set of nodes belonging to every live component;
* ``succs`` / ``preds`` -- the adjacency dictionaries of the condensation;
* ``tombs`` -- the set of component ids that have disappeared through merging.

Inserting a node creates a fresh singleton component.  Inserting an edge
``(u, v)`` has three structural outcomes:

1. ``u`` and ``v`` are already in the same SCC: the structure is unchanged.
2. The edge runs between two SCCs without a path ``cid(v) -> cid(u)`` in the
   condensation: the edge is an acyclic cross-component edge, and one
   directed condensation edge is recorded.
3. A path ``cid(v) -> cid(u)`` already exists: the new edge closes a cycle of
   components.  Every component that lies on such a path is merged into one
   new component; condensation edges incident on the merged components are
   redirected to the new component and every contained node is remapped.

Component ids and caching
-------------------------

Component ids are integers.  They are *stable*: the id of a component that is
not involved in a merge never changes.  A merge never reuses an old id;
instead a brand new id is allocated and all old ids of the merged components
are tombstoned.  Looking up a tombstoned id with
:meth:`IncrementalSCC.component` raises :class:`StaleComponentError`;
:meth:`IncrementalSCC.resolve_cid` follows the tombstone chain to the current
successor of an old id, so caches can detect exactly when they hold a
disappeared component instead of silently mutating foreign state.

Protecting the underlying graph
--------------------------------

The maintainer takes ownership of the input graph and freezes it, so that
direct mutation through the graph API raises ``NetworkXError``.  Graph *views*
are rejected at construction (their storage is shared with another graph).
Internal updates go through low-level bypass helpers that do not go through
the frozen API.  :meth:`IncrementalSCC.verify` rebuilds the SCCs and the
condensation with the standard NetworkX algorithms and compares them against
the maintained state; call it to detect any out-of-band structural change.
"""

from collections import Counter

import networkx as nx

__all__ = [
    "IncrementalSCC",
    "StaleComponentError",
    "UnknownComponentError",
]

# Status strings returned by :meth:`IncrementalSCC.add_edge`.
EDGE_DUPLICATE = "duplicate"
EDGE_INTERNAL = "internal"
EDGE_ACYCLIC = "acyclic"
EDGE_MERGED = "merged"
EDGE_REDUNDANT = "redundant"


class IncrementalSCCError(Exception):
    """Base class for errors raised by :class:`IncrementalSCC`."""


class StaleComponentError(IncrementalSCCError, KeyError):
    """Raised when a tombstoned (merged-away) component id is used.

    The offending id is available as ``args[0]`` and its current successor
    component id (following the full tombstone chain) as ``successor``.
    """

    def __init__(self, cid, successor):
        super().__init__(cid)
        self.cid = cid
        self.successor = successor

    def __str__(self):
        return (
            f"component {self.cid!r} no longer exists; it was merged into "
            f"component {self.successor!r}. Use resolve_cid() to follow old ids."
        )


class UnknownComponentError(IncrementalSCCError, KeyError):
    """Raised when a component id has never been allocated."""


class IncrementalSCC:
    """Insertion-only maintainer for SCCs and the condensation of a DiGraph.

    Parameters
    ----------
    G : networkx.DiGraph
        The initial graph.  It must be a *concrete* ``DiGraph`` (not a graph
        view such as ``G.subgraph(...)`` or ``G.reverse()``), because the
        maintainer takes ownership of the graph's storage.  The graph is
        frozen so that later mutation through ``G.add_*``/``G.remove_*``
        raises :exc:`networkx.NetworkXError`.
    freeze_input : bool, default True
        If ``False``, the graph is not frozen; the caller is then responsible
        for never mutating it directly.  :meth:`verify` still detects any
        divergence.

    Attributes
    ----------
    graph : networkx.DiGraph
        The (frozen) owned graph.  It is safe to read but must not be mutated
        directly.

    Examples
    --------
    >>> import networkx as nx
    >>> m = IncrementalSCC(nx.DiGraph([(0, 1), (1, 2)]))
    >>> m.number_of_components()
    3
    >>> m.add_edge(2, 0)
    'merged'
    >>> m.number_of_components()
    1
    >>> m.component_of(1) == m.component_of(0)
    True

    Notes
    -----
    The bidirectional reachability probe on the condensation only explores
    components near the inserted edge: for an acyclic insertion that connects
    two regions that cannot reach each other, it does not scan the whole
    condensation, so its work does not grow with the total graph size.
    """

    def __init__(self, G, *, freeze_input=True):
        if type(G) is not nx.DiGraph:
            if not isinstance(G, nx.DiGraph):
                raise nx.NetworkXNotImplemented(
                    "IncrementalSCC requires a DiGraph (directed graph)."
                )
            raise TypeError(
                "IncrementalSCC requires a concrete DiGraph, not a view or "
                f"subclass ({type(G).__name__}); pass G.copy() instead."
            )
        if "_graph" in G.__dict__:  # generic_graph_view / reverse_view / subgraph_view
            raise TypeError(
                "IncrementalSCC cannot take ownership of a graph view; pass "
                "an independent DiGraph (e.g. view.copy()) instead."
            )
        self.graph = G
        self.freeze_input = freeze_input
        if freeze_input:
            nx.freeze(self.graph)

        # node -> current live component id
        self.cid_of_node = {}
        # live component id -> set of nodes
        self.members = {}
        # live component id -> set of live component ids (condensation)
        self.succs = {}
        self.preds = {}
        # old component id -> id of the component it was first merged into
        self._tomb_next = {}

        # Seed the partition from a full Tarjan pass.  The enumeration order
        # matches ``nx.strongly_connected_components``, which is also the id
        # assignment used by ``nx.condensation``.
        for cid, component in enumerate(nx.strongly_connected_components(self.graph)):
            self.members[cid] = set(component)
            self.succs[cid] = set()
            self.preds[cid] = set()
            for node in component:
                self.cid_of_node[node] = cid
        self._next_cid = len(self.members)

        # Build the condensation edges directly from graph adjacency.
        for node, cid in self.cid_of_node.items():
            targets = self.succs[cid]
            for w in self.graph.succ[node]:
                other = self.cid_of_node[w]
                if other != cid:
                    targets.add(other)
        for cid, neighbors in self.succs.items():
            for other in neighbors:
                self.preds[other].add(cid)

        # Number of condensation adjacency entries examined by reachability
        # probes since construction (see ``probe_visits``).
        self._probe_visits = 0

    # ------------------------------------------------------------------ utils
    def _raw_add_node(self, node):
        """Insert a node bypassing the frozen public API."""
        if node not in self.graph._node:
            self.graph._node[node] = self.graph.node_dict_factory()
            self.graph._adj[node] = self.graph.adjlist_outer_dict_factory()
            self.graph._pred[node] = self.graph.adjlist_outer_dict_factory()

    def _raw_add_edge(self, u, v, attr=None):
        """Insert edge ``u -> v`` bypassing the frozen public API."""
        inner_factory = self.graph.adjlist_inner_dict_factory
        if v not in self.graph._adj[u]:
            self.graph._adj[u][v] = inner_factory()
        if u not in self.graph._pred[v]:
            # DiGraph shares the single edge attribute dict between _succ/_pred
            self.graph._pred[v][u] = self.graph._adj[u][v]
        if attr:
            self.graph._adj[u][v].update(attr)

    def _clear_networkx_cache(self):
        cache = self.graph.__dict__.get("__networkx_cache__")
        if cache is not None:
            cache.clear()

    # -------------------------------------------------------------- mutation
    def add_node(self, node):
        """Insert a node, creating a fresh singleton component.

        Returns
        -------
        bool
            ``True`` if the node was newly added, ``False`` if it already
            existed (the existing structure is then untouched).
        """
        if node in self.cid_of_node:
            return False
        cid = self._alloc_cid()
        self._raw_add_node(node)
        self.cid_of_node[node] = cid
        self.members[cid] = {node}
        self.succs[cid] = set()
        self.preds[cid] = set()
        self._clear_networkx_cache()
        return True

    def _alloc_cid(self):
        cid = self._next_cid
        self._next_cid += 1
        return cid

    def add_edge(self, u, v, **attr):
        """Insert an edge ``u -> v`` and update SCCs and the condensation.

        Missing endpoints are added automatically, as with ordinary NetworkX
        graphs.

        Returns
        -------
        str
            One of

            * ``"duplicate"``  -- the edge already existed in the graph;
            * ``"internal"``   -- new edge, endpoints already in the same SCC;
            * ``"redundant"``  -- new edge between two SCCs already joined by
              a condensation edge in this direction (parallel graph edge);
            * ``"acyclic"``    -- new edge, no cycle of components; one
              condensation edge is added;
            * ``"merged"``     -- the edge closed a cycle of components; all
              components on the cycle paths were merged into a new component.
        """
        u_existed = u in self.cid_of_node
        v_existed = v in self.cid_of_node
        if not u_existed:
            self.add_node(u)
        if not v_existed:
            self.add_node(v)

        if v in self.graph._adj[u] and u in self.graph._pred[v]:
            # An existing edge changes no structure but NetworkX updates
            # edge attributes on repeated insertion.
            if attr:
                self.graph._adj[u][v].update(attr)
                self._clear_networkx_cache()
            return EDGE_DUPLICATE
        self._raw_add_edge(u, v, attr)
        self._clear_networkx_cache()

        cu = self.cid_of_node[u]
        cv = self.cid_of_node[v]
        if cu == cv:
            return EDGE_INTERNAL
        if cv in self.succs[cu]:
            return EDGE_REDUNDANT

        # A cycle of components appears iff cv can already reach cu.
        reaches, forward, backward = self._bidirectional_reach(cv, cu)
        if not reaches:
            self.succs[cu].add(cv)
            self.preds[cv].add(cu)
            return EDGE_ACYCLIC
        region = self._merge_region(cv, cu, forward, backward)
        self._merge(region)
        return EDGE_MERGED

    def add_nodes_from(self, nodes):
        """Insert many nodes.

        Returns
        -------
        int
            Number of newly created nodes (already-present nodes are skipped).
        """
        return sum(1 for node in nodes if self.add_node(node))

    def add_edges_from(self, ebunch):
        """Insert many edges, updating after each one.

        Missing endpoints are added automatically.  A :class:`Counter` is
        returned counting edges by the status strings of :meth:`add_edge`
        (``"duplicate"``, ``"internal"``, ``"redundant"``, ``"acyclic"``,
        ``"merged"``).
        """
        result = Counter()
        for edge in ebunch:
            if len(edge) == 2:
                u, v = edge
                result[self.add_edge(u, v)] += 1
            else:  # 3-tuple (u, v, data) -- data must be an attr dict
                u, v, data = edge
                result[self.add_edge(u, v, **data)] += 1
        return result

    # --------------------------------------------------------- core algorithm
    def _bidirectional_reach(self, source, target):
        """Test reachability ``source -> target`` on the condensation.

        A bidirectional breadth-first search is used.  Every condensation
        adjacency set examined is counted in ``_probe_visits``.  The search
        only explores components in the forward region of ``source`` and the
        backward region of ``target``; when the two regions never meet, the
        rest of the condensation is not scanned.

        Returns
        -------
        (bool, set, set)
            ``(reaches, forward_visited, backward_visited)``.  If ``reaches``
            is ``True`` the two visited sets together contain enough of the
            condensation for :meth:`_merge_region` to be computed without a
            second search.
        """
        if source == target:
            return True, {source}, {target}

        forward_seen = {source}
        backward_seen = {target}
        forward_frontier = [source]
        backward_frontier = [target]

        while forward_frontier and backward_frontier:
            if len(forward_frontier) <= len(backward_frontier):
                new_frontier = []
                for cid in forward_frontier:
                    self._probe_visits += 1
                    for nbr in self.succs[cid]:
                        if nbr not in forward_seen:
                            if nbr in backward_seen:
                                forward_seen.add(nbr)
                                return True, forward_seen, backward_seen
                            forward_seen.add(nbr)
                            new_frontier.append(nbr)
                forward_frontier = new_frontier
            else:
                new_frontier = []
                for cid in backward_frontier:
                    self._probe_visits += 1
                    for nbr in self.preds[cid]:
                        if nbr not in backward_seen:
                            if nbr in forward_seen:
                                backward_seen.add(nbr)
                                return True, forward_seen, backward_seen
                            backward_seen.add(nbr)
                            new_frontier.append(nbr)
                backward_frontier = new_frontier
        return False, forward_seen, backward_seen

    def _merge_region(self, source, target, forward, backward):
        """Return every component lying on a path ``source -> target``.

        These are exactly the components reachable from ``source`` that can
        also reach ``target``.  When the bidirectional probe of
        :meth:`add_edge` found that a path exists, its forward/backward
        visited sets already meet; the sets are completed here so the result
        is exact (the probe may stop early once the frontiers touch).
        """
        # Complete forward search from source over the condensation.
        stack = list(forward)
        seen_forward = set(forward)
        while stack:
            cid = stack.pop()
            for nbr in self.succs[cid]:
                if nbr not in seen_forward:
                    seen_forward.add(nbr)
                    stack.append(nbr)
        # Complete backward search from target.
        stack = list(backward)
        seen_backward = set(backward)
        while stack:
            cid = stack.pop()
            for nbr in self.preds[cid]:
                if nbr not in seen_backward:
                    seen_backward.add(nbr)
                    stack.append(nbr)
        region = seen_forward & seen_backward
        # Defensive: the endpoints are always part of the region.
        region.add(source)
        region.add(target)
        return region

    def _merge(self, region):
        """Merge all live component ids in ``region`` into one new component."""
        new_cid = self._alloc_cid()
        region = set(region)

        new_members = set()
        for cid in region:
            new_members |= self.members[cid]
            del self.members[cid]

        new_succs = set()
        new_preds = set()
        for cid in region:
            new_succs |= self.succs[cid]
            new_preds |= self.preds[cid]
        new_succs -= region
        new_preds -= region

        # Redirect incident edges of every merged component to the new one.
        for nbr in new_succs:
            incoming = self.preds[nbr]
            incoming.difference_update(region)
            incoming.add(new_cid)
        for nbr in new_preds:
            outgoing = self.succs[nbr]
            outgoing.difference_update(region)
            outgoing.add(new_cid)

        for cid in region:
            del self.succs[cid]
            del self.preds[cid]
            # The old id now points at the component that replaced it.
            self._tomb_next[cid] = new_cid
        for node in new_members:
            self.cid_of_node[node] = new_cid

        self.members[new_cid] = new_members
        self.succs[new_cid] = new_succs
        self.preds[new_cid] = new_preds

    # ---------------------------------------------------------------- queries
    @property
    def probe_visits(self):
        """Condensation adjacency sets examined by acyclic reachability probes."""
        return self._probe_visits

    def reset_probe_visits(self):
        """Reset :attr:`probe_visits` to zero and return the previous value."""
        previous = self._probe_visits
        self._probe_visits = 0
        return previous

    def component_ids(self):
        """Return the set of currently live component ids."""
        return set(self.members)

    def number_of_components(self):
        """Return the number of live components."""
        return len(self.members)

    def component_of(self, node):
        """Return the current component id of ``node``.

        Unlike methods that take a component id, this never raises
        :class:`StaleComponentError`: a node is always mapped to its current
        component.
        """
        try:
            return self.cid_of_node[node]
        except KeyError:
            raise nx.NetworkXError(f"node {node!r} is not in the graph") from None

    def is_stale(self, cid):
        """Return whether ``cid`` was allocated but later merged away."""
        return cid in self._tomb_next

    def is_known(self, cid):
        """Return whether ``cid`` is a live or tombstoned allocated id."""
        return cid in self.members or cid in self._tomb_next

    def resolve_cid(self, cid):
        """Follow tombstone chains; return the live successor of ``cid``.

        Raises
        ------
        UnknownComponentError
            If ``cid`` has never been allocated.
        """
        seen = set()
        current = cid
        while current in self._tomb_next:
            if current in seen:  # defensive; chains cannot cycle by construction
                raise IncrementalSCCError("tombstone chain contains a cycle")
            seen.add(current)
            current = self._tomb_next[current]
        if current not in self.members:
            raise UnknownComponentError(cid)
        return current

    def component(self, cid):
        """Return the set of nodes of the live component ``cid``.

        Raises
        ------
        StaleComponentError
            If ``cid`` has been merged away.  The exception carries the
            current successor id.
        UnknownComponentError
            If ``cid`` has never been allocated.
        """
        if cid in self._tomb_next:
            raise StaleComponentError(cid, self.resolve_cid(cid))
        try:
            return set(self.members[cid])
        except KeyError:
            raise UnknownComponentError(cid) from None

    def condensation_successors(self, cid):
        """Live components directly reachable from the live component ``cid``."""
        if cid in self._tomb_next:
            raise StaleComponentError(cid, self.resolve_cid(cid))
        if cid not in self.members:
            raise UnknownComponentError(cid)
        return set(self.succs[cid])

    def condensation_predecessors(self, cid):
        """Live components that directly reach the live component ``cid``."""
        if cid in self._tomb_next:
            raise StaleComponentError(cid, self.resolve_cid(cid))
        if cid not in self.members:
            raise UnknownComponentError(cid)
        return set(self.preds[cid])

    def condensation_graph(self):
        """Return an independent frozen ``DiGraph`` of the condensation.

        Nodes are the live component ids; edges are the condensation edges.
        The returned graph is a fresh snapshot and may be read or frozen
        without affecting the maintainer.
        """
        C = nx.DiGraph()
        C.add_nodes_from(self.members)
        C.add_edges_from(
            (cid, nbr) for cid, nbrs in self.succs.items() for nbr in nbrs
        )
        return nx.freeze(C)

    # -------------------------------------------------------------- checking
    def verify(self):
        """Check the maintained state against full NetworkX recomputation.

        Returns
        -------
        bool
            ``True`` if the maintained partition, node map and condensation
            agree with ``nx.strongly_connected_components`` and
            ``nx.condensation``.

        Raises
        ------
        AssertionError
            On any discrepancy (including direct external mutation of the
            owned graph).
        """
        graph = self.graph

        # Partition and node map agree with a full SCC run.
        expected_components = list(nx.strongly_connected_components(graph))
        expected_sets = sorted((sorted(c) for c in expected_components), key=sorted)
        got_sets = sorted(
            (sorted(nodes) for nodes in self.members.values()), key=sorted
        )
        assert got_sets == expected_sets, "SCC partition differs from full SCC run"

        # Compare the partition induced by cid_of_node against the reference
        # partition (ids are stable but not necessarily Tarjan ids after a
        # merge, so compare node sets, not ids).
        induced = {}
        for node, cid in self.cid_of_node.items():
            induced.setdefault(cid, set()).add(node)
        assert (
            sorted((sorted(s) for s in induced.values()), key=sorted)
            == expected_sets
        )
        assert set(self.cid_of_node) == set(graph.nodes), "node map mismatch"

        # Condensation edges agree with the reference condensation.
        reference = nx.condensation(graph)
        expected_edges = {tuple(sorted(edge)) for edge in reference.edges}

        # Match each maintained component to the reference component with the
        # same node set, then translate maintained condensation edges.
        ref_id_by_set = {
            tuple(sorted(component)): cid
            for cid, component in enumerate(expected_components)
        }
        live_to_ref = {}
        for cid, nodes in self.members.items():
            key = tuple(sorted(nodes))
            assert key in ref_id_by_set, f"component {cid} is not an SCC"
            live_to_ref[cid] = ref_id_by_set[key]

        maintained_edges = {
            tuple(sorted((live_to_ref[a], live_to_ref[b])))
            for a, nbrs in self.succs.items()
            for b in nbrs
        }
        assert maintained_edges == expected_edges, "condensation edges differ"

        # No self loops; preds/succs symmetric; tombstones never live.
        for cid, nbrs in self.succs.items():
            assert cid not in nbrs, "condensation self loop"
            for nbr in nbrs:
                assert cid in self.preds[nbr], "asymmetric condensation edge"
        for cid in self._tomb_next:
            assert cid not in self.members, "tombstoned id still live"
            assert self.resolve_cid(cid) in self.members, "dangling tombstone chain"
        assert set(self.members) == set(self.succs) == set(self.preds)
        return True
