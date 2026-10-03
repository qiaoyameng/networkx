"""Incremental (insertion-only) maintenance of strongly connected components.

The :class:`IncrementalSCC` class maintains the strongly connected
components (SCCs) and the condensation (component DAG) of a directed graph
under node and edge insertions without recomputing the decomposition of the
whole graph on every update.

See Also
--------
strongly_connected_components
condensation
"""

import networkx as nx

__all__ = ["IncrementalSCC"]


class IncrementalSCC:
    r"""Maintain SCCs and the condensation under node/edge insertions.

    The maintainer owns a private copy of the input graph; mutating the
    original graph after construction never affects the maintained state.
    Use :meth:`check_source` to detect such external mutation, or
    :meth:`verify` to check internal consistency.

    Parameters
    ----------
    G : NetworkX DiGraph
        The initial graph. Undirected graphs and multigraphs are not
        supported.

    Attributes
    ----------
    last_visit_count : int
        Number of condensation vertices inspected by the most recent
        structural edge insertion. Intra-component insertions (including
        duplicate edges and self loops) perform no search and report zero.

    Notes
    -----
    **Component identifiers.** Every SCC has an integer id. Ids of
    components that are untouched by an insertion remain valid forever.
    When components merge, exactly one old id survives: the smallest id
    among the merged components. Every other merged id becomes a *dead
    alias* that :meth:`component_of` resolves to the surviving id. Dead
    ids are never reused. A newly inserted isolated node receives a fresh
    id larger than any previously issued id.

    **Insertion rule.** Adding ``(u, v)``:

    * If ``u`` and ``v`` are in the same SCC nothing changes.
    * Otherwise the condensation gains an arc ``cu -> cv``. If that arc
      creates no cycle only the arc is added; the set of SCCs is
      unchanged and the work is local to the search around ``cu`` and
      ``cv``.
    * If the arc creates a cycle, precisely the components on
      condensation paths between the two newly connected SCCs are merged.

    Examples
    --------
    >>> G = nx.path_graph(3, create_using=nx.DiGraph)
    >>> scc = IncrementalSCC(G)
    >>> scc.number_of_components()
    3
    >>> scc.add_edge(2, 0)
    'merged'
    >>> scc.number_of_components()
    1
    """

    def __init__(self, G):
        if not G.is_directed() or G.is_multigraph():
            raise nx.NetworkXNotImplemented(
                "IncrementalSCC requires a directed, non-multi NetworkX graph"
            )
        self._adj = {n: set(G.neighbors(n)) for n in G}
        self._pred = {n: set() for n in G}
        for u, v in G.edges:
            self._pred[v].add(u)

        self._members = {}
        self._node_comp = {}
        initial = nx.condensation(G)
        mapping = initial.graph["mapping"]
        for node, cid in mapping.items():
            self._node_comp[node] = cid
            self._members.setdefault(cid, set()).add(node)

        self._cadj = {cid: set(initial.successors(cid)) for cid in initial}
        self._cpred = {cid: set(initial.predecessors(cid)) for cid in initial}

        self._next_id = len(self._members)
        self._alias = {}

        self.last_visit_count = 0
        self.total_visit_count = 0

        self._source = G
        self._source_nodes = frozenset(G)
        self._source_edges = frozenset(G.edges)
        self._source_size = G.size()

    # ------------------------------------------------------------------
    # Insertions
    # ------------------------------------------------------------------
    def add_node(self, node):
        """Add one isolated node if not already present.

        Returns ``True`` if the node was new (and received a fresh
        component id), ``False`` otherwise.
        """
        if node in self._node_comp:
            return False
        cid = self._next_id
        self._next_id += 1
        self._adj[node] = set()
        self._pred[node] = set()
        self._node_comp[node] = cid
        self._members[cid] = {node}
        self._cadj[cid] = set()
        self._cpred[cid] = set()
        return True

    def add_nodes_from(self, nodes):
        """Add several isolated nodes. Returns the number of new nodes."""
        added = 0
        for node in nodes:
            added += self.add_node(node)
        return added

    def add_edge(self, u, v):
        """Insert the edge ``u -> v``, creating endpoints as needed.

        Returns
        -------
        str
            One of:

            * ``"internal"``: endpoints are already in the same SCC
              (this covers duplicate edges and self loops); no search.
            * ``"acyclic"``: the edge joins two SCCs without creating a
              cycle; only the condensation arc is added.
            * ``"merged"``: the edge created a cycle; the affected SCCs
              were merged.
        """
        self.add_node(u)
        self.add_node(v)

        cu = self._node_comp[u]
        cv = self._node_comp[v]
        if cu == cv:
            if v not in self._adj[u]:
                self._adj[u].add(v)
                self._pred[v].add(u)
            self.last_visit_count = 0
            return "internal"

        self._adj[u].add(v)
        self._pred[v].add(u)

        if cv in self._cadj[cu]:
            # The condensation arc already exists: another original edge
            # across the same pair of SCCs changes nothing.
            self.last_visit_count = 0
            return "acyclic"

        visited = {"nodes": 0}
        # cu -> cv creates a cycle iff cv already reaches cu.
        closes_cycle = self._reachable(self._cadj, cv, cu, visited)
        if not closes_cycle:
            self._cadj[cu].add(cv)
            self._cpred[cv].add(cu)
            self.last_visit_count = visited["nodes"]
            self.total_visit_count += visited["nodes"]
            return "acyclic"

        # Components on old paths cv ~> x ~> cu are exactly
        # Fwd(cv) & Bwd(cu); collect it with a local two-sided search.
        merged = self._merge_region(cv, cu, visited)
        self._merge(merged, cu, cv)

        self.last_visit_count = visited["nodes"]
        self.total_visit_count += visited["nodes"]
        return "merged"

    def add_edges_from(self, ebunch):
        """Insert several edges, creating endpoints as needed.

        Returns a dict mapping status names (``"internal"``,
        ``"acyclic"``, ``"merged"``) to insertion counts.
        """
        counts = {"internal": 0, "acyclic": 0, "merged": 0}
        for edge in ebunch:
            edge = tuple(edge)
            if len(edge) < 2:
                raise nx.NetworkXError(f"Edge tuple {edge} must be a 2-tuple")
            counts[self.add_edge(edge[0], edge[1])] += 1
        return counts

    # ------------------------------------------------------------------
    # Component / condensation access
    # ------------------------------------------------------------------
    def component_of(self, node):
        """Return the current component id containing ``node``.

        Raises :exc:`networkx.NetworkXError` if ``node`` is not in the
        graph.
        """
        try:
            return self._node_comp[node]
        except KeyError:
            raise nx.NetworkXError(f"Node {node} is not in the graph") from None

    def members(self, component):
        """Return the set of original nodes in component ``component``.

        Dead (merged-away) ids are first resolved to the surviving id.
        """
        return set(self._members[self.resolve(component)])

    def components(self):
        """Return a list of ``(component_id, members_set)`` pairs."""
        return [(cid, set(members)) for cid, members in self._members.items()]

    def component_ids(self):
        """Return the set of currently live component ids."""
        return set(self._members)

    def number_of_components(self):
        """Return the number of current SCCs."""
        return len(self._members)

    def __len__(self):
        return len(self._members)

    def resolve(self, component):
        """Resolve a possibly dead component id to its current live id.

        Dead ids are the ids of components absorbed by a merge. They keep
        resolving to the surviving id (possibly through a chain), so
        cached ids never silently collide with a different component.

        Raises :exc:`networkx.NetworkXError` for an id that was never
        issued.
        """
        if component in self._members:
            return component
        if component not in self._alias:
            raise nx.NetworkXError(f"Unknown component id {component}")
        root = component
        while root in self._alias:
            root = self._alias[root]
        while component != root:
            nxt = self._alias[component]
            self._alias[component] = root
            component = nxt
        return root

    def is_merged_alias(self, component):
        """Return ``True`` if ``component`` is an id absorbed by a merge."""
        return component in self._alias

    def condensation(self):
        """Return a frozen copy of the current condensation DiGraph.

        Each condensation vertex carries a ``"members"`` attribute with
        the set of original nodes, and the graph carries a ``"mapping"``
        attribute from original nodes to component ids. Vertex labels are
        the stable component ids (which may be non-contiguous after
        merges). The returned graph is frozen; call
        :meth:`add_edge`/:meth:`add_node` to mutate the maintained state.
        """
        C = nx.DiGraph()
        C.add_nodes_from(self._members)
        C.add_edges_from(
            (u, w) for u in self._cadj for w in self._cadj[u] if u in self._members
        )
        members = {cid: set(self._members[cid]) for cid in self._members}
        nx.set_node_attributes(C, members, "members")
        C.graph["mapping"] = dict(self._node_comp)
        nx.freeze(C)
        return C

    def graph(self):
        """Return a frozen copy of the maintained DiGraph."""
        G = nx.DiGraph()
        G.add_nodes_from(self._adj)
        G.add_edges_from((u, w) for u in self._adj for w in self._adj[u])
        nx.freeze(G)
        return G

    def __iter__(self):
        return iter(self._members)

    def __contains__(self, component):
        return component in self._members

    # ------------------------------------------------------------------
    # Consistency checks
    # ------------------------------------------------------------------
    def verify(self):
        """Check internal consistency against a full NetworkX SCC run.

        Raises :exc:`networkx.NetworkXError` if the maintained SCC
        partition, node-to-component mapping, or condensation differs
        from :func:`~networkx.algorithms.components.strongly_connected_components`
        / :func:`~networkx.algorithms.components.strongly_connected.condensation`.
        """
        G = self.graph()
        expected = {
            frozenset(component)
            for component in nx.strongly_connected_components(G)
        }
        actual = {frozenset(members) for members in self._members.values()}
        if actual != expected:
            raise nx.NetworkXError(
                "Maintained SCC partition differs from strongly_connected_components"
            )
        for node, cid in self._node_comp.items():
            if node not in self._members[cid]:
                raise nx.NetworkXError(f"Stale component mapping for node {node}")
        C = self.condensation()
        if not nx.is_directed_acyclic_graph(C):
            raise nx.NetworkXError("Maintained condensation contains a cycle")
        expected_arcs = {
            (self._node_comp[u], self._node_comp[v])
            for u, v in G.edges
            if self._node_comp[u] != self._node_comp[v]
        }
        if set(C.edges) != expected_arcs:
            raise nx.NetworkXError("Maintained condensation arcs are stale")
        return True

    def check_source(self):
        """Raise if the original graph passed to the constructor changed.

        The maintainer keeps a private copy of the graph and is therefore
        unaffected by external mutation; this method detects that the
        caller modified the source graph so that such edits are not
        silently ignored.
        """
        G = self._source
        if (
            frozenset(G) != self._source_nodes
            or frozenset(G.edges) != self._source_edges
            or G.size() != self._source_size
        ):
            raise nx.NetworkXError(
                "The source graph passed to IncrementalSCC was modified after "
                "construction; construct a new maintainer from the updated graph "
                "or apply changes through add_node/add_edge."
            )
        return True

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _reachable(self, neighbors, source, target, visited):
        """Whether ``target`` is reachable from ``source`` via ``neighbors``.

        The search terminates as soon as ``target`` is found and only
        explores the region around the endpoints.
        """
        if source == target:
            return True
        seen = {source}
        stack = [source]
        while stack:
            node = stack.pop()
            visited["nodes"] += 1
            for nxt in neighbors[node]:
                if nxt == target:
                    visited["nodes"] += 1
                    return True
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return False

    def _merge_region(self, cv, cu, visited):
        """Return ``Fwd(cv) & Bwd(cu)`` using a local two-sided search.

        The forward search from ``cv`` and backward search from ``cu``
        expand in alternation. Whichever side exhausts first has visited
        its full reachable region; the other side then prunes every
        vertex outside that completed set, so unrelated parts of the
        condensation are never scanned when only a local region merges.
        """
        seen_f = {cv}
        seen_b = {cu}
        front_f = [cv]
        front_b = [cu]
        merged = {cv, cu}
        # Expand both sides without pruning until one side exhausts
        # naturally; only then is its visited set a complete closure and
        # safe to use as a bound for the remaining search.
        while front_f and front_b:
            if len(front_f) <= len(front_b):
                node = front_f.pop()
                visited["nodes"] += 1
                for nxt in self._cadj[node] - seen_f:
                    seen_f.add(nxt)
                    front_f.append(nxt)
                    if nxt in seen_b:
                        merged.add(nxt)
            else:
                node = front_b.pop()
                visited["nodes"] += 1
                for nxt in self._cpred[node] - seen_b:
                    seen_b.add(nxt)
                    front_b.append(nxt)
                    if nxt in seen_f:
                        merged.add(nxt)

        if front_f:
            # Backward closure is complete: stay inside seen_b.
            stack = front_f
            seen = seen_f
            neighbors = self._cadj
            bound = seen_b
        else:
            stack = front_b
            seen = seen_b
            neighbors = self._cpred
            bound = seen_f
        while stack:
            node = stack.pop()
            visited["nodes"] += 1
            for nxt in neighbors[node] - seen:
                seen.add(nxt)
                if nxt in bound:
                    merged.add(nxt)
                    stack.append(nxt)
        return merged

    def _merge(self, merged, cu, cv):
        survivor = min(merged)
        new_members = set()
        for cid in merged:
            new_members |= self._members[cid]

        out_neighbors = set()
        in_neighbors = set()
        for cid in merged:
            out_neighbors |= self._cadj[cid] - merged
            in_neighbors |= self._cpred[cid] - merged

        for cid in merged:
            if cid == survivor:
                continue
            self._alias[cid] = survivor
            del self._members[cid]
            del self._cadj[cid]
            del self._cpred[cid]

        self._members[survivor] = new_members
        for node in new_members:
            self._node_comp[node] = survivor

        self._cadj[survivor] = out_neighbors
        self._cpred[survivor] = in_neighbors
        for other in out_neighbors:
            self._cpred[other] = (self._cpred[other] - merged) | {survivor}
        for other in in_neighbors:
            self._cadj[other] = (self._cadj[other] - merged) | {survivor}
