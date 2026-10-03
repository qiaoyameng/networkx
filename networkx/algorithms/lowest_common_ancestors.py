"""Algorithms for finding the lowest common ancestor of trees and DAGs."""

from collections import defaultdict
from collections.abc import Mapping, Set
from itertools import combinations_with_replacement

import networkx as nx
from networkx.utils import UnionFind, arbitrary_element, not_implemented_for

__all__ = [
    "all_pairs_lowest_common_ancestor",
    "all_pairs_lowest_common_ancestors",
    "tree_all_pairs_lowest_common_ancestor",
    "lowest_common_ancestor",
    "DAGLCAIndex",
]


class DAGLCAIndex:
    """Reusable index of ancestor sets for repeated set-valued LCA queries.

    A DAG may have more than one lowest common ancestor of two nodes.  This
    index precomputes the full set of ancestors of every node (including the
    node itself) once, so that answering many pairs does not require
    re-enumerating the ancestors -- or any paths -- for each query.  "Lowest"
    is defined purely by the reachability order: a common ancestor ``a`` is a
    lowest common ancestor when no *other* common ancestor is a proper
    descendant of ``a``.  Node orderability is never assumed.

    Parameters
    ----------
    G : NetworkX directed graph

    Raises
    ------
    NetworkXPointlessConcept
        If `G` is null.
    NetworkXNotImplemented
        If `G` is undirected.
    NetworkXError
        If `G` is not a DAG.

    Examples
    --------
    Two incomparable lowest common ancestors ``a`` and ``b`` sit below a
    higher common ancestor ``r``; only the two lower ones are returned:

    >>> G = nx.DiGraph(
    ...     [("r", "a"), ("r", "b"), ("a", "x"), ("b", "x"),
    ...      ("a", "y"), ("b", "y")]
    ... )
    >>> index = nx.DAGLCAIndex(G)
    >>> sorted(index.lowest_common_ancestors("x", "y"))
    ['a', 'b']

    Reuse the same index across several queries:

    >>> dict(index([("x", "y"), ("x", "x")])) == {
    ...     ("x", "y"): {"a", "b"}, ("x", "x"): {"x"}}
    True

    **Staleness and version invalidation.** The index stores a checksum of
    the graph's node and edge sets.  Every query verifies the checksum: if
    the graph was structurally mutated (nodes or edges added, removed, or
    replaced) without rebuilding, a :exc:`NetworkXError` is raised instead
    of returning stale results.  After such a mutation, call
    :meth:`rebuild` (the explicit invalidation rule); a different graph
    always requires a new index.  Node/edge attributes do not affect LCA
    results and do not invalidate the index.

    >>> G.add_edge("z", "x")
    >>> G.add_edge("z", "y")
    >>> index.lowest_common_ancestors("x", "y")
    Traceback (most recent call last):
    ...
    networkx.exception.NetworkXError: stale LCA index: ...
    >>> _ = index.rebuild()
    >>> sorted(index.lowest_common_ancestors("x", "y"))
    ['a', 'b', 'z']

    See Also
    --------
    all_pairs_lowest_common_ancestors
    lowest_common_ancestor
    """

    def __init__(self, G):
        self.G = G
        self.ancestor_sets = {}
        self.rebuild()

    def rebuild(self):
        """(Re)build the ancestor index from the current state of the graph.

        Call this after mutating the graph (adding or removing nodes or
        edges).  This is the supported way of invalidating a stale index.
        """
        G = self.G
        if not G.is_directed():
            raise nx.NetworkXNotImplemented(
                "LCA only defined on directed graphs."
            )
        if not nx.is_directed_acyclic_graph(G):
            raise nx.NetworkXError("LCA only defined on directed acyclic graphs.")
        if len(G) == 0:
            raise nx.NetworkXPointlessConcept("LCA meaningless on null graphs.")

        # Full ancestor sets, including the node itself, accumulated in a
        # single topological sweep (no path enumeration).
        ancestor_sets = {}
        for node in nx.topological_sort(G):
            ancestors = {node}
            for parent in G.predecessors(node):
                ancestors |= ancestor_sets[parent]
            ancestor_sets[node] = ancestors
        self.ancestor_sets = ancestor_sets
        self._signature = self._checksum()
        return self

    def _checksum(self):
        # Cheap version token over the node and edge sets.  Nodes must be
        # hashable in NetworkX; orderability is not required.  Edge keys are
        # included so parallel edges in multigraphs are distinguished.
        G = self.G
        node_checksum = 0
        edge_checksum = 0
        for node in G:
            node_checksum += hash(node)
        edges = G.edges(keys=True) if G.is_multigraph() else G.edges()
        for edge in edges:
            edge_checksum += hash(edge)
        return len(G), G.number_of_edges(), node_checksum, edge_checksum

    def _require_fresh(self):
        if self._checksum() != self._signature:
            raise nx.NetworkXError(
                "stale LCA index: the graph has changed since the index was "
                "built; call rebuild() or create a new DAGLCAIndex"
            )

    def lowest_common_ancestors(self, node1, node2):
        """Return the set of all lowest common ancestors of two nodes.

        Parameters
        ----------
        node1, node2 : nodes in the graph.

        Returns
        -------
        set
            Every common ancestor that has no other common ancestor among its
            proper descendants.  Includes the queried node for a self-pair;
            empty if the nodes have no common ancestor.

        Raises
        ------
        NodeNotFound
            If either node is not in the graph.
        NetworkXError
            If the index is stale (the graph was mutated without a rebuild).
        """
        self._require_fresh()
        if node1 not in self.G or node2 not in self.G:
            missing = {n for n in (node1, node2) if n not in self.G}
            raise nx.NodeNotFound(f"Node(s) {missing} not in G.")
        return self._lowest_common_ancestors(node1, node2)

    def _lowest_common_ancestors(self, node1, node2):
        # Assumes the index is fresh and both nodes belong to the graph.
        ancestors1 = self.ancestor_sets[node1]
        ancestors2 = self.ancestor_sets[node2]
        common = ancestors1 & ancestors2
        if not common:
            return set()
        # Keep precisely the common ancestors with no other common ancestor
        # among their immediate descendants.  This uses only the edge
        # relation; it never sorts nodes or compares "topological numbers".
        lowest = set()
        for node in common:
            for child in self.G.successors(node):
                if child in common:
                    break
            else:
                lowest.add(node)
        return lowest

    def all_pairs(self, pairs=None):
        """Yield ``((u, v), lca_set)`` for every requested pair.

        Parameters
        ----------
        pairs : iterable of pairs of nodes, optional
            The pairs of interest.  If None, every unordered pair of nodes
            with repetition is used (including self-pairs).  Duplicate pairs
            are yielded once.

        Raises
        ------
        NodeNotFound
            If a pair contains a node that is not in the graph.
        NetworkXError
            If the index is stale (the graph was mutated without a rebuild).
        """
        self._require_fresh()
        if pairs is None:
            pairs = combinations_with_replacement(self.G, 2)
        else:
            pairs = dict.fromkeys(pairs)
            nodeset = set(self.G)
            for pair in pairs:
                missing = set(pair) - nodeset
                if missing:
                    raise nx.NodeNotFound(
                        f"Node(s) {missing} from pair {pair} not in G."
                    )
        for node1, node2 in pairs:
            yield (node1, node2), self._lowest_common_ancestors(node1, node2)

    __call__ = all_pairs


@not_implemented_for("undirected")
@nx._dispatchable
def all_pairs_lowest_common_ancestors(G, pairs=None):
    """Yield the set of all lowest common ancestors for pairs of nodes.

    Unlike :func:`all_pairs_lowest_common_ancestor`, which returns a single
    (arbitrary) lowest common ancestor per pair, this yields the *complete*
    set of lowest common ancestors.  A node is a lowest common ancestor of
    ``u`` and ``v`` if it is a common ancestor and has no other common
    ancestor among its proper descendants.  A DAG can therefore have more
    than one lowest common ancestor for a pair.  Nodes need not be orderable
    and no topological numbering is used.

    Parameters
    ----------
    G : NetworkX directed graph

    pairs : iterable of pairs of nodes, optional (default: all pairs)
        The pairs of nodes of interest.
        If None, yields results for all unordered pairs with repetition,
        i.e. ``(u, v)`` with ``u`` appearing before ``v`` in ``G`` but not
        both ``(u, v)`` and ``(v, u)``.

    Yields
    ------
    ((node1, node2), lca_set) : 2-tuple
        ``lca_set`` is the set of all lowest common ancestors.  It contains
        the node itself for a self-pair and is empty when the pair has no
        common ancestor.

    Raises
    ------
    NetworkXPointlessConcept
        If `G` is null.
    NetworkXNotImplemented
        If `G` is undirected.
    NetworkXError
        If `G` is not a DAG.
    NodeNotFound
        If a pair contains a node that is not in the graph.

    Examples
    --------
    >>> G = nx.DiGraph(
    ...     [("r", "a"), ("r", "b"), ("a", "x"), ("b", "x"),
    ...      ("a", "y"), ("b", "y")]
    ... )
    >>> dict(nx.all_pairs_lowest_common_ancestors(G, [("x", "y")])) == {
    ...     ("x", "y"): {"a", "b"}}
    True
    >>> dict(nx.all_pairs_lowest_common_ancestors(G, [("x", "x")])) == {
    ...     ("x", "x"): {"x"}}
    True

    For many queries on the same graph, build a :class:`DAGLCAIndex` once;
    this function rebuilds the ancestor information on every call.

    See Also
    --------
    DAGLCAIndex
    all_pairs_lowest_common_ancestor
    lowest_common_ancestor
    """
    index = DAGLCAIndex(G)
    return index.all_pairs(pairs)


@not_implemented_for("undirected")
@nx._dispatchable
def all_pairs_lowest_common_ancestor(G, pairs=None):
    """Return the lowest common ancestor of all pairs or the provided pairs

    Parameters
    ----------
    G : NetworkX directed graph

    pairs : iterable of pairs of nodes, optional (default: all pairs)
        The pairs of nodes of interest.
        If None, will find the LCA of all pairs of nodes.

    Yields
    ------
    ((node1, node2), lca) : 2-tuple
        Where lca is least common ancestor of node1 and node2.
        Note that for the default case, the order of the node pair is not considered,
        e.g. you will not get both ``(a, b)`` and ``(b, a)``

    Raises
    ------
    NetworkXPointlessConcept
        If `G` is null.
    NetworkXError
        If `G` is not a DAG.

    Examples
    --------
    >>> from pprint import pprint

    The default behavior is to yield the lowest common ancestor for all
    possible combinations of nodes in `G`, including self-pairings:

    >>> G = nx.DiGraph([(0, 1), (0, 3), (1, 2)])
    >>> pprint(dict(nx.all_pairs_lowest_common_ancestor(G)))
    {(0, 0): 0,
     (0, 1): 0,
     (0, 2): 0,
     (0, 3): 0,
     (1, 1): 1,
     (1, 2): 1,
     (1, 3): 0,
     (2, 2): 2,
     (3, 2): 0,
     (3, 3): 3}

    The pairs argument can be used to limit the output to only the
    specified node pairings:

    >>> dict(nx.all_pairs_lowest_common_ancestor(G, pairs=[(1, 2), (2, 3)]))
    {(1, 2): 1, (2, 3): 0}

    Notes
    -----
    Only defined on non-null directed acyclic graphs.

    See Also
    --------
    lowest_common_ancestor
    """
    if not nx.is_directed_acyclic_graph(G):
        raise nx.NetworkXError("LCA only defined on directed acyclic graphs.")
    if len(G) == 0:
        raise nx.NetworkXPointlessConcept("LCA meaningless on null graphs.")

    if pairs is None:
        pairs = combinations_with_replacement(G, 2)
    else:
        # Convert iterator to iterable, if necessary. Trim duplicates.
        pairs = dict.fromkeys(pairs)
        # Verify that each of the nodes in the provided pairs is in G
        nodeset = set(G)
        for pair in pairs:
            if set(pair) - nodeset:
                raise nx.NodeNotFound(
                    f"Node(s) {set(pair) - nodeset} from pair {pair} not in G."
                )

    # Once input validation is done, construct the generator
    def generate_lca_from_pairs(G, pairs):
        ancestor_cache = {}

        for v, w in pairs:
            if v not in ancestor_cache:
                ancestor_cache[v] = nx.ancestors(G, v)
                ancestor_cache[v].add(v)
            if w not in ancestor_cache:
                ancestor_cache[w] = nx.ancestors(G, w)
                ancestor_cache[w].add(w)

            common_ancestors = ancestor_cache[v] & ancestor_cache[w]

            if common_ancestors:
                common_ancestor = next(iter(common_ancestors))
                while True:
                    successor = None
                    for lower_ancestor in G.successors(common_ancestor):
                        if lower_ancestor in common_ancestors:
                            successor = lower_ancestor
                            break
                    if successor is None:
                        break
                    common_ancestor = successor
                yield ((v, w), common_ancestor)

    return generate_lca_from_pairs(G, pairs)


@not_implemented_for("undirected")
@nx._dispatchable
def lowest_common_ancestor(G, node1, node2, default=None):
    """Compute the lowest common ancestor of the given pair of nodes.

    Parameters
    ----------
    G : NetworkX directed graph

    node1, node2 : nodes in the graph.

    default : object
        Returned if no common ancestor between `node1` and `node2`

    Returns
    -------
    The lowest common ancestor of node1 and node2,
    or default if they have no common ancestors.

    Examples
    --------
    >>> G = nx.DiGraph()
    >>> nx.add_path(G, (0, 1, 2, 3))
    >>> nx.add_path(G, (0, 4, 3))
    >>> nx.lowest_common_ancestor(G, 2, 4)
    0

    See Also
    --------
    all_pairs_lowest_common_ancestor"""

    ans = list(all_pairs_lowest_common_ancestor(G, pairs=[(node1, node2)]))
    if ans:
        assert len(ans) == 1
        return ans[0][1]
    return default


@not_implemented_for("undirected")
@nx._dispatchable
def tree_all_pairs_lowest_common_ancestor(G, root=None, pairs=None):
    r"""Yield the lowest common ancestor for sets of pairs in a tree.

    Parameters
    ----------
    G : NetworkX directed graph (must be a tree)

    root : node, optional (default: None)
        The root of the subtree to operate on.
        If None, assume the entire graph has exactly one source and use that.

    pairs : iterable or iterator of pairs of nodes, optional (default: None)
        The pairs of interest. If None, Defaults to all pairs of nodes
        under `root` that have a lowest common ancestor.

    Returns
    -------
    lcas : generator of tuples `((u, v), lca)` where `u` and `v` are nodes
        in `pairs` and `lca` is their lowest common ancestor.

    Examples
    --------
    >>> import pprint
    >>> G = nx.DiGraph([(1, 3), (2, 4), (1, 2)])
    >>> pprint.pprint(dict(nx.tree_all_pairs_lowest_common_ancestor(G)))
    {(1, 1): 1,
     (2, 1): 1,
     (2, 2): 2,
     (3, 1): 1,
     (3, 2): 1,
     (3, 3): 3,
     (3, 4): 1,
     (4, 1): 1,
     (4, 2): 2,
     (4, 4): 4}

    We can also use `pairs` argument to specify the pairs of nodes for which we
    want to compute lowest common ancestors. Here is an example:

    >>> dict(nx.tree_all_pairs_lowest_common_ancestor(G, pairs=[(1, 4), (2, 3)]))
    {(2, 3): 1, (1, 4): 1}

    Notes
    -----
    Only defined on non-null trees represented with directed edges from
    parents to children. Uses Tarjan's off-line lowest-common-ancestors
    algorithm. Runs in time $O(4 \times (V + E + P))$ time, where 4 is the largest
    value of the inverse Ackermann function likely to ever come up in actual
    use, and $P$ is the number of pairs requested (or $V^2$ if all are needed).

    Tarjan, R. E. (1979), "Applications of path compression on balanced trees",
    Journal of the ACM 26 (4): 690-715, doi:10.1145/322154.322161.

    See Also
    --------
    all_pairs_lowest_common_ancestor: similar routine for general DAGs
    lowest_common_ancestor: just a single pair for general DAGs
    """
    if len(G) == 0:
        raise nx.NetworkXPointlessConcept("LCA meaningless on null graphs.")

    # Index pairs of interest for efficient lookup from either side.
    if pairs is not None:
        pair_dict = defaultdict(set)
        # See note on all_pairs_lowest_common_ancestor.
        if not isinstance(pairs, Mapping | Set):
            pairs = set(pairs)
        for u, v in pairs:
            for n in (u, v):
                if n not in G:
                    msg = f"The node {str(n)} is not in the digraph."
                    raise nx.NodeNotFound(msg)
            pair_dict[u].add(v)
            pair_dict[v].add(u)

    # If root is not specified, find the exactly one node with in degree 0 and
    # use it. Raise an error if none are found, or more than one is. Also check
    # for any nodes with in degree larger than 1, which would imply G is not a
    # tree.
    if root is None:
        for n, deg in G.in_degree:
            if deg == 0:
                if root is not None:
                    msg = "No root specified and tree has multiple sources."
                    raise nx.NetworkXError(msg)
                root = n
            # checking deg>1 is not sufficient for MultiDiGraphs
            elif deg > 1 and len(G.pred[n]) > 1:
                msg = "Tree LCA only defined on trees; use DAG routine."
                raise nx.NetworkXError(msg)
    if root is None:
        raise nx.NetworkXError("Graph contains a cycle.")

    # Iterative implementation of Tarjan's offline lca algorithm
    # as described in CLRS on page 521 (2nd edition)/page 584 (3rd edition)
    uf = UnionFind()
    ancestors = {}
    for node in G:
        ancestors[node] = uf[node]

    colors = defaultdict(bool)
    for node in nx.dfs_postorder_nodes(G, root):
        colors[node] = True
        for v in pair_dict[node] if pairs is not None else G:
            if colors[v]:
                # If the user requested both directions of a pair, give it.
                # Otherwise, just give one.
                if pairs is not None and (node, v) in pairs:
                    yield (node, v), ancestors[uf[v]]
                if pairs is None or (v, node) in pairs:
                    yield (v, node), ancestors[uf[v]]
        if node != root:
            parent = arbitrary_element(G.pred[node])
            uf.union(parent, node)
            ancestors[uf[parent]] = parent
