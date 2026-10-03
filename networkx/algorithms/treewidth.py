"""Exact algorithms on graphs of bounded treewidth.

The functions in this module solve otherwise NP-hard graph problems
exactly, provided the user supplies a tree decomposition of the graph.
The algorithms are exponential in the width of the given decomposition
but only linear in the number of vertices of the graph, so they are
practical when the treewidth is small. The user is responsible for
finding a decomposition of small width; no attempt is made to optimize
the width of a supplied decomposition.

A tree decomposition of a graph ``G`` is a tree ``T`` whose nodes (the
*bags*) are sets of vertices of ``G`` satisfying:

1. every vertex of ``G`` belongs to at least one bag (vertex coverage),
2. every edge of ``G`` has both endpoints in a common bag (edge
   coverage), and
3. for every vertex of ``G``, the bags containing it induce a connected
   subtree of ``T`` (the running-intersection property).

The representation is the same as the one returned by
:func:`~networkx.algorithms.approximation.treewidth.treewidth_decomp`:
``T`` is a :class:`networkx.Graph` whose nodes are
:class:`frozenset` instances of vertices of ``G``.
"""

import networkx as nx
from networkx.utils import not_implemented_for

__all__ = [
    "is_valid_tree_decomposition",
    "max_weight_independent_set",
    "maximum_weight_independent_set",
]


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@nx._dispatchable
def is_valid_tree_decomposition(G, T):
    """Check whether ``T`` is a tree decomposition of ``G``.

    A tree decomposition must satisfy the three defining conditions:

    1. every vertex of ``G`` appears in at least one bag (coverage),
    2. both endpoints of every edge of ``G`` appear together in some
       bag (edge coverage), and
    3. for every vertex, the bags containing it induce a connected
       subtree of ``T`` (running-intersection property).

    The bags are the nodes of ``T`` and must be sets (or frozensets)
    of vertices of ``G``. The decomposition tree itself must be a tree.

    Parameters
    ----------
    G : NetworkX graph
        The graph that ``T`` is supposed to decompose.

    T : NetworkX graph
        The candidate tree decomposition. Its nodes are sets of vertices
        of ``G``.

    Returns
    -------
    bool
        ``True`` if ``T`` is a tree decomposition of ``G``, ``False``
        otherwise.

    Examples
    --------
    >>> G = nx.path_graph(3)
    >>> T = nx.Graph()
    >>> T.add_edge(frozenset({0, 1}), frozenset({1, 2}))
    >>> nx.is_valid_tree_decomposition(G, T)
    True
    """
    bags = list(T.nodes)
    if not bags:
        return len(G) == 0

    vertices = set(G)
    bag_sets = []
    covered = set()
    for bag in bags:
        try:
            bag_vertices = set(bag)
        except TypeError:
            return False
        # every bag must contain vertices of G
        if not bag_vertices <= vertices:
            return False
        bag_sets.append(bag_vertices)
        covered.update(bag_vertices)

    # vertex coverage
    if covered != vertices:
        return False

    # T must be a tree
    if len(T.edges) != len(bags) - 1 or not nx.is_connected(T):
        return False

    # edge coverage: every edge of G lies in a common bag
    for u, v in G.edges:
        if not any(u in bag and v in bag for bag in bag_sets):
            return False

    # running-intersection property: for each vertex, the bags
    # containing it induce a connected subtree of T. A subtree of a
    # tree is connected iff it contains (#bags - 1) internal edges.
    for v in vertices:
        containing = [bag for bag in bags if v in bag]
        internal_edges = sum(
            1 for u_bag, w_bag in T.edges if v in u_bag and v in w_bag
        )
        if internal_edges != len(containing) - 1:
            return False

    return True


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@nx._dispatchable
def max_weight_independent_set(G, T, weight="weight", max_bag_size=None):
    """Compute a maximum-weight independent set on a tree decomposition.

    An independent set is a set of vertices no two of which are joined
    by an edge. This function finds an independent set of maximum total
    vertex weight by dynamic programming over the bags of the supplied
    tree decomposition. Weights may be zero or negative and the empty
    set is always a candidate, so when all weights are non-positive the
    returned set is empty with weight zero.

    The running time is exponential in the size of the largest bag
    (that is, in the treewidth plus one), but only linear in the number
    of vertices of ``G``. The decomposition itself is not modified and
    no attempt is made to improve its width.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    T : NetworkX graph
        A tree decomposition of ``G`` represented as a graph whose
        nodes are sets (typically frozensets) of vertices of ``G``, as
        returned by
        :func:`~networkx.algorithms.approximation.treewidth.treewidth_decomp`.
        It is validated with :func:`is_valid_tree_decomposition`.

    weight : string or None, optional (default="weight")
        The node attribute holding the weight of each vertex. Vertices
        without this attribute, or any vertex when ``weight is None``,
        have weight 1.

    max_bag_size : int or None, optional (default=None)
        Optional upper bound on the number of vertices in a bag. If any
        bag of ``T`` contains more vertices than this bound, a
        :exc:`~networkx.NetworkXTreewidthBoundExceeded` is raised
        instead of running the (exponential-time) computation.

    Returns
    -------
    value : number
        The total weight of a maximum-weight independent set.

    indep_nodes : set
        The vertices of a maximum-weight independent set of ``G``.

    Raises
    ------
    NetworkXTreewidthBoundExceeded
        If a bag of ``T`` contains more than ``max_bag_size`` vertices.

    ValueError
        If ``T`` is not a valid tree decomposition of ``G``.

    Examples
    --------
    >>> G = nx.path_graph(3)
    >>> T = nx.Graph()
    >>> T.add_edge(frozenset({0, 1}), frozenset({1, 2}))
    >>> value, nodes = nx.max_weight_independent_set(G, T)
    >>> value
    2
    >>> nodes == {0, 2}
    True
    """
    if max_bag_size is not None and any(len(bag) > max_bag_size for bag in T):
        raise nx.NetworkXTreewidthBoundExceeded(
            f"decomposition contains a bag with more than {max_bag_size} "
            "vertices; refusing to run exponential-time computation"
        )

    if not is_valid_tree_decomposition(G, T):
        raise ValueError("T is not a valid tree decomposition of G")

    if len(G) == 0:
        return 0, set()

    vertex_weight = dict(G.nodes(data=weight, default=1)) if weight else {
        v: 1 for v in G
    }

    # Root the decomposition tree at an arbitrary bag and process the
    # bags bottom-up (postorder), keeping the parent of each bag.
    root = next(iter(T))
    parent = {}
    postorder = []
    stack = [(root, None)]
    while stack:
        bag, par = stack.pop()
        parent[bag] = par
        postorder.append(bag)
        for neighbor in T.neighbors(bag):
            if neighbor != par:
                stack.append((neighbor, bag))

    # For each bag, table maps an independent subset of the bag to the
    # best total weight obtainable in the processed subtree, and choice
    # records, for every child bag, the subset of that child attaining
    # the best value. The weight of a vertex is counted in exactly one
    # bag: the highest (closest to the root) bag in which it appears.
    table = {}
    choice = {}
    for bag in reversed(postorder):
        bag_vertices = list(bag)
        independent_subsets = _independent_subsets(G, bag_vertices)

        weight_here = {
            v: vertex_weight[v]
            for v in bag_vertices
            if parent[bag] is None or v not in parent[bag]
        }
        children = [c for c in T.neighbors(bag) if parent[c] is bag]

        bag_table = {}
        bag_choice = {}
        for subset in independent_subsets:
            value = sum(
                (weight_here[v] for v in subset if v in weight_here), 0
            )
            subset_choices = {}
            for child in children:
                separator = child & bag
                child_value, child_subset = max(
                    (
                        (candidate_value, candidate)
                        for candidate, candidate_value in table[
                            child
                        ].items()
                        if candidate & separator == subset & separator
                    ),
                    key=lambda item: item[0],
                )
                value += child_value
                subset_choices[child] = child_subset
            bag_table[subset] = value
            bag_choice[subset] = subset_choices
        table[bag] = bag_table
        choice[bag] = bag_choice

    # Read off the best state at the root and reconstruct the chosen
    # vertices by walking back down the decomposition tree.
    root_subset = max(table[root], key=table[root].get)
    best_value = table[root][root_subset]

    result = set()
    stack = [(root, root_subset)]
    while stack:
        bag, subset = stack.pop()
        result.update(subset)
        for child, child_subset in choice[bag][subset].items():
            stack.append((child, child_subset))

    return best_value, result


maximum_weight_independent_set = max_weight_independent_set


def _independent_subsets(G, vertices):
    """All independent subsets of ``vertices`` with respect to ``G``.

    Subsets are returned as frozensets.
    """
    n = len(vertices)
    masks = []
    for i, u in enumerate(vertices):
        mask = 0
        for j, v in enumerate(vertices):
            if u != v and G.has_edge(u, v):
                mask |= 1 << j
        masks.append(mask)

    result = []
    for bits in range(1 << n):
        if all(not (bits & masks[i]) for i in range(n) if bits & (1 << i)):
            result.append(
                frozenset(vertices[i] for i in range(n) if bits & (1 << i))
            )
    return result
