"""Functions for computing treewidth decomposition.

Treewidth of an undirected graph is a number associated with the graph.
It can be defined as the size of the largest vertex set (bag) in a tree
decomposition of the graph minus one.

`Wikipedia: Treewidth <https://en.wikipedia.org/wiki/Treewidth>`_

The notions of treewidth and tree decomposition have gained their
attractiveness partly because many graph and network problems that are
intractable (e.g., NP-hard) on arbitrary graphs become efficiently
solvable (e.g., with a linear time algorithm) when the treewidth of the
input graphs is bounded by a constant [1]_ [2]_.

There are two different functions for computing a tree decomposition:
:func:`treewidth_min_degree` and :func:`treewidth_min_fill_in`.

.. [1] Hans L. Bodlaender and Arie M. C. A. Koster. 2010. "Treewidth
      computations I.Upper bounds". Inf. Comput. 208, 3 (March 2010),259-275.
      http://dx.doi.org/10.1016/j.ic.2009.03.008

.. [2] Hans L. Bodlaender. "Discovering Treewidth". Institute of Information
      and Computing Sciences, Utrecht University.
      Technical Report UU-CS-2005-018.
      http://www.cs.uu.nl

.. [3] K. Wang, Z. Lu, and J. Hicks *Treewidth*.
      https://web.archive.org/web/20210507025929/http://web.eecs.utk.edu/~cphill25/cs594_spring2015_projects/treewidth.pdf

"""

import itertools
import sys
from heapq import heapify, heappop, heappush

import networkx as nx
from networkx.utils import not_implemented_for

__all__ = [
    "treewidth_min_degree",
    "treewidth_min_fill_in",
    "maximum_weight_independent_set",
]


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@nx._dispatchable(returns_graph=True)
def treewidth_min_degree(G):
    """Returns a treewidth decomposition using the Minimum Degree heuristic.

    The heuristic chooses the nodes according to their degree, i.e., first
    the node with the lowest degree is chosen, then the graph is updated
    and the corresponding node is removed. Next, a new node with the lowest
    degree is chosen, and so on.

    Parameters
    ----------
    G : NetworkX graph

    Returns
    -------
    Treewidth decomposition : (int, Graph) tuple
          2-tuple with treewidth and the corresponding decomposed tree.
    """
    deg_heuristic = MinDegreeHeuristic(G)
    return treewidth_decomp(G, lambda graph: deg_heuristic.best_node(graph))


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@nx._dispatchable(returns_graph=True)
def treewidth_min_fill_in(G):
    """Returns a treewidth decomposition using the Minimum Fill-in heuristic.

    The heuristic chooses a node from the graph, where the number of edges
    added turning the neighborhood of the chosen node into clique is as
    small as possible.

    Parameters
    ----------
    G : NetworkX graph

    Returns
    -------
    Treewidth decomposition : (int, Graph) tuple
        2-tuple with treewidth and the corresponding decomposed tree.
    """
    return treewidth_decomp(G, min_fill_in_heuristic)


class MinDegreeHeuristic:
    """Implements the Minimum Degree heuristic.

    The heuristic chooses the nodes according to their degree
    (number of neighbors), i.e., first the node with the lowest degree is
    chosen, then the graph is updated and the corresponding node is
    removed. Next, a new node with the lowest degree is chosen, and so on.
    """

    def __init__(self, graph):
        self._graph = graph

        # nodes that have to be updated in the heap before each iteration
        self._update_nodes = []

        self._degreeq = []  # a heapq with 3-tuples (degree,unique_id,node)
        self.count = itertools.count()

        # build heap with initial degrees
        for n in graph:
            self._degreeq.append((len(graph[n]), next(self.count), n))
        heapify(self._degreeq)

    def best_node(self, graph):
        # update nodes in self._update_nodes
        for n in self._update_nodes:
            # insert changed degrees into degreeq
            heappush(self._degreeq, (len(graph[n]), next(self.count), n))

        # get the next valid (minimum degree) node
        while self._degreeq:
            (min_degree, _, elim_node) = heappop(self._degreeq)
            if elim_node not in graph or len(graph[elim_node]) != min_degree:
                # outdated entry in degreeq
                continue
            elif min_degree == len(graph) - 1:
                # fully connected: abort condition
                return None

            # remember to update nodes in the heap before getting the next node
            self._update_nodes = graph[elim_node]
            return elim_node

        # the heap is empty: abort
        return None


def min_fill_in_heuristic(graph_dict):
    """Implements the Minimum Degree heuristic.

    graph_dict: dict keyed by node to sets of neighbors (no self-loops)

    Returns the node from the graph, where the number of edges added when
    turning the neighborhood of the chosen node into clique is as small as
    possible. This algorithm chooses the nodes using the Minimum Fill-In
    heuristic. The running time of the algorithm is :math:`O(V^3)` and it uses
    additional constant memory.
    """

    if len(graph_dict) == 0:
        return None

    min_fill_in_node = None

    min_fill_in = sys.maxsize

    # sort nodes by degree
    nodes_by_degree = sorted(graph_dict, key=lambda x: len(graph_dict[x]))
    min_degree = len(graph_dict[nodes_by_degree[0]])

    # abort condition (handle complete graph)
    if min_degree == len(graph_dict) - 1:
        return None

    for node in nodes_by_degree:
        num_fill_in = 0
        nbrs = graph_dict[node]
        for nbr in nbrs:
            # count how many nodes in nbrs current nbr is not connected to
            # subtract 1 for the node itself
            num_fill_in += len(nbrs - graph_dict[nbr]) - 1
            if num_fill_in >= 2 * min_fill_in:
                break

        num_fill_in /= 2  # divide by 2 because of double counting

        if num_fill_in < min_fill_in:  # update min-fill-in node
            if num_fill_in == 0:
                return node
            min_fill_in = num_fill_in
            min_fill_in_node = node

    return min_fill_in_node


@nx._dispatchable(returns_graph=True)
def treewidth_decomp(G, heuristic=min_fill_in_heuristic):
    """Returns a treewidth decomposition using the passed heuristic.

    Parameters
    ----------
    G : NetworkX graph
    heuristic : heuristic function

    Returns
    -------
    Treewidth decomposition : (int, Graph) tuple
        2-tuple with treewidth and the corresponding decomposed tree.
    """

    # make dict-of-sets structure
    graph_dict = {n: set(G[n]) - {n} for n in G}

    # stack containing nodes and neighbors in the order from the heuristic
    node_stack = []

    # get first node from heuristic
    elim_node = heuristic(graph_dict)
    while elim_node is not None:
        # connect all neighbors with each other
        nbrs = graph_dict[elim_node]
        for u, v in itertools.permutations(nbrs, 2):
            if v not in graph_dict[u]:
                graph_dict[u].add(v)

        # push node and its current neighbors on stack
        node_stack.append((elim_node, nbrs))

        # remove node from graph_dict
        for u in graph_dict[elim_node]:
            graph_dict[u].remove(elim_node)

        del graph_dict[elim_node]
        elim_node = heuristic(graph_dict)

    # the abort condition is met; put all remaining nodes into one bag
    decomp = nx.Graph()
    first_bag = frozenset(graph_dict.keys())
    decomp.add_node(first_bag)

    treewidth = len(first_bag) - 1

    while node_stack:
        # get node and its neighbors from the stack
        (curr_node, nbrs) = node_stack.pop()

        # find a bag all neighbors are in
        old_bag = None
        for bag in decomp.nodes:
            if nbrs <= bag:
                old_bag = bag
                break

        if old_bag is None:
            # no old_bag was found: just connect to the first_bag
            old_bag = first_bag

        # create new node for decomposition
        nbrs.add(curr_node)
        new_bag = frozenset(nbrs)

        # update treewidth
        treewidth = max(treewidth, len(new_bag) - 1)

        # add edge to decomposition (implicitly also adds the new node)
        decomp.add_edge(old_bag, new_bag)

    return treewidth, decomp


def _validate_tree_decomposition(G, decomp):
    """Raise `NetworkXError` if `decomp` is not a valid tree decomposition of `G`.

    A valid tree decomposition is a tree whose nodes ("bags") are sets of
    nodes of `G` such that every node of `G` appears in some bag, every edge
    of `G` has both endpoints in some common bag, and the bags containing any
    given node of `G` form a connected subtree.
    """
    if not nx.is_tree(decomp):
        raise nx.NetworkXError("decomp must be a tree.")

    bags = list(decomp.nodes)
    for bag in bags:
        for v in bag:
            if v not in G:
                raise nx.NetworkXError(
                    f"decomp bag contains {v!r}, which is not a node of G."
                )

    # Every node of G must appear in some bag, and the bags containing it
    # must form a connected subtree of the decomposition.
    for v in G:
        containing = [bag for bag in bags if v in bag]
        if not containing:
            raise nx.NetworkXError(f"node {v!r} of G does not appear in any bag.")
        if len(containing) > 1 and not nx.is_connected(decomp.subgraph(containing)):
            raise nx.NetworkXError(
                f"the bags containing node {v!r} do not form a connected subtree."
            )

    # Every edge of G must have both endpoints together in some bag.
    for u, v in G.edges():
        if not any(u in bag and v in bag for bag in bags):
            raise nx.NetworkXError(
                f"edge ({u!r}, {v!r}) of G is not covered by any bag."
            )


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@nx._dispatchable
def maximum_weight_independent_set(G, decomp, weight="weight", max_bag_size=None):
    r"""Returns a maximum weight independent set of `G` using a tree decomposition.

    An independent set is a set of nodes no two of which are adjacent. This
    function computes an independent set of maximum total node weight by
    dynamic programming over the given tree decomposition `decomp` of `G`.
    The running time is exponential in the size of the largest bag but only
    linear in the number of bags, so it is efficient when `G` has small
    treewidth and a decomposition of small width is provided.

    Parameters
    ----------
    G : NetworkX graph
        The undirected graph on which to compute the independent set.

    decomp : NetworkX graph
        A tree decomposition of `G` as returned, e.g., by
        :func:`treewidth_min_degree` or :func:`treewidth_min_fill_in`: a tree
        whose nodes are containers ("bags") of nodes of `G`. Every node of
        `G` must appear in some bag, every edge of `G` must have both
        endpoints in some common bag, and the bags containing any given node
        must form a connected subtree.

    weight : string or None, optional (default="weight")
        The node attribute used as weight. Nodes without the attribute have
        weight 1. If None, every node has weight 1. Weights may be zero or
        negative; the empty set is always a feasible solution (of weight 0).

    max_bag_size : int or None, optional (default=None)
        Maximum allowed bag size. If the largest bag of `decomp` exceeds
        this value, a `NetworkXError` is raised before running the
        (exponential in the bag size) dynamic program.

    Returns
    -------
    (weight, nodes) : tuple
        The total weight of a maximum weight independent set and the set of
        nodes of `G` forming it.

    Raises
    ------
    NetworkXNotImplemented
        If `G` is directed or a multigraph.

    NetworkXError
        If `decomp` is not a valid tree decomposition of `G`, or if the
        largest bag exceeds `max_bag_size`.

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> tw, decomp = nx.algorithms.approximation.treewidth_min_degree(G)
    >>> w, nodes = nx.algorithms.approximation.maximum_weight_independent_set(G, decomp)
    >>> w
    2
    >>> all(not G.has_edge(u, v) for u in nodes for v in nodes)
    True

    Notes
    -----
    The decomposition tree is rooted at an arbitrary bag and processed
    bottom-up. For each bag, a table maps every independent subset of the
    bag (its "state") to the maximum weight of an independent set of the
    processed subtree that agrees with the state on the bag. Merging a child
    bag into its parent handles the three classical cases: nodes present in
    the child but not the parent are *forgotten* (their weight is folded
    into the best compatible child state), nodes present in the parent but
    not the child are *introduced*, and a bag with several children *joins*
    their tables, subtracting the weight of shared nodes so that each node
    weight is counted exactly once.

    See Also
    --------
    treewidth_min_degree
    treewidth_min_fill_in
    maximal_independent_set
    """
    if len(G) == 0:
        return 0, set()

    if max_bag_size is not None:
        largest = max((len(bag) for bag in decomp.nodes), default=0)
        if largest > max_bag_size:
            raise nx.NetworkXError(
                f"largest bag of decomp has size {largest},"
                f" which exceeds max_bag_size={max_bag_size}."
            )

    _validate_tree_decomposition(G, decomp)

    if weight is None:
        weights = {v: 1 for v in G}
    else:
        weights = {v: data.get(weight, 1) for v, data in G.nodes(data=True)}

    # Root the decomposition tree at an arbitrary bag and compute a
    # bottom-up processing order (children before parents).
    root = next(iter(decomp.nodes))
    parent = {root: None}
    children = {root: []}
    order = [root]
    for bag in order:
        for nbr in decomp.neighbors(bag):
            if nbr not in parent:
                parent[nbr] = bag
                children[nbr] = []
                children[bag].append(nbr)
                order.append(nbr)

    bag_members = {}  # bag -> list of its nodes (fixes a position per node)
    tables = {}  # bag -> {state bitmask: max weight in the processed subtree}
    choices = {}  # bag -> {(child, parent state): chosen child state}

    for bag in reversed(order):
        members = list(bag)
        bag_members[bag] = members
        position = {v: i for i, v in enumerate(members)}
        # Adjacency (including self-loops) restricted to the bag, as bitmasks.
        adjacency = [0] * len(members)
        for i, u in enumerate(members):
            for v in G[u]:
                j = position.get(v)
                if j is not None:
                    adjacency[i] |= 1 << j

        # States are the independent subsets of the bag; the weight of the
        # selected bag nodes is counted here, at the highest bag (closest to
        # the root) containing them, so it is counted exactly once.
        table = {}
        for mask in range(1 << len(members)):
            if all(
                not adjacency[i] & mask for i in range(len(members)) if mask >> i & 1
            ):
                table[mask] = sum(
                    weights[members[i]] for i in range(len(members)) if mask >> i & 1
                )

        bag_choices = {}
        for child in children[bag]:
            child_members = bag_members[child]
            # Best child state for each possible intersection with this bag.
            # The weight of shared nodes is subtracted so it is not counted
            # twice; forgotten child nodes are folded into `best`.
            best = {}
            for cmask, value in tables[child].items():
                cap = 0
                cap_weight = 0
                for j, v in enumerate(child_members):
                    if cmask >> j & 1 and v in position:
                        cap |= 1 << position[v]
                        cap_weight += weights[v]
                candidate = value - cap_weight
                if cap not in best or candidate > best[cap][0]:
                    best[cap] = (candidate, cmask)
            child_set = set(child_members)
            merged = {}
            for mask, value in table.items():
                key = 0
                for i, v in enumerate(members):
                    if mask >> i & 1 and v in child_set:
                        key |= 1 << i
                gain, cmask = best[key]
                merged[mask] = value + gain
                bag_choices[child, mask] = cmask
            table = merged
        tables[bag] = table
        choices[bag] = bag_choices

    best_mask = max(tables[root], key=tables[root].get)
    best_weight = tables[root][best_mask]

    # Backtrack from the root to recover the selected nodes.
    selected = set()
    stack = [(root, best_mask)]
    while stack:
        bag, mask = stack.pop()
        members = bag_members[bag]
        selected.update(members[i] for i in range(len(members)) if mask >> i & 1)
        for child in children[bag]:
            stack.append((child, choices[bag][child, mask]))

    return best_weight, selected
