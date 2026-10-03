import pytest

import networkx as nx


def test_dfa_requires_label_order():
    G = nx.MultiDiGraph()
    G.add_edge("client", "server", label="login", weight=1)
    read_key = G.add_edge("server", "server", label="read", weight=2)
    logout_key = G.add_edge("server", "client", label="logout", weight=1)

    states = {"start", "out", "in", "read"}
    transitions = {
        "start": {"login": "in"},
        "in": {"read": "read"},
        "read": {"read": "read", "logout": "out"},
        "out": {},
    }

    length, nodes, edges = nx.constrained_shortest_path(
        G, "client", "client", states, transitions, "start", "out"
    )

    assert length == 4
    assert nodes == ["client", "server", "server", "client"]
    assert edges == [
        ("client", "server", 0),
        ("server", "server", read_key),
        ("server", "client", logout_key),
    ]
    labels = ["login", "read", "logout"]
    assert all(G.edges[edge]["label"] == label for edge, label in zip(edges, labels))

    transitions = {
        "start": {"logout": "in"},
        "in": {"read": "in", "login": "out"},
        "out": {},
    }
    with pytest.raises(nx.NetworkXNoPath):
        nx.constrained_shortest_path(
            G, "client", "client", states, transitions, "start", "out"
        )


def test_empty_path_acceptance_is_automaton_specific():
    G = nx.path_graph(2, create_using=nx.DiGraph)
    states = {"reject", "accept"}

    length, nodes, edges = nx.constrained_shortest_path(
        G, 0, 0, states, {}, "accept", "accept"
    )
    assert (length, nodes, edges) == (0, [0], [])

    with pytest.raises(nx.NetworkXNoPath):
        nx.constrained_shortest_path(G, 0, 0, states, {}, "reject", "accept")


def test_requires_second_visit_to_same_node():
    G = nx.MultiDiGraph()
    visit_key = G.add_edge("a", "a", label="visit", weight=1)
    finish_key = G.add_edge("a", "b", label="finish", weight=2)

    states = {"first", "visited", "done"}
    transitions = {
        "first": {"visit": "visited"},
        "visited": {"finish": "done"},
        "done": {},
    }

    length, nodes, edges = nx.constrained_shortest_path(
        G, "a", "b", states, transitions, "first", "done"
    )
    assert length == 3
    assert nodes == ["a", "a", "b"]
    assert edges == [("a", "a", visit_key), ("a", "b", finish_key)]


def test_tuple_state_can_be_grouped_as_scalar():
    G = nx.DiGraph([("s", "t")])
    initial = ("phase", 0)
    final = ("phase", 1)
    states = {initial, final}
    transitions = {initial: {None: [final]}, final: {}}

    length, nodes, edges = nx.constrained_shortest_path(
        G, "s", "t", states, transitions, initial, [final]
    )
    assert (length, nodes, edges) == (1, ["s", "t"], [("s", "t")])


def test_tuple_contains_multiple_automaton_states():
    G = nx.DiGraph([("s", "t")])
    states = ("start", "end")
    transitions = {"start": {None: ("end",)}, "end": {}}

    length, _, _ = nx.constrained_shortest_path(
        G, "s", "t", states, transitions, "start", ("end",)
    )
    assert length == 1


def test_nondeterministic_transitions_and_epsilon_closure():
    G = nx.MultiDiGraph()
    direct_key = G.add_edge("s", "t", label="x", weight=10)
    epsilon_key = G.add_edge("s", "t", label="x", weight=3)

    states = {"start", "cheap", "expensive", "cheap_accept", "expensive_accept"}
    transitions = {
        "start": {"x": {"cheap", "expensive"}},
        "cheap": {"epsilon": "cheap_accept"},
        "expensive": {"epsilon": "expensive_accept"},
        "cheap_accept": {},
        "expensive_accept": {},
    }

    length, _, edges = nx.constrained_shortest_path(
        G,
        "s",
        "t",
        states,
        transitions,
        "start",
        {"cheap_accept", "expensive_accept"},
        epsilon="epsilon",
    )
    assert length == 3
    assert edges == [("s", "t", epsilon_key)]
    assert edges[0][2] != direct_key


def test_initial_epsilon_closure_and_epsilon_labeled_graph_edge():
    G = nx.DiGraph([("s", "t")])
    states = {"start", "ready", "after"}
    transitions = {
        "start": {"eps": "ready"},
        "ready": {None: "after"},
        "after": {},
    }

    length, nodes, edges = nx.constrained_shortest_path(
        G, "s", "t", states, transitions, "start", "after", epsilon="eps"
    )
    assert length == 1
    assert nodes == ["s", "t"]
    assert edges == [("s", "t")]


def test_unlabeled_edges_use_default_none_epsilon():
    G = nx.DiGraph([("s", "t")])
    states = {"start", "ready", "end"}
    transitions = {
        "start": {None: "ready"},
        "ready": {None: "end"},
        "end": {},
    }

    length, nodes, edges = nx.constrained_shortest_path(
        G, "s", "t", states, transitions, "start", "end"
    )
    assert length == 1
    assert nodes == ["s", "t"]
    assert edges == [("s", "t")]


def test_parallel_edges_preserve_distinct_labels_and_keys():
    G = nx.MultiDiGraph()
    bad_key = G.add_edge("s", "t", label="blocked", weight=1)
    good_key = G.add_edge("s", "t", label="allowed", weight=5)
    states = {"start", "end"}
    transitions = {"start": {"allowed": "end"}, "end": {}}

    length, nodes, edges = nx.constrained_shortest_path(
        G, "s", "t", states, transitions, "start", "end"
    )
    assert length == 5
    assert nodes == ["s", "t"]
    assert edges == [("s", "t", good_key)]
    assert edges[0][2] != bad_key


def test_undirected_graph_traverses_edges_in_either_direction():
    G = nx.Graph([("s", "m"), ("m", "t")])
    states = {"start", "after"}
    transitions = {"start": {None: "after"}, "after": {None: "after"}}

    length, nodes, edges = nx.constrained_shortest_path(
        G, "t", "s", states, transitions, "start", "after"
    )
    assert length == 2
    assert nodes == ["t", "m", "s"]
    assert edges == [("t", "m"), ("m", "s")]


def test_callable_label_and_weight_receive_multigraph_key():
    G = nx.MultiDiGraph()
    G.add_edge("s", "t", symbol="go", cost=4, key=0)

    states = {"start", "end"}
    transitions = {"start": {"go": "end"}, "end": {}}

    length, _, edges = nx.constrained_shortest_path(
        G,
        "s",
        "t",
        states,
        transitions,
        "start",
        "end",
        label=lambda u, v, key, data: data["symbol"],
        weight=lambda u, v, key, data: data["cost"],
    )
    assert length == 4
    assert edges == [("s", "t", 0)]


def test_weight_zero_cycles_allow_repeated_graph_nodes():
    G = nx.DiGraph([("s", "a"), ("a", "t")])
    G.add_edge("a", "a", label="again", weight=0)
    for edge in G.edges:
        if G.edges[edge].get("label") != "again":
            G.edges[edge]["label"] = "move"
    states = {"start", "looped"}
    transitions = {
        "start": {"move": "start", "again": "looped"},
        "looped": {"move": "looped", "again": "looped"},
    }

    length, nodes, edges = nx.constrained_shortest_path(
        G, "s", "t", states, transitions, "start", "looped"
    )
    assert length == 2
    assert nodes == ["s", "a", "a", "t"]
    assert edges == [("s", "a"), ("a", "a"), ("a", "t")]


@pytest.mark.parametrize(
    ("states", "initial", "accepting", "message"),
    [({"a"}, "b", "a", "initial state"), ({"a"}, "a", "b", "accepting state")],
)
def test_invalid_automaton_references(states, initial, accepting, message):
    G = nx.DiGraph([("s", "t")])
    with pytest.raises(ValueError, match=message):
        nx.constrained_shortest_path(G, "s", "t", states, {}, initial, accepting)


def test_invalid_transition_references():
    G = nx.DiGraph([("s", "t")])
    states = {"start", "end"}
    transitions = {"start": {"label": "missing"}, "missing": {}, "end": {}}
    with pytest.raises(ValueError, match="references a state not in states"):
        nx.constrained_shortest_path(
            G, "s", "t", states, transitions, "start", "end"
        )


def test_invalid_transition_source():
    G = nx.DiGraph([("s", "t")])
    transitions = {"missing": {None: "start"}, "start": {}}
    with pytest.raises(ValueError, match="transition source state"):
        nx.constrained_shortest_path(
            G, "s", "t", {"start"}, transitions, "start", "start"
        )


def test_max_states_limits_product_configurations():
    G = nx.path_graph(4, create_using=nx.DiGraph)
    states = {0, 1}
    transitions = {0: {None: {0, 1}}, 1: {None: {0, 1}}}
    with pytest.raises(nx.NetworkXError, match="max_states=3"):
        nx.constrained_shortest_path(
            G, 0, 3, states, transitions, 0, {0, 1}, max_states=3
        )


def test_nonpositive_max_states_is_rejected():
    G = nx.DiGraph([("s", "t")])
    transitions = {"start": {}}
    with pytest.raises(ValueError, match="max_states"):
        nx.constrained_shortest_path(
            G, "s", "t", {"start"}, transitions, "start", "start", max_states=0
        )


def test_negative_weight_is_rejected():
    G = nx.DiGraph()
    G.add_edge("s", "t", label="x", weight=-1)
    states = {"start", "end"}
    transitions = {"start": {"x": "end"}, "end": {}}
    with pytest.raises(ValueError, match="negative edge weight"):
        nx.constrained_shortest_path(
            G, "s", "t", states, transitions, "start", "end"
        )


def test_missing_graph_nodes():
    G = nx.DiGraph([("s", "t")])
    states = {"start", "end"}
    transitions = {"start": {None: "end"}, "end": {}}
    with pytest.raises(nx.NodeNotFound, match="source"):
        nx.constrained_shortest_path(
            G, "missing", "t", states, transitions, "start", "end"
        )
    with pytest.raises(nx.NodeNotFound, match="target"):
        nx.constrained_shortest_path(
            G, "s", "missing", states, transitions, "start", "end"
        )
