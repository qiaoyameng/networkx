import pytest

import networkx as nx


def test_empty_path_acceptance_is_defined_by_automaton():
    G = nx.DiGraph()
    G.add_node(0)

    rejected = dict(
        automaton_states=(0, 1),
        automaton_transitions=((0, "a", 1),),
        initial_automaton_state=0,
        accepting_automaton_states=(1,),
    )
    with pytest.raises(nx.NetworkXNoPath, match="automaton-accepted"):
        nx.constrained_shortest_path(G, 0, 0, **rejected)

    accepted = rejected | {"accepting_automaton_states": (0,)}
    result = nx.constrained_shortest_path(G, 0, 0, **accepted)
    assert result == (0, [0], [], [], [0])
    assert result.path is result.nodes


def test_label_order_constraint_and_nonnegative_weights():
    G = nx.DiGraph()
    G.add_edge("start", "session", label="login", weight=1)
    G.add_edge("start", "file", label="read", weight=1)
    G.add_edge("session", "file", label="read", weight=10)
    G.add_edge("file", "done", label="logout", weight=1)

    transitions = (
        ("out", "login", "in"),
        ("in", "read", "in"),
        ("in", "logout", "out"),
    )
    result = nx.constrained_shortest_path(
        G,
        "start",
        "done",
        automaton_states=("out", "in"),
        automaton_transitions=transitions,
        initial_automaton_state="out",
        accepting_automaton_states=("out",),
    )

    assert result.length == 12
    assert result.nodes == ["start", "session", "file", "done"]
    assert result.edges == [
        ("start", "session"),
        ("session", "file"),
        ("file", "done"),
    ]
    assert result.labels == ["login", "read", "logout"]
    assert result.automaton_states == ["out", "in", "in", "out"]

    invalid = dict(
        automaton_states=("out", "in"),
        automaton_transitions=(
            ("out", "login", "in"),
            ("in", "logout", "out"),
        ),
        initial_automaton_state="out",
        accepting_automaton_states=("out",),
    )
    with pytest.raises(nx.NetworkXNoPath):
        nx.constrained_shortest_path(G, "start", "done", **invalid)


def test_nondeterministic_transitions_and_epsilon_closure():
    G = nx.cycle_graph(4, create_using=nx.DiGraph())
    for u, v in G.edges:
        G.edges[u, v]["label"] = "x"

    transitions = (
        (0, "epsilon", 1),
        (1, "x", 2),
        (2, "x", 3),
        (3, "epsilon", 4),
    )
    result = nx.constrained_shortest_path(
        G,
        0,
        2,
        automaton_states=range(5),
        automaton_transitions=transitions,
        initial_automaton_state=0,
        accepting_automaton_states=(4,),
        epsilon="epsilon",
    )

    assert result.length == 2
    assert result.nodes == [0, 1, 2]
    assert result.labels == ["x", "x"]
    assert result.automaton_states == [1, 2, 4]


def test_nondeterministic_same_symbol_transition():
    G = nx.DiGraph([(0, 1), (1, 2), (0, 2)])
    for u, v in G.edges:
        G.edges[u, v]["label"] = "x"

    result = nx.constrained_shortest_path(
        G,
        0,
        2,
        automaton_states=(0, 1),
        automaton_transitions=(
            (0, "x", 0),
            (0, "x", 1),
            (1, "x", 1),
        ),
        initial_automaton_state=0,
        accepting_automaton_states=(1,),
    )

    assert result.length == 1
    assert result.nodes == [0, 2]
    assert result.automaton_states == [0, 1]


def test_none_label_is_regular_when_epsilon_disabled():
    G = nx.DiGraph([(0, 1)])

    result = nx.constrained_shortest_path(
        G,
        0,
        1,
        automaton_states=(0, 1),
        automaton_transitions=((0, None, 1),),
        initial_automaton_state=0,
        accepting_automaton_states=(1,),
    )

    assert result.labels == [None]
    assert result.automaton_states == [0, 1]


def test_repeated_graph_node_has_distinct_product_states():
    G = nx.DiGraph([("a", "b"), ("b", "a"), ("a", "goal")])
    for u, v in G.edges:
        G.edges[u, v]["label"] = "go"

    result = nx.constrained_shortest_path(
        G,
        "a",
        "goal",
        automaton_states=(0, 1, 2),
        automaton_transitions=(
            (0, "go", 1),
            (1, "go", 2),
            (2, "go", 2),
        ),
        initial_automaton_state=0,
        accepting_automaton_states=(2,),
    )

    assert result.nodes == ["a", "b", "a", "goal"]
    assert result.labels == ["go", "go", "go"]
    assert result.automaton_states == [0, 1, 2, 2]


def test_epsilon_symbol_cannot_label_graph_edge():
    G = nx.DiGraph([(0, 1)])
    G.edges[0, 1]["label"] = "epsilon"

    with pytest.raises(nx.NetworkXError, match="epsilon symbol"):
        nx.constrained_shortest_path(
            G,
            0,
            1,
            automaton_states=(0, 1),
            automaton_transitions=((0, "epsilon", 1),),
            initial_automaton_state=0,
            accepting_automaton_states=(1,),
            epsilon="epsilon",
        )


def test_multigraph_edges_keep_keys_and_labels():
    G = nx.MultiDiGraph()
    G.add_edge("s", "t", key="cheap", label="a", weight=10)
    G.add_edge("s", "t", key="valid", label="b", weight=2)
    G.add_edge("s", "t", key="invalid", label="c", weight=1)

    result = nx.constrained_shortest_path(
        G,
        "s",
        "t",
        automaton_states=(0, 1),
        automaton_transitions=((0, "b", 1),),
        initial_automaton_state=0,
        accepting_automaton_states=(1,),
    )

    assert result.length == 2
    assert result.edges == [("s", "t", "valid")]
    assert result.edge_keys == ["valid"]
    assert result.labels == ["b"]
    assert result.automaton_states == [0, 1]


def test_invalid_automaton_references():
    G = nx.path_graph(2)
    kwargs = dict(
        automaton_states=(0,),
        automaton_transitions=((0, "x", 2),),
        initial_automaton_state=0,
        accepting_automaton_states=(0,),
    )
    with pytest.raises(nx.NetworkXError, match="transition target"):
        nx.constrained_shortest_path(G, 0, 1, **kwargs)

    kwargs["automaton_transitions"] = ((2, "x", 0),)
    with pytest.raises(nx.NetworkXError, match="transition source"):
        nx.constrained_shortest_path(G, 0, 1, **kwargs)

    kwargs["automaton_transitions"] = ()
    kwargs["initial_automaton_state"] = 2
    with pytest.raises(nx.NetworkXError, match="initial automaton state"):
        nx.constrained_shortest_path(G, 0, 1, **kwargs)


def test_product_state_limit():
    G = nx.DiGraph([(0, 1), (1, 2)])
    for u, v in G.edges:
        G.edges[u, v]["label"] = "x"
    kwargs = dict(
        automaton_states=(0, 1),
        automaton_transitions=((0, "x", 0), (0, "x", 1), (1, "x", 1)),
        initial_automaton_state=0,
        accepting_automaton_states=(1,),
    )

    result = nx.constrained_shortest_path(G, 0, 2, max_product_states=5, **kwargs)
    assert result.nodes == [0, 1, 2]

    with pytest.raises(nx.NetworkXAlgorithmError, match="product-state limit"):
        nx.constrained_shortest_path(G, 0, 2, max_product_states=3, **kwargs)

    with pytest.raises(nx.NetworkXError, match="positive integer"):
        nx.constrained_shortest_path(G, 0, 2, max_product_states=0, **kwargs)


def test_negative_weight_is_rejected():
    G = nx.DiGraph([(0, 1)])
    G.edges[0, 1].update(label="x", weight=-1)
    with pytest.raises(ValueError, match="non-negative"):
        nx.constrained_shortest_path(
            G,
            0,
            1,
            automaton_states=(0, 1),
            automaton_transitions=((0, "x", 1),),
            initial_automaton_state=0,
            accepting_automaton_states=(1,),
        )
