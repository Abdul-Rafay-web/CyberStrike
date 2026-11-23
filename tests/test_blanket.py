import pytest

from core.generator import generate_network
from prob.blanket import BayesNetDAG, dbn_dag, markov_blanket, network_causal_dag
from tests.factories import chain_network, diamond_network


def test_markov_blanket_includes_parents_children_and_co_parents() -> None:
    dag: BayesNetDAG[str] = BayesNetDAG()
    # Classic v-structure + chain: P -> X -> C <- S, plus an unrelated ancestor U -> P
    dag.add_edge("U", "P")
    dag.add_edge("P", "X")
    dag.add_edge("X", "C")
    dag.add_edge("S", "C")
    dag.add_edge("C", "D")

    mb = dag.markov_blanket("X")
    assert mb.parents == frozenset({"P"})
    assert mb.children == frozenset({"C"})
    assert mb.co_parents == frozenset({"S"})
    assert mb.members == frozenset({"P", "C", "S"})
    assert "U" not in mb.members
    assert "D" not in mb.members


def test_dag_rejects_cycles_and_self_loops() -> None:
    dag: BayesNetDAG[int] = BayesNetDAG()
    dag.add_edge(0, 1)
    dag.add_edge(1, 2)
    with pytest.raises(ValueError):
        dag.add_edge(2, 0)
    with pytest.raises(ValueError):
        dag.add_edge(1, 1)


def test_diamond_network_markov_blanket_captures_co_parent_at_jewel() -> None:
    net = diamond_network()
    # Causal DAG: 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3
    mb_1 = markov_blanket(net, 1)
    assert mb_1.parents == frozenset({0})
    assert mb_1.children == frozenset({3})
    assert mb_1.co_parents == frozenset({2})
    assert mb_1.members == frozenset({0, 2, 3})

    mb_entry = markov_blanket(net, 0)
    assert mb_entry.parents == frozenset()
    assert mb_entry.children == frozenset({1, 2})
    assert mb_entry.co_parents == frozenset()

    mb_jewel = markov_blanket(net, 3)
    assert mb_jewel.parents == frozenset({1, 2})
    assert mb_jewel.children == frozenset()
    assert mb_jewel.co_parents == frozenset()


def test_chain_network_markov_blanket_has_no_co_parents() -> None:
    mb = markov_blanket(chain_network(), 1)
    assert mb.parents == frozenset({0})
    assert mb.children == frozenset({2})
    assert mb.co_parents == frozenset()


def test_network_causal_dag_is_always_acyclic_on_generated_maps() -> None:
    for seed in range(20):
        net = generate_network(seed)
        dag = network_causal_dag(net)
        order = dag.topological_order()
        assert len(order) == net.size
        pos = {node: idx for idx, node in enumerate(order)}
        for node in dag.nodes:
            for child in dag.children(node):
                assert pos[node] < pos[child]


def test_dbn_dag_markov_blanket_matches_temporal_structure() -> None:
    dag = dbn_dag(network_size=4, steps=3)
    mb_x2 = dag.markov_blanket("X_2")
    assert mb_x2.parents == frozenset({"X_1"})
    assert mb_x2.children == frozenset({"X_3", "A_2_0", "A_2_1", "A_2_2", "A_2_3"})
    assert mb_x2.co_parents == frozenset()
    assert "X_0" not in mb_x2.members
    assert "A_1_0" not in mb_x2.members
    assert "A_3_0" not in mb_x2.members

    mb_alert = dag.markov_blanket("A_2_1")
    assert mb_alert.parents == frozenset({"X_2"})
    assert mb_alert.children == frozenset()
    assert mb_alert.co_parents == frozenset()


def test_markov_blanket_rejects_out_of_range_machine() -> None:
    with pytest.raises(ValueError):
        markov_blanket(diamond_network(), 99)

