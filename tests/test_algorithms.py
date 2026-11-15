import pytest

from graphrag_juridico.algorithms import build_graph, compute


def edge(source, target, weight=1):
    return {"source": source, "target": target, "weight": weight}


def two_clusters():
    """Dois triângulos (a, b, c) e (x, y, z) ligados por uma única aresta c -> x."""
    pairs = [("a", "b"), ("b", "c"), ("c", "a"), ("x", "y"), ("y", "z"), ("z", "x"), ("c", "x")]
    return build_graph([edge(s, t) for s, t in pairs])


def test_repeated_relations_sum_weights():
    graph = build_graph([edge("lei", "conceito", 2), edge("lei", "conceito", 3)])

    assert graph["lei"]["conceito"]["weight"] == 5


def test_pagerank_is_a_probability_distribution():
    scores = compute(two_clusters())

    assert sum(v["pagerank"] for v in scores.values()) == pytest.approx(1.0)


def test_most_cited_node_has_highest_pagerank():
    # várias normas apontando para o mesmo conceito
    graph = build_graph([edge(f"norma{i}", "dignidade") for i in range(5)] + [edge("norma0", "outro")])
    scores = compute(graph)

    assert max(scores, key=lambda n: scores[n]["pagerank"]) == "dignidade"


def test_louvain_separates_clusters():
    scores = compute(two_clusters())
    community = {n: v["community"] for n, v in scores.items()}

    assert community["a"] == community["b"] == community["c"]
    assert community["x"] == community["y"] == community["z"]
    assert community["a"] != community["x"]


def test_communities_numbered_by_size():
    graph = two_clusters()
    graph.add_edge("z", "w", weight=1)  # cluster (x, y, z) fica maior
    scores = compute(graph)

    assert scores["x"]["community"] == 0


def test_results_are_deterministic():
    assert compute(two_clusters()) == compute(two_clusters())
