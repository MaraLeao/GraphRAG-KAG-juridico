import pytest

from graphrag_juridico import multihop
from graphrag_juridico.multihop import Subgraph, confidence, path_coverage, path_to_text


def node(nid, label, name, key=None):
    return {"id": nid, "key": key or name, "label": label, "name": name, "pagerank": None, "community": None}


LGPD = node("e1", "Norm", "Lei 13.709/2018", key="lei:13709:2018")
CHUNK = node("e2", "Chunk", "art7:3", key="art7:3")
DIGNIDADE = node("e3", "Concept", "dignidade da pessoa humana")
PRIVACIDADE = node("e4", "Concept", "privacidade")

# LGPD <-MENTIONS- trecho -MENTIONS-> dignidade (2 saltos passando por um trecho)
VIA_CHUNK = ([LGPD, CHUNK, DIGNIDADE],
             [{"source": "e2", "target": "e1", "type": "MENTIONS"},
              {"source": "e2", "target": "e3", "type": "MENTIONS"}])
REGULATES = ([LGPD, PRIVACIDADE], [{"source": "e1", "target": "e4", "type": "REGULATES"}])


def test_path_to_text_shows_direction_and_chunk_ids():
    assert path_to_text(*VIA_CHUNK) == [
        "Lei 13.709/2018", "<-[MENTIONS]-", "trecho art7:3", "-[MENTIONS]->", "dignidade da pessoa humana"]


def test_subgraph_deduplicates_nodes_edges_and_paths():
    sub = Subgraph()
    sub.add(*VIA_CHUNK)
    sub.add(*REGULATES)
    sub.add(*REGULATES)

    assert set(sub.nodes) == {"e1", "e2", "e3", "e4"}
    assert len(sub.edges) == 3
    assert len(sub.paths) == 2
    assert sub.chunk_keys() == ["art7:3"]


def test_confidence_is_weighted_mean_of_available_components():
    result = confidence({"vector": 1.0, "fulltext": 0.0, "path": None})
    w = multihop.WEIGHTS

    assert result.score == pytest.approx(w["vector"] / (w["vector"] + w["fulltext"]), abs=1e-4)
    assert set(result.components) == {"vector", "fulltext"}


def test_confidence_clamps_components_and_handles_empty():
    assert confidence({"vector": 1.7}).score == 1.0
    assert confidence({}).score == 0.0


@pytest.mark.parametrize("entities, connected, expanded, expected", [
    (3, 2, True, 2 / 3),    # 3 pares, 2 ligados
    (2, 0, True, 0.0),      # par sem caminho
    (1, 0, True, 1.0),      # entidade única com vizinhos
    (1, 0, False, 0.0),
    (0, 0, False, 0.0),
])
def test_path_coverage(entities, connected, expanded, expected):
    assert path_coverage(entities, connected, expanded) == pytest.approx(expected)


@pytest.fixture
def offline(monkeypatch):
    """Substitui o banco e o embedding por dublês; devolve as opções recebidas pela expansão."""
    calls = {"shortest_path": 0, "expand_options": None}

    def fake_shortest_path(s, t, b):
        calls["shortest_path"] += 1
        return {"nodes": VIA_CHUNK[0], "rels": VIA_CHUNK[1]}

    def fake_expand(ids, b, options):
        calls["expand_options"] = options
        return [{"nodes": REGULATES[0], "rels": REGULATES[1]}]

    monkeypatch.setattr(multihop, "embed_question", lambda q: [0.1])
    monkeypatch.setattr(multihop, "link_entities", lambda q, vec: [
        {"id": "e1", "label": "Norm", "name": "Lei 13.709/2018", "via": "apelido"},
        {"id": "e3", "label": "Concept", "name": "dignidade da pessoa humana", "via": "vetor"}])
    monkeypatch.setattr(multihop, "supernodes", lambda: frozenset())
    monkeypatch.setattr(multihop, "shortest_path", fake_shortest_path)
    monkeypatch.setattr(multihop, "expand", fake_expand)
    monkeypatch.setattr(multihop, "hybrid_rankings", lambda q, vec: (["art9:1"], ["art7:3"]))
    monkeypatch.setattr(multihop, "entity_chunk_ids", lambda ids, vec: ["art7:3", "art8:2"])
    monkeypatch.setattr(multihop, "fetch_chunks", lambda ids, vec: [
        {"id": cid, "text": "...", "article_id": cid.split(":")[0], "title": cid, "link": None, "vector_score": 0.7}
        for cid in ids])
    return calls


def test_retrieve_joins_paths_entities_and_hybrid_search(offline):
    """Monta o contexto sem banco: os trechos dos caminhos entram no contexto e a confiança é calculada."""
    ctx = multihop.retrieve("Qual a relação entre a LGPD e a dignidade da pessoa humana?", k=3)

    assert ctx.chunks[0].id == "art7:3"     # aparece no caminho, nas entidades e no fulltext
    assert {c.id for c in ctx.chunks} == {"art7:3", "art8:2", "art9:1"}
    assert len(ctx.paths) == 2 and len(ctx.nodes) == 4
    assert ctx.confidence.components["path"] == 1.0
    assert 0.0 < ctx.confidence.score <= 1.0
    assert ctx.options == multihop.Options() and len(ctx.cypher) == 2


def test_algorithms_can_be_turned_off(offline):
    options = multihop.Options(use_shortest_path=False, use_pagerank=False, use_communities=False)

    ctx = multihop.retrieve("pergunta", k=3, options=options)

    assert offline["shortest_path"] == 0
    assert offline["expand_options"] == options          # PageRank/Louvain desligados chegam à consulta
    assert "path" not in ctx.confidence.components       # sem Shortest Path, a cobertura não é medida
    assert len(ctx.paths) == 1 and len(ctx.cypher) == 1  # só a expansão
