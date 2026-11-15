import pytest

from graphrag_juridico import ask as ask_module
from graphrag_juridico.ask import NO_INFO, ask, build_context, check_citations, is_no_info
from graphrag_juridico.contracts import ChunkRef, Confidence, GraphEdge, GraphNode
from graphrag_juridico.multihop import Context, Options
from graphrag_juridico.text_to_cypher import CypherResult

GENERATED = "MATCH (i:Institution)-[:INTERPRETS]->(:Norm {id: 'lei:13709:2018'}) RETURN DISTINCT i.name LIMIT 25"


def context(n_chunks=2):
    return Context(
        question="pergunta",
        entities=[{"id": "e1", "label": "Norm", "name": "Lei 13.709/2018", "via": "apelido"}],
        chunks=[ChunkRef(id=f"a{i}:1", text=f"texto {i}", article_title=f"Artigo {i}", score=0.8)
                for i in range(1, n_chunks + 1)],
        nodes=[GraphNode(id="e1", label="Norm", name="Lei 13.709/2018"),
               GraphNode(id="e2", label="Concept", name="privacidade")],
        edges=[GraphEdge(source="e1", target="e2", type="REGULATES")],
        paths=[["Lei 13.709/2018", "-[REGULATES]->", "privacidade"]],
        cypher=["UNWIND $ids AS sid ..."],
        confidence=Confidence(score=0.8, components={"vector": 0.8, "fulltext": 0.5, "path": 1.0}),
        options=Options(),
    )


def test_context_numbers_chunks_paths_and_query():
    cypher = CypherResult(question="q", cypher=GENERATED, rows=[{"i.name": "ANPD"}])

    text = build_context(context(), cypher)

    assert "[C1] (Artigo 1)\ntexto 1" in text and "[C2]" in text
    assert "[P1] Lei 13.709/2018 -[REGULATES]-> privacidade" in text
    assert "CONSULTA ESTRUTURADA [Q]" in text and "ANPD" in text


def test_context_without_query_and_with_empty_result():
    assert "[Q]" not in build_context(context(), None)
    assert "[Q]" not in build_context(context(), CypherResult(question="q"))  # Cypher falhou
    empty = CypherResult(question="q", cypher=GENERATED, rows=[])
    assert "não retornou resultados" in build_context(context(), empty)


@pytest.mark.parametrize("answer, has_query, expected_score, expected_invalid", [
    ("A LGPD regula a privacidade [C1][P1].", False, 1.0, []),
    ("Ver [C1], [C3] e [P2].", False, 1 / 3, ["[C3]", "[P2]"]),   # C3 e P2 não existem
    ("Segundo a consulta [Q].", True, 1.0, []),
    ("Segundo a consulta [Q].", False, 0.0, ["[Q]"]),            # citou [Q] sem consulta no contexto
    ("A LGPD protege dados.", False, 0.0, []),                    # afirmação sem nenhuma citação
    (f"{NO_INFO} O corpus não trata de soja.", False, 1.0, []),   # abstenção não precisa citar
])
def test_check_citations(answer, has_query, expected_score, expected_invalid):
    score, invalid = check_citations(answer, n_chunks=2, n_paths=1, has_query=has_query)

    assert score == pytest.approx(expected_score)
    assert invalid == expected_invalid


def test_no_info_detection_tolerates_missing_accents():
    assert is_no_info("Nao ha informacao suficiente no corpus.")
    assert not is_no_info("A LGPD trata do consentimento [C1].")


@pytest.fixture
def offline(monkeypatch):
    """Recuperação, Cypher e LLM substituídos por dublês; guarda o que o LLM recebeu."""
    calls = {"cypher": 0, "prompt": None, "options": None}

    def fake_retrieve(question, k, options):
        calls["options"] = options
        return context()

    def fake_cypher(question):
        calls["cypher"] += 1
        return CypherResult(question=question, cypher=GENERATED, rows=[{"i.name": "ANPD"}])

    def fake_answer(question, ctx_text):
        calls["prompt"] = ctx_text
        return "A ANPD interpreta a LGPD [Q][P1] e o tema aparece em [C1]."

    monkeypatch.delenv("GRAPHRAG_MOCK", raising=False)
    monkeypatch.setattr(ask_module, "retrieve", fake_retrieve)
    monkeypatch.setattr(ask_module, "run_cypher", fake_cypher)
    monkeypatch.setattr(ask_module, "generate_answer", fake_answer)
    return calls


def test_ask_joins_retrieval_cypher_and_answer(offline):
    result = ask("Quais instituições interpretam a LGPD?")

    assert result.answer.startswith("A ANPD")
    assert result.cypher[0] == GENERATED and len(result.cypher) == 2     # gerada + estrutural
    assert result.chunks and result.nodes and result.edges and result.paths
    assert result.confidence.components["citations"] == 1.0
    assert result.cypher_rows == [{"i.name": "ANPD"}]                 # resultado da [Q] fica rastreável
    assert "[Q]" in offline["prompt"] and "[C1]" in offline["prompt"]


def test_ask_without_cypher_skips_the_llm_query(offline):
    result = ask("pergunta", use_cypher=False)

    assert offline["cypher"] == 0
    assert result.cypher == ["UNWIND $ids AS sid ..."]
    assert result.confidence.components["citations"] < 1.0   # citou [Q] sem consulta no contexto


def test_ask_passes_algorithm_options_to_retrieval(offline):
    options = Options(use_pagerank=False)

    ask("pergunta", options=options)

    assert offline["options"] == options


def test_ask_output_is_valid_json_contract(offline):
    data = ask("pergunta").model_dump()

    assert set(data) == {"question", "answer", "chunks", "nodes", "edges", "paths", "cypher", "cypher_rows",
                         "confidence"}
    assert 0.0 <= data["confidence"]["score"] <= 1.0
