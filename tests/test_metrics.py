import pytest

from graphrag_juridico.contracts import AskResult, ChunkRef, Confidence
from graphrag_juridico.metrics import (Reference, evaluate, glossary, glossary_entries, grounded_fraction,
                                       norm_label, term_recall, unsupported_norms)

GENERATED = "MATCH (i:Institution)-[:INTERPRETS]->(n:Norm) RETURN DISTINCT i.name LIMIT 25"


def result(answer, chunks=(), paths=(), cypher=(), rows=(), question="pergunta"):
    return AskResult(question=question, answer=answer,
                     chunks=[ChunkRef(id=f"a:{i}", text=t, article_title="Artigo") for i, t in enumerate(chunks)],
                     paths=list(paths), cypher=list(cypher), cypher_rows=list(rows), confidence=Confidence(score=0.5))


def test_grounded_fraction_counts_sentences_with_citations():
    answer = "A LGPD protege os dados pessoais do titular [C1]. O consentimento deve ser livre e informado."

    assert grounded_fraction(answer) == 0.5


def test_abstention_without_claims_is_fully_grounded():
    assert grounded_fraction("Não há informação suficiente no corpus para responder.") == 1.0
    assert grounded_fraction("Sim.") == 0.0


def test_norms_missing_from_the_context_are_unsupported():
    r = result("Segundo a Lei 13.709/2018 e o CDC, o titular tem direitos.", chunks=["A Lei 13.709/2018 trata de dados."])

    assert unsupported_norms(r) == ["codigo:cdc:1990"]


def test_norms_from_query_rows_paths_and_question_are_supported():
    r = result("A ANPD interpreta a Lei 13.709/2018 e o Marco Civil da Internet; o CDC também.",
               paths=[["Lei 12.965/2014", "-[REGULATES]->", "privacidade"]],
               rows=[{"norma": "Lei 13.709/2018"}], question="O que diz o CDC?")

    assert unsupported_norms(r) == []


def test_llm_without_context_has_every_norm_unsupported_and_no_valid_citation():
    m = evaluate(result("A Lei 13.709/2018 define consentimento no art. 5º."))

    assert m.unsupported_norms == ["lei:13709:2018"]
    assert m.citations_valid == 0.0 and m.grounded_sentences == 0.0 and not m.abstained


def test_query_citation_is_valid_even_when_the_query_returned_nothing():
    m = evaluate(result("A consulta [Q] não retornou resultados.", cypher=[GENERATED]))

    assert m.citations_valid == 1.0


def test_term_recall_accepts_synonyms_and_ignores_accents():
    expected = [["Agência Nacional de Proteção de Dados", "ANPD"], ["Superior Tribunal de Justiça", "STJ"]]

    assert term_recall("A anpd interpreta a LGPD.", expected) == 0.5


@pytest.mark.parametrize("answer, reference, correct", [
    ("A ANPD interpreta a LGPD [Q].", Reference([["ANPD"]]), True),
    ("Não há informação suficiente no corpus.", Reference([["ANPD"]]), False),          # abstenção indevida
    ("Não há informação suficiente no corpus.", Reference([], should_abstain=True), True),
    ("A alíquota é de 10%.", Reference([], should_abstain=True), False),                 # respondeu a armadilha
])
def test_correctness_against_the_reference(answer, reference, correct):
    assert evaluate(result(answer, cypher=[GENERATED]), reference).correct is correct


@pytest.mark.parametrize("norm_id, label", [
    ("lei:8078:1990", "Lei 8.078/1990"), ("lc:101:2000", "Lei Complementar 101/2000"),
    ("codigo:cdc:1990", "CDC"), ("const:1988", "CF"), ("norm:desconhecida", "norm:desconhecida"),
])
def test_norm_label(norm_id, label):
    assert norm_label(norm_id) == label


def test_glossary_explains_known_labels_once_in_order():
    assert glossary(["CDC", "Lei 99.999/2099", "CF", "CDC"]) == [
        "CDC: Código de Defesa do Consumidor (Lei 8.078/1990)", "CF: Constituição Federal de 1988"]


def test_glossary_entries_keep_label_and_description_apart():
    assert glossary_entries(["CPP", "desconhecida"]) == [("CPP", "Código de Processo Penal (Decreto-Lei 3.689/1941)")]
