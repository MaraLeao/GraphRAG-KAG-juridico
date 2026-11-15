import pytest

from graphrag_juridico.entities import institution_acronyms, keywords, mentioned, merge, norm_ids_in


@pytest.mark.parametrize("name, expected", [
    ("erro médico", True),              # acentos e ordem não importam
    ("Supremo Tribunal Federal", False),
    ("dignidade sexual", False),        # parecido no vetor, mas não citado
    ("a", False),                       # sem palavras com conteúdo
])
def test_mentioned_requires_all_words_of_the_name(name, expected):
    assert mentioned(name, "Como a responsabilidade civil se aplica ao ERRO MEDICO na dignidade?") is expected


@pytest.mark.parametrize("question, expected", [
    ("Qual a relação entre a LGPD e a dignidade da pessoa humana?", ["lei:13709:2018"]),
    ("O que a Lei Geral de Proteção de Dados diz sobre consentimento?", ["lei:13709:2018"]),
    ("O que diz a Lei nº 13.709/2018?", ["lei:13709:2018"]),
    ("A Lei Complementar 101/2000 limita gastos?", ["lc:101:2000"]),
    ("O CDC se aplica junto com a CF/88?", ["codigo:cdc:1990", "const:1988"]),
    ("Como o Código de Processo Civil trata a prova?", ["codigo:cpc:2015"]),
])
def test_norms_found_by_alias_or_number(question, expected):
    assert norm_ids_in(question) == expected


def test_alias_must_be_a_whole_word():
    # "eca" dentro de "biblioteca" e "cf" dentro de outra palavra não contam
    assert norm_ids_in("A biblioteca do tribunal cfx") == []


def test_same_norm_is_returned_once():
    assert norm_ids_in("LGPD, ou Lei 13.709/2018") == ["lei:13709:2018"]


def test_institution_acronyms_skip_norm_aliases():
    assert institution_acronyms("O STF e a ANPD interpretam a LGPD?") == ["STF", "ANPD"]


def test_keywords_drop_short_and_empty_words():
    assert keywords("Qual a relação entre a LGPD e a dignidade da pessoa humana?") == [
        "lgpd", "dignidade", "pessoa", "humana"]


def entity(eid, name, via):
    return {"id": eid, "name": name, "via": via}


def test_merge_keeps_first_occurrence_and_limit():
    alias = [entity("n1", "Lei 13.709/2018", "apelido")]
    vector = [entity("c1", "privacidade", "vetor"), entity("n1", "Lei 13.709/2018", "vetor")]
    fulltext = [entity("c2", "consentimento", "fulltext"), entity("c3", "dados pessoais", "fulltext")]

    merged = merge(alias, vector, fulltext, limit=3)

    assert [(e["id"], e["via"]) for e in merged] == [("n1", "apelido"), ("c1", "vetor"), ("c2", "fulltext")]


def test_merge_skips_variations_of_a_chosen_name():
    vector = [entity("c1", "dignidade da pessoa humana", "vetor"), entity("c2", "dignidade da pessoa", "vetor"),
              entity("c3", "Consentimento", "vetor"), entity("c4", "consentimento informado", "vetor"),
              entity("c5", "dignidade da vida humana", "vetor")]

    assert [e["id"] for e in merge(vector)] == ["c1", "c3", "c5"]


def test_merge_only_compares_whole_words():
    vector = [entity("c1", "pena", "vetor"), entity("c2", "penal", "vetor")]

    assert [e["id"] for e in merge(vector)] == ["c1", "c2"]
