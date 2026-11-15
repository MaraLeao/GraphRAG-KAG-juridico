import pytest

from graphrag_juridico import text_to_cypher as t2c
from graphrag_juridico.text_to_cypher import CypherValidationError, build_messages, extract_cypher, validate

VALID = """MATCH (i:Institution)-[:INTERPRETS]->(:Norm)-[:REGULATES]->(c:Concept)
WHERE toLower(c.name) CONTAINS 'consentimento'
RETURN DISTINCT i.name AS instituicao, count(*) AS n LIMIT 25"""


def test_extract_cypher_from_code_block():
    text = "Claro! Aqui está:\n```cypher\nMATCH (n:Norm) RETURN n.name LIMIT 5\n```\nEspero ter ajudado."

    assert extract_cypher(text) == "MATCH (n:Norm) RETURN n.name LIMIT 5"


def test_extract_cypher_without_block_uses_whole_text():
    assert extract_cypher("  MATCH (n) RETURN n LIMIT 1 ") == "MATCH (n) RETURN n LIMIT 1"


def test_valid_multi_hop_query_passes_unchanged():
    assert validate(VALID) == VALID


@pytest.mark.parametrize("cypher, reason", [
    ("MATCH (n:Norm) DETACH DELETE n", "DETACH"),
    ("MATCH (n:Concept) SET n.name = 'x' RETURN n", "SET"),
    ("MERGE (n:Concept {name: 'x'}) RETURN n", "MERGE"),
    ("LOAD CSV FROM 'file:///x.csv' AS row RETURN row", "LOAD"),
    ("MATCH (n) RETURN n; MATCH (m) DELETE m", "uma consulta"),
    ("CALL dbms.components() YIELD name RETURN name", "procedimento"),
    ("CALL apoc.cypher.runWrite('CREATE (n)', {}) YIELD value RETURN value", "procedimento"),
    ("MATCH (n:Lei) RETURN n.name", "label inexistente"),
    ("MATCH (n:Norm)-[:CITA]->(m) RETURN m.name", "relação inexistente"),
    ("MATCH (n:Concept)-[:SPECIALIZES*]->(m) RETURN m.name", "limite"),
    ("MATCH (n:Concept)-[:SPECIALIZES*1..]->(m) RETURN m.name", "limite"),
    ("MATCH (n:Concept)-[:SPECIALIZES*1..9]->(m) RETURN m.name", "limite"),
    ("MATCH (n:Concept)", "RETURN"),
])
def test_blocked_queries(cypher, reason):
    with pytest.raises(CypherValidationError, match=reason):
        validate(cypher)


def test_keywords_inside_strings_are_not_clauses():
    cypher = "MATCH (c:Concept) WHERE toLower(c.name) CONTAINS 'set de regras; delete' RETURN c.name LIMIT 5"

    assert validate(cypher) == cypher


def test_relation_names_that_contain_keywords_are_allowed():
    assert validate("MATCH (n:Norm)-[:REVOKES]->(m:Norm) RETURN m.name LIMIT 5")


def test_bounded_variable_length_and_index_procedures_are_allowed():
    assert validate("MATCH (a:Concept)-[:SPECIALIZES*1..3]->(b:Concept) RETURN b.name LIMIT 5")
    assert validate("CALL db.index.fulltext.queryNodes('ft_entities', 'lgpd') YIELD node RETURN node.name LIMIT 5")


def test_limit_is_added_when_missing():
    assert validate("MATCH (n:Norm) RETURN n.name;").endswith(f"\nLIMIT {t2c.DEFAULT_LIMIT}")


def test_large_limit_is_capped():
    assert validate("MATCH (n:Norm) RETURN n.name LIMIT 5000").endswith(f"LIMIT {t2c.MAX_LIMIT}")


def test_question_with_norm_alias_gets_id_hint():
    messages = build_messages("Quais instituições interpretam a LGPD?", failed=[])

    assert "lei:13709:2018" in messages[-1]["content"]


def test_failed_attempt_and_error_go_back_to_the_llm():
    failed = [{"cypher": "MATCH (n:Lei) RETURN n", "error": "label inexistente no schema: Lei"}]

    messages = build_messages("pergunta", failed)

    assert "MATCH (n:Lei)" in messages[-2]["content"] and "label inexistente" in messages[-1]["content"]


class FakeDB:
    """Dublê de `db.read`: EXPLAIN sempre passa, a não ser que a consulta contenha `broken`."""

    def __init__(self):
        self.executed = []

    def __call__(self, cypher, **params):
        if "broken" in cypher:
            raise RuntimeError("Invalid input 'broken'\n(detalhes do Neo4j)")
        if not cypher.startswith("EXPLAIN"):
            self.executed.append(cypher)
        return [] if cypher.startswith("EXPLAIN") else [{"instituicao": "Agência Nacional de Proteção de Dados"}]


@pytest.fixture
def fake_db(monkeypatch):
    db = FakeDB()
    monkeypatch.setattr(t2c, "read", db)
    return db


def fake_llm(monkeypatch, answers):
    calls = []

    def generate(messages):
        calls.append(messages)
        return answers[len(calls) - 1]

    monkeypatch.setattr(t2c, "generate", generate)
    return calls


def test_run_returns_rows_on_first_valid_query(monkeypatch, fake_db):
    fake_llm(monkeypatch, [VALID])

    result = t2c.run("Quais instituições interpretam normas que regulam o consentimento?")

    assert result.ok and result.cypher == VALID and not result.attempts
    assert result.rows == [{"instituicao": "Agência Nacional de Proteção de Dados"}]


def test_run_retries_with_the_error_and_never_executes_invalid_queries(monkeypatch, fake_db):
    calls = fake_llm(monkeypatch, ["MATCH (n:Norm) DELETE n", VALID])

    result = t2c.run("pergunta")

    assert result.ok and len(calls) == 2
    assert "DELETE" in calls[1][-1]["content"]          # o motivo da recusa voltou ao LLM
    assert fake_db.executed == [VALID]                  # a consulta de escrita nunca chegou ao banco


def test_run_gives_up_after_max_attempts(monkeypatch, fake_db):
    fake_llm(monkeypatch, ["MATCH (n:Norm) WHERE broken RETURN n LIMIT 1"] * 2)

    result = t2c.run("pergunta")

    assert not result.ok and result.rows == []
    assert len(result.attempts) == t2c.MAX_ATTEMPTS
    assert result.attempts[0]["error"] == "Invalid input 'broken'"   # só a 1ª linha do erro
