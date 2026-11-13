import pytest

from graphrag_juridico import config


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch, tmp_path):
    """Cada teste lê um ambiente controlado, sem o .env real."""
    monkeypatch.setattr(config, "ROOT", tmp_path)
    for var in ["NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD", "OPENROUTER_KEY", "LLM_BASE_URL",
                "CYPHER_MODEL", "ANSWER_MODEL", "EMBEDDING_MODEL"]:
        monkeypatch.delenv(var, raising=False)
    config.get_settings.cache_clear()
    yield
    config.get_settings.cache_clear()


def _set_neo4j(monkeypatch):
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_USER", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "secret-pass")


def test_missing_neo4j_variable_has_clear_error():
    with pytest.raises(RuntimeError, match="NEO4J_URI"):
        config.get_settings()


def test_defaults_point_to_openrouter(monkeypatch):
    _set_neo4j(monkeypatch)
    s = config.get_settings()

    assert s.llm_base_url == config.OPENROUTER_BASE_URL
    assert s.cypher_model == s.answer_model == config.DEFAULT_LLM_MODEL
    assert s.embedding_model == "baai/bge-m3"


def test_llm_provider_can_be_overridden(monkeypatch):
    _set_neo4j(monkeypatch)
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("CYPHER_MODEL", "outro/modelo")

    s = config.get_settings()

    assert s.llm_base_url == "http://localhost:11434/v1"
    assert s.cypher_model == "outro/modelo"
