from types import SimpleNamespace

import pytest

from graphrag_juridico import config


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch, tmp_path):
    """Cada teste lê um ambiente controlado, sem o .env real."""
    monkeypatch.setattr(config, "ROOT", tmp_path)
    for var in ["NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD", "OPENROUTER_KEY", "FALLBACK_OPENROUTER_KEY", "LLM_BASE_URL",
                "CYPHER_MODEL", "ANSWER_MODEL", "EMBEDDING_MODEL"]:
        monkeypatch.delenv(var, raising=False)
    config.get_settings.cache_clear()
    monkeypatch.setattr(config, "_active_key", 0)
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


class QuotaError(Exception):
    status_code = 429


class FakeClient:
    """Dublê do cliente OpenAI: sem cota (429) ou respondendo com o nome da chave."""

    def __init__(self, key, exhausted):
        self.key, self.exhausted, self.calls = key, exhausted, 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, model, temperature, messages):
        self.calls += 1
        if self.key in self.exhausted:
            raise QuotaError("Rate limit exceeded: free-models-per-day")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=f"resposta via {self.key}"))])


@pytest.fixture
def two_keys(monkeypatch):
    _set_neo4j(monkeypatch)
    monkeypatch.setenv("OPENROUTER_KEY", "principal")
    monkeypatch.setenv("FALLBACK_OPENROUTER_KEY", "reserva")
    exhausted, clients = set(), {}
    monkeypatch.setattr(config, "_client", lambda key: clients.setdefault(key, FakeClient(key, exhausted)))
    return exhausted, clients


def test_chat_uses_the_main_key_while_it_has_quota(two_keys):
    assert config.chat([{"role": "user", "content": "oi"}], model="m") == "resposta via principal"


def test_chat_switches_to_the_fallback_key_and_stays_there(two_keys):
    exhausted, clients = two_keys
    exhausted.add("principal")

    assert config.chat([], model="m") == "resposta via reserva"
    assert config.chat([], model="m") == "resposta via reserva"
    assert clients["principal"].calls == 1                    # não insiste na chave esgotada


def test_chat_raises_when_every_key_is_exhausted(two_keys):
    exhausted, _ = two_keys
    exhausted.update({"principal", "reserva"})

    with pytest.raises(QuotaError):
        config.chat([], model="m")


def test_other_errors_do_not_trigger_the_fallback(two_keys, monkeypatch):
    def broken(key):
        raise ValueError("modelo inexistente")

    monkeypatch.setattr(config, "_client", lambda key: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: broken(key)))))

    with pytest.raises(ValueError):
        config.chat([], model="m")


def test_without_any_key_there_is_a_clear_error(monkeypatch):
    _set_neo4j(monkeypatch)

    with pytest.raises(RuntimeError, match="OPENROUTER_KEY"):
        config.chat([], model="m")
