import json

import pytest

from graphrag_juridico import mock
from graphrag_juridico.contracts import ChunkRef
from graphrag_juridico.mock import DEMO_QUESTIONS, is_recorded, mock_ask


def test_every_demo_question_has_a_recorded_real_run():
    assert all(is_recorded(q) for q in DEMO_QUESTIONS)


def test_recorded_runs_are_different_and_follow_the_contract():
    results = [mock_ask(q) for q in DEMO_QUESTIONS]

    assert len({r.answer for r in results}) == len(DEMO_QUESTIONS)
    assert all(r.chunks and r.nodes and r.paths for r in results)


def test_matching_ignores_case_spaces_and_question_mark():
    assert is_recorded("  qual a relação entre a LGPD e a  dignidade da pessoa humana ")


def test_other_questions_get_the_fixed_example():
    result = mock_ask("Pergunta que não foi gravada")

    assert not is_recorded("Pergunta que não foi gravada")
    assert result.question == "Pergunta que não foi gravada" and "[C1]" in result.answer


@pytest.fixture
def demo_file(monkeypatch, tmp_path):
    path = tmp_path / "demo.json"
    monkeypatch.setattr(mock, "DEMO_FILE", path)
    monkeypatch.setattr(mock, "EVALUATION_FILE", tmp_path / "sem_avaliacao.json")
    mock.recorded.cache_clear()
    yield path
    mock.recorded.cache_clear()


def test_record_saves_runs_truncates_chunks_and_keeps_old_ones_on_failure(monkeypatch, demo_file):
    def fake_ask(question, mock):
        if question == "falha":
            raise RuntimeError("429 rate limit")
        result = mock_ask("q")
        result.question = question
        result.chunks = [ChunkRef(id="x:1", text="palavra " * 500, article_title="Artigo")]
        return result

    monkeypatch.setattr("graphrag_juridico.ask.ask", fake_ask)

    failed = mock.record(["Primeira?", "falha"])

    saved = json.loads(demo_file.read_text("utf-8"))
    assert failed == ["falha"] and [r["question"] for r in saved] == ["Primeira?"]
    assert len(saved[0]["chunks"][0]["text"]) <= mock.MAX_CHUNK_CHARS + len(" [...]")
    assert is_recorded("primeira")
