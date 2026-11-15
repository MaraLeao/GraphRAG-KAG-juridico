import json

import pytest

from graphrag_juridico import evaluation
from graphrag_juridico.contracts import AskResult, Confidence
from graphrag_juridico.evaluation import find_question, load_questions


def fake(answer):
    return lambda q: AskResult(question=q, answer=answer, confidence=Confidence(score=0.5))


@pytest.fixture
def sandbox(monkeypatch, tmp_path):
    """Resultados e relatório em pasta temporária; métodos falsos (sem banco nem LLM)."""
    monkeypatch.setattr(evaluation, "RESULTS_FILE", tmp_path / "results.json")
    monkeypatch.setattr(evaluation, "REPORT_DIR", tmp_path / "evaluation")
    monkeypatch.setattr(evaluation, "METHODS", {
        "llm": ("LLM puro", fake("Pelo art. 6º da Lei 8.078/1990, a resposta é sim.")),
        "graphrag": ("GraphRAG/KAG", fake("Não há informação suficiente no corpus para responder.")),
    })
    return tmp_path


def test_question_set_has_answerable_and_trap_questions():
    questions = load_questions()

    assert len(questions) == 10 and len({q.id for q in questions}) == 10
    assert sum(q.should_abstain for q in questions) == 3
    assert all(q.expected for q in questions if not q.should_abstain)


def test_find_question_ignores_case_and_question_mark():
    assert find_question("quais instituições interpretam a LGPD").id == "E2"
    assert find_question("Pergunta fora do conjunto") is None


def test_run_all_records_every_method_and_resumes(sandbox):
    assert evaluation.run_all(only={"E2", "A1"}) == []

    saved = json.loads((sandbox / "results.json").read_text("utf-8"))
    assert set(saved) == {"E2", "A1"} and set(saved["E2"]) == {"llm", "graphrag"}
    assert evaluation.run_all(only={"E2", "A1"}) == []          # nada a refazer


def test_run_all_stops_at_the_first_error_keeping_what_was_saved(sandbox, monkeypatch):
    def quota_exceeded(question):
        raise RuntimeError("429 rate limit")

    monkeypatch.setitem(evaluation.METHODS, "graphrag", ("GraphRAG/KAG", quota_exceeded))

    assert evaluation.run_all(only={"E2"}) == ["E2/graphrag"]
    assert set(json.loads((sandbox / "results.json").read_text("utf-8"))["E2"]) == {"llm"}


def test_summary_and_report(sandbox):
    evaluation.run_all(only={"E2", "A1"})

    agg = evaluation.summary(evaluation.score())
    report = evaluation.write_report()

    assert agg["graphrag"]["trap_abstention"] == 1.0 and agg["graphrag"]["wrong_abstention"] == 1.0
    assert agg["llm"]["with_unsupported_norms"] == 1.0          # cita lei sem contexto nenhum
    assert "| Acerto (gabarito) |" in report.read_text("utf-8")
    assert (sandbox / "evaluation" / "metrics.png").stat().st_size > 0


def test_empty_answer_is_an_error_and_is_not_recorded(sandbox, monkeypatch):
    monkeypatch.setitem(evaluation.METHODS, "llm", ("LLM puro", fake("   ")))

    assert evaluation.run_all(only={"E2"}) == ["E2/llm"]
    assert not (sandbox / "results.json").exists()


def test_context_norms_are_computed_before_chunks_are_truncated(sandbox, monkeypatch):
    from graphrag_juridico.contracts import ChunkRef

    def with_long_chunk(question):
        text = "palavra " * 200 + "e o Código de Defesa do Consumidor também protege a privacidade."
        return AskResult(question=question, answer="O CDC protege a privacidade [C1].",
                         chunks=[ChunkRef(id="a:1", text=text, article_title="Artigo")],
                         confidence=Confidence(score=0.5))

    monkeypatch.setitem(evaluation.METHODS, "graphrag", ("GraphRAG/KAG", with_long_chunk))
    evaluation.run_all(only={"E1"})

    saved = json.loads((sandbox / "results.json").read_text("utf-8"))["E1"]["graphrag"]
    assert "Consumidor" not in saved["result"]["chunks"][0]["text"]       # trecho cortado na gravação
    assert saved["context_norms"] == ["codigo:cdc:1990"]
    row = next(r for r in evaluation.score() if r["method"] == "graphrag")
    assert row["unsupported_norms"] == []                                 # sem falso positivo
