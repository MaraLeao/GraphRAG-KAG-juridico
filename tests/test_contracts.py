import json
import re

import pytest
from pydantic import ValidationError

from graphrag_juridico.ask import ask
from graphrag_juridico.contracts import AskResult, Confidence


def test_mock_ask_follows_contract():
    result = ask("O que a LGPD diz sobre consentimento?", mock=True)

    assert isinstance(result, AskResult)
    assert result.question == "O que a LGPD diz sobre consentimento?"
    assert result.chunks and result.nodes and result.edges and result.cypher
    assert 0.0 <= result.confidence.score <= 1.0


def test_mock_edges_reference_existing_nodes():
    result = ask("qualquer pergunta", mock=True)
    node_ids = {n.id for n in result.nodes}

    assert all(e.source in node_ids and e.target in node_ids for e in result.edges)


def test_mock_citations_point_to_existing_chunks():
    result = ask("qualquer pergunta", mock=True)
    cited = {int(c) for c in re.findall(r"\[C(\d+)\]", result.answer)}

    assert cited and all(1 <= c <= len(result.chunks) for c in cited)


def test_result_is_json_serializable():
    data = json.loads(ask("pergunta", mock=True).model_dump_json())

    assert set(data) == {"question", "answer", "chunks", "nodes", "edges", "paths", "cypher", "confidence"}


def test_env_var_enables_mock(monkeypatch):
    monkeypatch.setenv("GRAPHRAG_MOCK", "1")

    assert ask("pergunta").answer


def test_real_ask_not_integrated_yet(monkeypatch):
    monkeypatch.delenv("GRAPHRAG_MOCK", raising=False)

    with pytest.raises(NotImplementedError):
        ask("pergunta")


@pytest.mark.parametrize("score", [-0.1, 1.5])
def test_confidence_score_must_be_between_0_and_1(score):
    with pytest.raises(ValidationError):
        Confidence(score=score)
