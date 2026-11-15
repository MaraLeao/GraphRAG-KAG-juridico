from types import SimpleNamespace

from graphrag_juridico import baselines


def test_llm_only_sends_just_the_question_without_context(monkeypatch):
    sent = []
    monkeypatch.setattr(baselines, "chat", lambda messages, model: sent.append(messages) or "A LGPD é a Lei 13.709/2018.")
    monkeypatch.setattr(baselines, "get_settings", lambda: SimpleNamespace(answer_model="modelo"))

    result = baselines.llm_only("O que é a LGPD?")

    assert sent[0][-1] == {"role": "user", "content": "O que é a LGPD?"}
    assert result.answer.startswith("A LGPD") and not result.chunks and result.confidence.score == 0.0


def test_vector_rag_uses_only_vector_chunks_and_the_grounded_prompt(monkeypatch):
    prompts = []
    monkeypatch.setattr(baselines, "embed_question", lambda q: [0.1])
    monkeypatch.setattr(baselines, "read", lambda cypher, **p: [{"id": "a1:1"}, {"id": "a1:2"}, {"id": "a2:1"}])
    monkeypatch.setattr(baselines, "fetch_chunks", lambda ids, vec: [
        {"id": i, "text": f"texto {i}", "article_id": i.split(":")[0], "title": i, "link": None, "vector_score": 0.8}
        for i in ids])
    monkeypatch.setattr(baselines, "generate_answer", lambda q, ctx: prompts.append(ctx) or "Resposta [C1] e [C9].")

    result = baselines.vector_rag("pergunta", k=2)

    assert [c.id for c in result.chunks] == ["a1:1", "a1:2"]
    assert "[C1]" in prompts[0] and "CAMINHOS DO GRAFO:\n(nenhum)" in prompts[0] and "[Q]" not in prompts[0]
    assert result.confidence.components["citations"] == 0.5        # [C9] não existe no contexto
    assert not result.paths and not result.nodes
