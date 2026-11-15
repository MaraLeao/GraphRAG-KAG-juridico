"""Métodos de comparação para o GraphRAG/KAG (`ask`), todos com o mesmo LLM (`ANSWER_MODEL`).

- `llm_only`: o LLM responde sozinho, sem contexto, com um prompt de assistente jurídico comum.
  É o uso "normal" de um LLM e a referência para medir alucinação.
- `vector_rag`: RAG tradicional. Mesmo prompt e mesmas regras de citação do GraphRAG, mas o contexto vem
  só da busca vetorial nos trechos (sem entidades, caminhos, algoritmos de grafo nem Cypher).
  A única diferença para o GraphRAG é o grafo.
"""
from .ask import build_context, check_citations, generate_answer
from .config import chat, get_settings
from .contracts import AskResult, ChunkRef, Confidence
from .db import read
from .multihop import Context, Options, confidence
from .retrieval import VECTOR_QUERY, diversify, embed_question, fetch_chunks

LLM_ONLY_SYSTEM = ("Você é um assistente jurídico especializado em Direito brasileiro. Responda à pergunta em "
                   "português, de forma objetiva, em até 250 palavras.")


def llm_only(question: str) -> AskResult:
    answer = chat([{"role": "system", "content": LLM_ONLY_SYSTEM}, {"role": "user", "content": question}],
                  model=get_settings().answer_model).strip()
    # sem contexto não há nada a citar: a confiança fica zerada de propósito
    return AskResult(question=question, answer=answer, confidence=Confidence(score=0.0))


def vector_rag(question: str, k: int = 8, per_article: int = 2) -> AskResult:
    vec = embed_question(question)
    ranked = [r["id"] for r in read(VECTOR_QUERY, n=k * 5, vec=vec)]
    rows = diversify(fetch_chunks(ranked, vec), k, per_article)
    chunks = [ChunkRef(id=r["id"], text=r["text"], article_title=r["title"], link=r["link"], score=r["vector_score"])
              for r in rows]
    ctx = Context(question=question, entities=[], chunks=chunks, nodes=[], edges=[], paths=[], cypher=[],
                  confidence=Confidence(score=0.0), options=Options(False, False, False))
    answer = generate_answer(question, build_context(ctx, None))

    citations, _ = check_citations(answer, len(chunks), 0, has_query=False)
    top = sorted((c.score or 0.0 for c in chunks), reverse=True)[:3]
    return AskResult(question=question, answer=answer, chunks=chunks, cypher=[VECTOR_QUERY.strip()],
                     confidence=confidence({"vector": sum(top) / len(top) if top else 0.0, "citations": citations}))
