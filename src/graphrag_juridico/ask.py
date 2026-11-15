"""Orquestrador: pergunta -> recuperação (vetor + Cypher + multi-hop) -> resposta grounded.

Etapas:
1. recuperação multi-hop (`multihop.retrieve`): trechos [C1..Cn], caminhos do grafo [P1..Pn], subgrafo, confiança;
2. Text-to-Cypher (`text_to_cypher.run`): resultado estruturado [Q]; se falhar, segue só com o item 1;
3. o LLM responde usando APENAS esse contexto e citando [Cn], [Pn] e [Q], ou diz que não há informação;
4. as citações são conferidas: citar algo que não está no contexto derruba a confiança.

Para desenvolver a UI e a avaliação sem banco nem LLM, use o modo mock:
    ask("pergunta", mock=True)      ou      GRAPHRAG_MOCK=1 no ambiente

Uso:
    python -m graphrag_juridico.ask "Qual a relação entre a LGPD e a dignidade da pessoa humana?"
"""
import argparse
import os
import re

from .config import get_llm, get_settings
from .contracts import AskResult
from .mock import mock_ask
from .multihop import EXPAND_QUERY, SHORTEST_PATH_QUERY, Context, Options, confidence, retrieve
from .text_to_cypher import CypherResult
from .text_to_cypher import run as run_cypher

MAX_PATHS = 20       # caminhos enviados ao LLM (os mais relevantes vêm primeiro)
MAX_ROWS = 25        # linhas do resultado da Cypher enviadas ao LLM
NO_INFO = "Não há informação suficiente no corpus para responder."
CITATION = re.compile(r"\[(C|P)(\d+)\]|\[(Q)\]")
STRUCTURAL_QUERIES = {SHORTEST_PATH_QUERY.strip(), EXPAND_QUERY.strip()}  # da recuperação, não do LLM
NO_INFO_PATTERN = re.compile(r"n[ãa]o h[áa] informa[çc][ãa]o suficiente", re.IGNORECASE)

SYSTEM = f"""Você é um assistente jurídico que responde perguntas usando SOMENTE o contexto fornecido:
trechos de artigos [C1], [C2]..., caminhos do grafo de conhecimento [P1], [P2]... e, quando houver,
o resultado de uma consulta estruturada ao grafo [Q].

Regras:
- Toda afirmação deve citar a fonte entre colchetes: [C2], [P1], [Q]. Cite os caminhos do grafo que usar.
- Não use conhecimento externo, mesmo que você saiba a resposta.
- Se o contexto não responde à pergunta, escreva exatamente "{NO_INFO}" e explique em uma frase o que falta.
- Responda em português, de forma objetiva, em até 250 palavras."""


def build_context(ctx: Context, cypher: CypherResult | None) -> str:
    """Contexto numerado que o LLM recebe; os números são os mesmos que a UI mostra."""
    chunks = [f"[C{i}] ({c.article_title})\n{c.text}" for i, c in enumerate(ctx.chunks, 1)]
    paths = [f"[P{i}] " + " ".join(p) for i, p in enumerate(ctx.paths[:MAX_PATHS], 1)]
    parts = ["TRECHOS:\n" + ("\n\n".join(chunks) or "(nenhum)"),
             "CAMINHOS DO GRAFO:\n" + ("\n".join(paths) or "(nenhum)")]
    if cypher is not None and cypher.ok:
        rows = "\n".join(str(r) for r in cypher.rows[:MAX_ROWS]) or "(a consulta não retornou resultados)"
        parts.append(f"CONSULTA ESTRUTURADA [Q]:\n{cypher.cypher}\nResultado:\n{rows}")
    return "\n\n".join(parts)


def check_citations(answer: str, n_chunks: int, n_paths: int, has_query: bool) -> tuple[float | None, list[str]]:
    """Fração de citações que apontam para algo do contexto e a lista das inválidas.
    Sem nenhuma citação: 1.0 se a resposta é "não há informação" (nada a citar), senão 0.0."""
    valid, invalid = 0, []
    for kind, number, query in CITATION.findall(answer):
        ok = bool(query and has_query) if query else (1 <= int(number) <= (n_chunks if kind == "C" else n_paths))
        if ok:
            valid += 1
        else:
            invalid.append(f"[{kind}{number}]" if kind else "[Q]")
    total = valid + len(invalid)
    if not total:
        return (1.0 if is_no_info(answer) else 0.0), []
    return valid / total, invalid


def is_no_info(answer: str) -> bool:
    """A resposta declara que o corpus não tem a informação (abstenção)."""
    return bool(NO_INFO_PATTERN.search(answer))


def generate_answer(question: str, context: str) -> str:
    response = get_llm().chat.completions.create(
        model=get_settings().answer_model, temperature=0,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": f"{context}\n\nPERGUNTA: {question}"}])
    return (response.choices[0].message.content or "").strip()


def ask(question: str, *, mock: bool | None = None, options: Options = Options(),
        use_cypher: bool = True, k: int = 8) -> AskResult:
    """Responde à pergunta com citações; `options` liga/desliga cada algoritmo de grafo."""
    if mock is None:
        mock = os.getenv("GRAPHRAG_MOCK", "0") == "1"
    if mock:
        return mock_ask(question)

    ctx = retrieve(question, k=k, options=options)
    cypher = run_cypher(question) if use_cypher else None
    has_query = cypher is not None and cypher.ok
    answer = generate_answer(question, build_context(ctx, cypher))

    citations, _invalid = check_citations(answer, len(ctx.chunks), min(len(ctx.paths), MAX_PATHS), has_query)
    return AskResult(
        question=question,
        answer=answer,
        chunks=ctx.chunks,
        nodes=ctx.nodes,
        edges=ctx.edges,
        paths=ctx.paths[:MAX_PATHS],
        cypher=([cypher.cypher] if has_query else []) + ctx.cypher,
        confidence=confidence({**ctx.confidence.components, "citations": citations}),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Pergunta ao GraphRAG (usa o LLM: 2 a 3 requisições).")
    parser.add_argument("question")
    parser.add_argument("--no-cypher", action="store_true", help="não gera Cypher pelo LLM (economiza requisições)")
    args = parser.parse_args()

    result = ask(args.question, use_cypher=not args.no_cypher)
    print(result.answer)
    print(f"\nConfiança: {result.confidence.score:.2f} {result.confidence.components}")
    print("\nFontes:")
    for i, c in enumerate(result.chunks, 1):
        print(f"  [C{i}] {c.article_title} - {c.link}")
    for i, p in enumerate(result.paths, 1):
        print(f"  [P{i}] {' '.join(p)}")
    generated = [q for q in result.cypher if q not in STRUCTURAL_QUERIES]
    if generated:
        print(f"  [Q] {generated[0]}")


if __name__ == "__main__":
    main()
