"""Respostas do modo demonstração, no formato do contrato, sem depender do Neo4j nem do LLM.

- Perguntas de exemplo (`DEMO_QUESTIONS`): reproduzem uma execução REAL gravada
  (resposta do LLM, subgrafo, caminhos, Cypher e confiança), em `resources/demo_answers.json`.
  Os textos dos trechos são cortados.
- Qualquer outra pergunta: uma resposta de exemplo fixa, escrita à mão com nomes, ids, links,
  PageRank e comunidades do grafo real.

Gravar de novo (usa o banco e o LLM: 2 a 3 requisições por pergunta):
    python -m graphrag_juridico.mock --record
"""
import argparse
import json
from functools import lru_cache
from pathlib import Path

from .contracts import AskResult, ChunkRef, Confidence, GraphEdge, GraphNode

LGPD = "lei:13709:2018"
DEMO_FILE = Path(__file__).parent / "resources" / "demo_answers.json"
DEMO_QUESTIONS = [
    "Qual a relação entre a LGPD e a dignidade da pessoa humana?",
    "Quais instituições interpretam normas que regulam o consentimento?",
    "Como a responsabilidade civil se aplica ao erro médico?",
    "Qual a alíquota de exportação de soja prevista na LGPD?",
]
MAX_CHUNK_CHARS = 700   # trechos gravados são cortados: o texto completo dos artigos fica no banco


def _key(question: str) -> str:
    return " ".join(question.lower().split()).rstrip("?").strip()


@lru_cache(maxsize=1)
def recorded() -> dict[str, AskResult]:
    """Execuções reais gravadas, indexadas pela pergunta normalizada."""
    if not DEMO_FILE.exists():
        return {}
    return {_key(r["question"]): AskResult.model_validate(r) for r in json.loads(DEMO_FILE.read_text("utf-8"))}


def is_recorded(question: str) -> bool:
    return _key(question) in recorded()


def mock_ask(question: str) -> AskResult:
    if is_recorded(question):
        return recorded()[_key(question)].model_copy(update={"question": question})
    return fixed_example(question)


def fixed_example(question: str) -> AskResult:
    return AskResult(
        question=question,
        answer=(
            "Segundo o corpus, a LGPD (Lei 13.709/2018) define o consentimento como a manifestação livre, "
            "informada e inequívoca pela qual o titular concorda com o tratamento de seus dados [C1]. "
            "Ele é a primeira hipótese que autoriza o tratamento de dados pessoais (art. 7º, I) [C2]. "
            "Caminho no grafo: (Lei 13.709/2018) -[REGULATES]-> (consentimento); "
            "(Agência Nacional de Proteção de Dados) -[INTERPRETS]-> (Lei 13.709/2018)."
        ),
        chunks=[
            ChunkRef(id="38cad1b3e69d:7", article_title="Os impactos da Lei Geral de Proteção de Dados no Direito Coletivo do Trabalho",
                     text="O consentimento é a manifestação livre, informada e inequívoca pela qual o titular concorda...",
                     link="https://ambitojuridico.com.br/cadernos/direito-do-trabalho/lei-geral-de-protecao-de-dados-direito-do-trabalho/",
                     score=0.82),
            ChunkRef(id="1842069a9b0e:6", article_title="A proteção de dados extensível a pessoa jurídica de direito privado",
                     text="O art. 7º, inciso I, da Lei 13.709/2018 prevê o fornecimento de consentimento pelo titular...",
                     link="https://ambitojuridico.com.br/cadernos/direito-civil/a-protecao-de-dados-extensivel-a-pessoa-juridica-de-direito-privado/",
                     score=0.77),
        ],
        nodes=[
            GraphNode(id=LGPD, label="Norm", name="Lei 13.709/2018", pagerank=0.000115, community=7),
            GraphNode(id="consentimento", label="Concept", name="consentimento", pagerank=0.00015, community=7),
            GraphNode(id="privacidade", label="Concept", name="privacidade", pagerank=0.00082, community=0),
            GraphNode(id="anpd", label="Institution", name="Agência Nacional de Proteção de Dados", pagerank=0.000064, community=7),
        ],
        edges=[
            GraphEdge(source=LGPD, target="consentimento", type="REGULATES"),
            GraphEdge(source=LGPD, target="privacidade", type="REGULATES"),
            GraphEdge(source="anpd", target=LGPD, type="INTERPRETS"),
        ],
        paths=[
            ["Lei 13.709/2018", "REGULATES", "consentimento"],
            ["Agência Nacional de Proteção de Dados", "INTERPRETS", "Lei 13.709/2018"],
        ],
        cypher=[
            "MATCH (n:Norm {id: 'lei:13709:2018'})-[:REGULATES]->(c:Concept) "
            "WHERE toLower(c.name) CONTAINS 'consentimento' RETURN n.name, c.name LIMIT 25"
        ],
        confidence=Confidence(score=0.78, components={"vector": 0.82, "fulltext": 0.70, "path": 1.0, "citations": 1.0}),
    )


def record(questions: list[str] = DEMO_QUESTIONS) -> list[str]:
    """Roda o pipeline real para cada pergunta e grava o resultado; devolve as que falharam."""
    from .ask import ask  # import tardio: ask importa este módulo

    saved = {_key(r.question): r for r in recorded().values()}
    failed = []
    for question in questions:
        try:
            result = ask(question, mock=False)
        except Exception as error:  # cota esgotada, banco fora do ar: mantém a gravação anterior
            print(f"FALHOU: {question} ({type(error).__name__}: {error})")
            failed.append(question)
            continue
        for chunk in result.chunks:
            if len(chunk.text) > MAX_CHUNK_CHARS:
                chunk.text = chunk.text[:MAX_CHUNK_CHARS].rsplit(" ", 1)[0] + " [...]"
        saved[_key(question)] = result
        print(f"ok: {question} (confiança {result.confidence.score:.2f})")
    ordered = [saved[_key(q)] for q in questions if _key(q) in saved]
    ordered += [r for k, r in saved.items() if k not in {_key(q) for q in questions}]
    DEMO_FILE.write_text(json.dumps([r.model_dump() for r in ordered], ensure_ascii=False, indent=1), "utf-8")
    recorded.cache_clear()
    return failed


def main() -> None:
    parser = argparse.ArgumentParser(description="Respostas do modo demonstração.")
    parser.add_argument("--record", action="store_true", help="grava execuções reais das perguntas de exemplo")
    args = parser.parse_args()
    if args.record:
        record()
    else:
        for question in DEMO_QUESTIONS:
            print(("gravada  " if is_recorded(question) else "faltando ") + question)


if __name__ == "__main__":
    main()
