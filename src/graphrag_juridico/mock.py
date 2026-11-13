"""Resposta falsa, no formato do contrato, para desenvolver a UI e a avaliação
sem depender do Neo4j nem do LLM. Os nomes vêm do grafo real.
"""
from .contracts import AskResult, ChunkRef, Confidence, GraphEdge, GraphNode

LGPD = "lei:13709:2018"


def mock_ask(question: str) -> AskResult:
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
            ChunkRef(id="mock-article-1:3", article_title="Os impactos da Lei Geral de Proteção de Dados no Direito Coletivo do Trabalho",
                     text="O consentimento é a manifestação livre, informada e inequívoca pela qual o titular concorda...",
                     link="https://ambitojuridico.com.br/", score=0.82),
            ChunkRef(id="mock-article-2:1", article_title="A proteção de dados extensível a pessoa jurídica de direito privado",
                     text="O art. 7º, inciso I, da Lei 13.709/2018 prevê o fornecimento de consentimento pelo titular...",
                     link="https://ambitojuridico.com.br/", score=0.77),
        ],
        nodes=[
            GraphNode(id=LGPD, label="Norm", name="Lei 13.709/2018", pagerank=0.012, community=4),
            GraphNode(id="consentimento", label="Concept", name="consentimento", pagerank=0.004, community=4),
            GraphNode(id="privacidade", label="Concept", name="privacidade", pagerank=0.006, community=4),
            GraphNode(id="anpd", label="Institution", name="Agência Nacional de Proteção de Dados", pagerank=0.003, community=4),
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
