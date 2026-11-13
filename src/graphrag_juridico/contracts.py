"""Contrato de dados do sistema de perguntas.

`AskResult` é o que `ask()` devolve. A UI e a avaliação dependem só dele.
Mudanças aqui devem ser combinadas com o grupo antes.
"""
from pydantic import BaseModel, Field


class ChunkRef(BaseModel):
    """Trecho de artigo usado como contexto; citado na resposta como [C1], [C2]..."""
    id: str
    text: str
    article_title: str
    link: str | None = None
    score: float | None = None


class GraphNode(BaseModel):
    """Nó do grafo exibido na UI (Norm, Concept, Institution, Scholar, Article...)."""
    id: str
    label: str
    name: str
    pagerank: float | None = None
    community: int | None = None


class GraphEdge(BaseModel):
    """Relação entre dois nós (source/target são `GraphNode.id`)."""
    source: str
    target: str
    type: str


class Confidence(BaseModel):
    """Nível de confiança (0 a 1) e os componentes que o compõem."""
    score: float = Field(ge=0.0, le=1.0)
    components: dict[str, float] = {}


class AskResult(BaseModel):
    question: str
    answer: str
    chunks: list[ChunkRef] = []
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    paths: list[list[str]] = []          # ex.: ["Lei 13.709/2018", "REGULATES", "consentimento"]
    cypher: list[str] = []               # consultas Cypher executadas (para a UI)
    confidence: Confidence
