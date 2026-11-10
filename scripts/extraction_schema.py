"""Schema (Pydantic) da saída do LLM -- espelha os nós e relações do plano_grafo_neo4j.md."""
from typing import Literal, Optional
from pydantic import BaseModel, Field

EntityType = Literal["Norm", "Institution", "Concept", "Scholar"]
RelationType = str  # validado contra RELATION_RULES (tipos fora da lista são descartados, não derrubam o chunk)

# (origem, destino) permitidos por relação -- o validador descarta o resto
RELATION_RULES = {
    "REGULATES":   ("Norm", "Concept"),
    "AMENDS":      ("Norm", "Norm"),
    "REVOKES":     ("Norm", "Norm"),
    "ISSUED_BY":   ("Norm", "Institution"),
    "INTERPRETS":  ("Institution", "Norm"),
    "SPECIALIZES": ("Concept", "Concept"),
}


class Entity(BaseModel):
    type: EntityType
    name: str = Field(description="Nome canônico, como aparece no texto")
    norm_type: Optional[str] = Field(None, description="Só Norm: lei, código, constituição, resolução, súmula, tratado")
    number: Optional[str] = Field(None, description="Só Norm: número, ex. '11.105'")
    year: Optional[int] = Field(None, description="Só Norm: ano")
    acronym: Optional[str] = Field(None, description="Só Institution: sigla")
    evidence: str = Field(description="Trecho LITERAL do chunk que sustenta a entidade")


class Relation(BaseModel):
    type: RelationType
    source: str = Field(description="name de uma entidade listada em entities")
    target: str = Field(description="name de uma entidade listada em entities")
    evidence: str = Field(description="Trecho LITERAL do chunk que sustenta a relação")


class Extraction(BaseModel):
    entities: list[Entity] = []
    relations: list[Relation] = []
