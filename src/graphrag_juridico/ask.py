"""Orquestrador: pergunta -> recuperação (vetor + Cypher + multi-hop) -> resposta grounded.

Enquanto a integração não fica pronta, use o modo mock:
    ask("pergunta", mock=True)      ou      GRAPHRAG_MOCK=1 no ambiente
"""
import os

from .contracts import AskResult
from .mock import mock_ask


def ask(question: str, *, mock: bool | None = None) -> AskResult:
    if mock is None:
        mock = os.getenv("GRAPHRAG_MOCK", "0") == "1"
    if mock:
        return mock_ask(question)
    # Pipeline previsto: busca híbrida em chunks -> entidades da pergunta -> Cypher gerada pelo LLM
    # -> multi-hop / Shortest Path -> resposta citando [Cn] e caminhos -> confiança
    raise NotImplementedError("ask() real ainda não integrado; use ask(..., mock=True) ou GRAPHRAG_MOCK=1.")
