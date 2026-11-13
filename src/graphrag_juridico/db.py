"""Acesso ao Neo4j. Toda consulta do sistema de perguntas passa por `read()`,
que executa em transação SOMENTE LEITURA: o Neo4j recusa qualquer escrita nela,
mesmo que a Cypher venha de um LLM.
"""
from typing import Any

from .config import get_driver


def read(cypher: str, **params: Any) -> list[dict[str, Any]]:
    """Executa uma consulta de leitura e devolve as linhas como dicionários."""

    def work(tx):
        return [record.data() for record in tx.run(cypher, **params)]

    with get_driver().session() as session:
        return session.execute_read(work)
