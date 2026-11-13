"""Configuração única do projeto, lida do `.env` na raiz do repositório.

Uso:
    from graphrag_juridico.config import get_settings, get_driver, get_llm
"""
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_LLM_MODEL = "baidu/ernie-5.5-flash-preview:free"


@dataclass(frozen=True)
class Settings:
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    llm_api_key: str | None
    llm_base_url: str
    cypher_model: str
    answer_model: str
    embedding_model: str


def _require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Variável {name} ausente. Copie .env.example para .env e preencha.")
    return value


@lru_cache
def get_settings() -> Settings:
    load_dotenv(ROOT / ".env")  # não sobrescreve variáveis já definidas no ambiente
    return Settings(
        neo4j_uri=_require("NEO4J_URI"),
        neo4j_user=_require("NEO4J_USER"),
        neo4j_password=_require("NEO4J_PASSWORD"),
        llm_api_key=os.getenv("OPENROUTER_KEY"),
        # LLM_BASE_URL permite trocar de provedor (ex.: Ollama em http://localhost:11434/v1)
        llm_base_url=os.getenv("LLM_BASE_URL", OPENROUTER_BASE_URL),
        cypher_model=os.getenv("CYPHER_MODEL", DEFAULT_LLM_MODEL),
        answer_model=os.getenv("ANSWER_MODEL", DEFAULT_LLM_MODEL),
        embedding_model=os.getenv("EMBEDDING_MODEL", "baai/bge-m3"),
    )


@lru_cache
def get_driver():
    """Driver Neo4j compartilhado (Aura ou local, conforme o .env)."""
    from neo4j import GraphDatabase

    s = get_settings()
    return GraphDatabase.driver(s.neo4j_uri, auth=(s.neo4j_user, s.neo4j_password))


@lru_cache
def get_llm():
    """Cliente compatível com a API da OpenAI (OpenRouter por padrão)."""
    from openai import OpenAI

    s = get_settings()
    if not s.llm_api_key:
        raise RuntimeError("OPENROUTER_KEY ausente no .env.")
    return OpenAI(base_url=s.llm_base_url, api_key=s.llm_api_key)
