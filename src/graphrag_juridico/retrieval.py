"""Busca híbrida de trechos (Chunk): similaridade de cosseno (índice vetorial) + fulltext + Cypher.

Três fontes de candidatos, fundidas por Reciprocal Rank Fusion (RRF):
- vetor: `vec_chunk` com o embedding da pergunta (bge-m3, o mesmo modelo usado na indexação);
- fulltext: `ft_chunk` com as palavras da pergunta;
- entidades: trechos que MENCIONAM as entidades ligadas à pergunta, ordenados por cosseno em Cypher.
"""
import re
from functools import lru_cache
from typing import Any

from .db import read

EMBEDDING_MODEL = "BAAI/bge-m3"
LUCENE_SPECIAL = re.compile(r'([+\-&|!(){}\[\]^"~*?:\\/])')

VECTOR_QUERY = """
CALL db.index.vector.queryNodes('vec_chunk', $n, $vec) YIELD node, score
RETURN node.id AS id, score
"""
FULLTEXT_QUERY = """
CALL db.index.fulltext.queryNodes('ft_chunk', $q, {limit: $n}) YIELD node, score
RETURN node.id AS id, score
"""
ENTITY_CHUNKS_QUERY = """
UNWIND $ids AS eid
CALL (eid) {
  MATCH (c:Chunk)-[:MENTIONS]->(e) WHERE elementId(e) = eid
  WITH c ORDER BY vector.similarity.cosine(c.embedding, $vec) DESC
  LIMIT $per_entity
  RETURN collect(c.id) AS chunk_ids
}
RETURN eid, chunk_ids
"""
FETCH_QUERY = """
MATCH (a:Article)-[:HAS_CHUNK]->(c:Chunk) WHERE c.id IN $ids
RETURN c.id AS id, c.text AS text, a.id AS article_id, a.title AS title, a.link AS link,
       vector.similarity.cosine(c.embedding, $vec) AS vector_score
"""


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL)
    model.max_seq_length = 512
    return model


@lru_cache(maxsize=256)
def _embed(text: str) -> tuple[float, ...]:
    return tuple(_model().encode([text], normalize_embeddings=True)[0].tolist())


def embed_question(text: str) -> list[float]:
    """Embedding local da pergunta (a pergunta não sai da máquina)."""
    return list(_embed(text))


def lucene(text: str) -> str:
    """Escapa os caracteres especiais da sintaxe Lucene usada pelos índices fulltext."""
    return " ".join(LUCENE_SPECIAL.sub(r"\\\1", w) for w in text.split())


def rrf(*rankings: list[str], k: int = 60) -> list[str]:
    """Reciprocal Rank Fusion: soma 1/(k + posição) de cada lista; listas vazias são ignoradas."""
    score: dict[str, float] = {}
    for ranking in rankings:
        for pos, key in enumerate(ranking):
            score[key] = score.get(key, 0.0) + 1 / (k + pos + 1)
    return sorted(score, key=score.get, reverse=True)


def diversify(rows: list[dict[str, Any]], k: int, per_article: int = 2) -> list[dict[str, Any]]:
    """Mantém a ordem e corta em `k` trechos, com no máximo `per_article` do mesmo artigo."""
    chosen, per = [], {}
    for row in rows:
        if per.get(row["article_id"], 0) < per_article:
            per[row["article_id"]] = per.get(row["article_id"], 0) + 1
            chosen.append(row)
        if len(chosen) == k:
            break
    return chosen


def hybrid_rankings(question: str, vec: list[float], n: int = 30) -> tuple[list[str], list[str]]:
    """Ids de trechos ordenados pela busca vetorial e pela fulltext."""
    vector_ids = [r["id"] for r in read(VECTOR_QUERY, n=n, vec=vec)]
    fulltext_ids = [r["id"] for r in read(FULLTEXT_QUERY, n=n, q=lucene(question))]
    return vector_ids, fulltext_ids


def interleave(rankings: list[list[str]]) -> list[str]:
    """Trechos citados por mais listas primeiro; depois alterna entre as listas (1º de cada, 2º de cada...),
    para que nenhuma entidade monopolize o contexto."""
    hits: dict[str, int] = {}
    for ranking in rankings:
        for cid in set(ranking):
            hits[cid] = hits.get(cid, 0) + 1
    shared = sorted((c for c in hits if hits[c] > 1), key=lambda c: -hits[c])
    alternated = [r[i] for i in range(max(map(len, rankings), default=0)) for r in rankings if i < len(r)]
    return list(dict.fromkeys(shared + alternated))


def entity_chunk_ids(entity_ids: list[str], vec: list[float], per_entity: int = 10) -> list[str]:
    """Trechos que mencionam as entidades: os de maior cosseno de cada uma, intercalados."""
    if not entity_ids:
        return []
    rows = {r["eid"]: r["chunk_ids"] for r in read(ENTITY_CHUNKS_QUERY, ids=entity_ids, vec=vec, per_entity=per_entity)}
    return interleave([rows.get(eid, []) for eid in entity_ids])


def fetch_chunks(ids: list[str], vec: list[float]) -> list[dict[str, Any]]:
    """Texto, artigo e cosseno de cada trecho, na ordem de `ids`."""
    if not ids:
        return []
    position = {cid: i for i, cid in enumerate(ids)}
    rows = read(FETCH_QUERY, ids=ids, vec=vec)
    return sorted(rows, key=lambda r: position[r["id"]])
