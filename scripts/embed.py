"""Calcula embeddings para Chunk e Concept e grava no Neo4j (retomável: só processa nós sem embedding).
Uso: python scripts/embed.py [--label Chunk|Concept|all] [--limit N] [--batch 64] [--local]
Variáveis (.env): OPENROUTER_KEY, EMBEDDING_MODEL (padrão baai/bge-m3, 1024 dimensões), NEO4J_*.
Depois de embutir, aplique os índices vetoriais: python scripts/apply_schema.py schema_vector.cypher
"""
import argparse, os, time
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
DIM = 1024
# texto que representa cada nó; Concept usa nome + descrição quando houver
QUERIES = {
    "Chunk": ("MATCH (n:Chunk) WHERE n.embedding IS NULL RETURN n.id AS id, n.text AS text LIMIT $n", "id"),
    "Concept": ("MATCH (n:Concept) WHERE n.embedding IS NULL "
                "RETURN n.name AS id, n.name + coalesce('. ' + n.description, '') AS text LIMIT $n", "name"),
}


_local = None


def embed_local(texts):
    """bge-m3 local (CPU/GPU) via sentence-transformers; mesmo modelo da API, vetores compatíveis."""
    global _local
    if _local is None:
        from sentence_transformers import SentenceTransformer
        _local = SentenceTransformer("BAAI/bge-m3")
        _local.max_seq_length = 512
    return _local.encode(texts, batch_size=8, normalize_embeddings=True).tolist()


def embed(client, model, texts, tries=4):
    texts = [t[:8000] for t in texts]  # limite de segurança de caracteres
    if client is None:
        return embed_local(texts)
    for i in range(tries):
        try:
            r = client.embeddings.create(model=model, input=texts)
            return [d.embedding for d in sorted(r.data, key=lambda d: d.index)]
        except Exception as e:  # rate limit / falha transitória
            if i == tries - 1:
                raise
            time.sleep(2 ** i)


def run(driver, client, model, label, batch, limit):
    query, key = QUERIES[label]
    done = 0
    while limit is None or done < limit:
        n = batch if limit is None else min(batch, limit - done)
        with driver.session() as s:
            rows = s.run(query, n=n).data()
            if not rows:
                break
            vecs = embed(client, model, [r["text"] or r["id"] for r in rows])
            assert all(len(v) == DIM for v in vecs), f"dimensão != {DIM}"
            s.run(f"UNWIND $rows AS r MATCH (n:{label} {{{key}: r.id}}) "
                  "CALL db.create.setNodeVectorProperty(n, 'embedding', r.vec)",
                  rows=[{"id": r["id"], "vec": v} for r, v in zip(rows, vecs)]).consume()
        done += len(rows)
        print(f"{label}: {done} embutidos", flush=True)
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="all", choices=["Chunk", "Concept", "all"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--local", action="store_true", help="embeddings locais (sem API/chave)")
    a = ap.parse_args()
    load_dotenv(ROOT / ".env")
    model = os.getenv("EMBEDDING_MODEL", "baai/bge-m3")
    client = None if a.local else OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_KEY"])
    driver = GraphDatabase.driver(os.environ["NEO4J_URI"], auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]))
    for label in (["Chunk", "Concept"] if a.label == "all" else [a.label]):
        run(driver, client, model, label, a.batch, a.limit)


if __name__ == "__main__":
    main()
