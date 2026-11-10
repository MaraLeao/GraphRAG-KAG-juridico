"""Carga estrutural (sem LLM): Article, Category, Chunk, IN_CATEGORY, HAS_CHUNK.
Uso: python scripts/load_structure.py [arquivo.parquet]   (padrão: subconjunto_ambito_juridico.parquet)
"""
import os, sys
import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase

from chunking import chunk_article

BATCH = 200


load_dotenv()
path = sys.argv[1] if len(sys.argv) > 1 else "subconjunto_ambito_juridico.parquet"
df = pd.read_parquet(path)

articles = []
for r in df.itertuples():
    articles.append({
        "id": r.id, "title": r.titulo, "summary": r.resumo, "link": r.link,
        "word_count": int(r.n_palavras), "category": r.categoria,
        "chunks": chunk_article(r.id, r.texto),
    })

QUERY = """
UNWIND $rows AS row
MERGE (a:Article {id: row.id})
  SET a.title = row.title, a.summary = row.summary, a.link = row.link, a.word_count = row.word_count
MERGE (c:Category {name: row.category})
MERGE (a)-[:IN_CATEGORY]->(c)
WITH a, row
UNWIND row.chunks AS ch
MERGE (k:Chunk {id: ch.id})
  SET k.text = ch.text, k.order = ch.order, k.section = ch.section
MERGE (a)-[:HAS_CHUNK]->(k)
"""

driver = GraphDatabase.driver(os.environ["NEO4J_URI"], auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]))
with driver:
    driver.verify_connectivity()
    for i in range(0, len(articles), BATCH):
        driver.execute_query(QUERY, rows=articles[i:i + BATCH])
        print(f"{min(i + BATCH, len(articles))}/{len(articles)} artigos")
    for label in ["Article", "Category", "Chunk"]:
        n = driver.execute_query(f"MATCH (n:{label}) RETURN count(n) AS n").records[0]["n"]
        print(f"{label}: {n}")
