"""GraphRAG híbrido: vetor + fulltext (RRF) -> expansão multi-hop no grafo -> resposta com citações.
Uso: python scripts/graphrag.py "pergunta" [--k 6] [--no-answer] [--supernode 300]
Pré-requisitos: embeddings (scripts/embed.py) e índices vetoriais (schema_vector.cypher).
Variáveis (.env): OPENROUTER_KEY, EMBEDDING_MODEL, ANSWER_MODEL (padrão = EXTRACTION_MODEL), NEO4J_*.
"""
import argparse, json, os, re
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
LUCENE_SPECIAL = re.compile(r'([+\-&|!(){}\[\]^"~*?:\\/])')

# entidades citadas pelos chunks + vizinhos a 1 salto por relações entre entidades (ignora super-nós)
EXPAND = """
UNWIND $chunk_ids AS cid
MATCH (c:Chunk {id: cid})-[:MENTIONS]->(e)
WHERE (e:Norm OR e:Institution OR e:Concept) AND COUNT { (e)<-[:MENTIONS]-() } < $supernode
WITH DISTINCT e
OPTIONAL MATCH (e)-[r:REGULATES|AMENDS|REVOKES|ISSUED_BY|INTERPRETS|SPECIALIZES]-(o)
WHERE COUNT { (o)<-[:MENTIONS]-() } < $supernode
RETURN labels(e)[0] AS tl, coalesce(e.name, e.id) AS tn, type(r) AS rel,
       startNode(r) = e AS outgoing, labels(o)[0] AS ol, coalesce(o.name, o.id) AS on_
LIMIT $limit
"""

SYSTEM = ("Você responde perguntas jurídicas usando APENAS o contexto fornecido (trechos de artigos e relações do grafo). "
          "Cite os trechos como [C1], [C2]... e as relações do grafo que usou. Se o contexto não contém a resposta, "
          "diga claramente que não há informação suficiente no corpus; não use conhecimento externo.")


def lucene(q):
    return " ".join(LUCENE_SPECIAL.sub(r"\\\1", w) for w in q.split())


def rrf(*rankings, k=60):
    score = {}
    for ranking in rankings:
        for pos, key in enumerate(ranking):
            score[key] = score.get(key, 0) + 1 / (k + pos + 1)
    return sorted(score, key=score.get, reverse=True)


def retrieve(driver, client, question, k=6, supernode=300, edge_limit=40, per_article=2, local_embed=True):
    model = os.getenv("EMBEDDING_MODEL", "baai/bge-m3")
    if local_embed:  # bge-m3 local: vetores compatíveis com os da API (cos ~0,998), sem custo nem envio da pergunta
        from embed import embed_local
        vec = embed_local([question])[0]
    else:
        vec = client.embeddings.create(model=model, input=[question]).data[0].embedding
    with driver.session() as s:
        vec_rows = s.run("CALL db.index.vector.queryNodes('vec_chunk', $k, $v) YIELD node, score "
                         "RETURN node.id AS id, score", k=k * 5, v=vec).data()
        ft_rows = s.run("CALL db.index.fulltext.queryNodes('ft_chunk', $q, {limit: $k}) YIELD node, score "
                        "RETURN node.id AS id, score", q=lucene(question), k=k * 5).data()
        vec_score = {r["id"]: r["score"] for r in vec_rows}
        ranked = rrf([r["id"] for r in vec_rows], [r["id"] for r in ft_rows])
        cands = s.run("MATCH (a:Article)-[:HAS_CHUNK]->(c:Chunk) WHERE c.id IN $ids "
                      "RETURN c.id AS id, c.text AS text, a.id AS aid, a.title AS title, a.link AS link", ids=ranked).data()
        cands.sort(key=lambda r: ranked.index(r["id"]))
        chunks, per = [], {}
        for c in cands:  # no máximo `per_article` chunks por artigo, para diversificar as fontes
            if per.get(c["aid"], 0) < per_article:
                per[c["aid"]] = per.get(c["aid"], 0) + 1
                chunks.append(c)
            if len(chunks) == k:
                break
        ids = [c["id"] for c in chunks]
        edges = s.run(EXPAND, chunk_ids=ids, supernode=supernode, limit=edge_limit).data()
    for c in chunks:
        c["vector_score"] = vec_score.get(c["id"])
    return chunks, [e for e in edges if e["rel"]]


def format_context(chunks, edges):
    parts = [f"[C{i}] ({c['title']})\n{c['text']}" for i, c in enumerate(chunks, 1)]
    graph = [f"({e['tn']}) -[{e['rel']}]-> ({e['on_']})" if e["outgoing"] else f"({e['on_']}) -[{e['rel']}]-> ({e['tn']})"
             for e in edges]
    return "TRECHOS:\n" + "\n\n".join(parts) + "\n\nRELAÇÕES DO GRAFO:\n" + ("\n".join(graph) or "(nenhuma)")


def answer(client, question, context):
    model = os.getenv("ANSWER_MODEL") or os.environ["EXTRACTION_MODEL"]
    r = client.chat.completions.create(
        model=model, temperature=0,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": f"{context}\n\nPERGUNTA: {question}"}])
    return r.choices[0].message.content


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--per-article", type=int, default=2, help="máximo de chunks por artigo")
    ap.add_argument("--api-embedding", action="store_true", help="embute a pergunta pela API em vez do modelo local")
    ap.add_argument("--supernode", type=int, default=300)
    ap.add_argument("--no-answer", action="store_true", help="só mostra a recuperação")
    a = ap.parse_args()
    load_dotenv(ROOT / ".env")
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_KEY"])
    driver = GraphDatabase.driver(os.environ["NEO4J_URI"], auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]))
    chunks, edges = retrieve(driver, client, a.question, a.k, a.supernode, per_article=a.per_article, local_embed=not a.api_embedding)
    context = format_context(chunks, edges)
    print(context[:6000] if a.no_answer else "")
    if not a.no_answer:
        print(answer(client, a.question, context))
        print("\nFontes:")
        for i, c in enumerate(chunks, 1):
            print(f"  [C{i}] {c['title']} - {c['link']}")


if __name__ == "__main__":
    main()
