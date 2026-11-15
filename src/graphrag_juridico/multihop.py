"""Recuperação multi-hop (KAG): parte das entidades da pergunta e navega no grafo.

- Shortest Path entre cada par de entidades (até 4 saltos), passando por relações entre entidades
  e por trechos que mencionam as duas (`(:Norm)<-[:MENTIONS]-(:Chunk)-[:MENTIONS]->(:Concept)`);
- expansão de 1 a 2 saltos pelas relações entre entidades, priorizando vizinhos da mesma comunidade
  (Louvain) da entidade de partida e, depois, os de maior PageRank;
- super-nós (entidades citadas por muitos trechos, como a CF/88) não servem de ponte, para não ligar tudo a tudo.

Os trechos dos caminhos, os das entidades e os da busca híbrida são fundidos por RRF e viram o contexto.
Cada algoritmo pode ser desligado (`Options`), para comparar o efeito de cada um na resposta.

Uso:
    python -m graphrag_juridico.multihop "Qual a relação entre a LGPD e a dignidade da pessoa humana?"
"""
import argparse
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import combinations
from typing import Any

from .algorithms import ENTITY_RELATIONS
from .contracts import ChunkRef, Confidence, GraphEdge, GraphNode
from .db import read
from .entities import link_entities
from .retrieval import diversify, embed_question, entity_chunk_ids, fetch_chunks, hybrid_rankings, rrf

SUPERNODE = 300          # entidades com mais menções que isso não servem de ponte
MAX_HOPS = 4
EXPAND_PER_ENTITY = 8
WEIGHTS = {"vector": 0.35, "fulltext": 0.15, "path": 0.3, "citations": 0.2}

NODE = ("{id: elementId(n), key: coalesce(n.id, n.name), label: labels(n)[0], "
        "name: coalesce(n.name, n.title, n.id), pagerank: n.pagerank, community: n.community}")
REL = "{source: elementId(startNode(r)), target: elementId(endNode(r)), type: type(r)}"
PATH_RETURN = f"RETURN [n IN nodes(p) | {NODE}] AS nodes, [r IN relationships(p) | {REL}] AS rels"

SUPERNODES_QUERY = """
MATCH (e)<-[m:MENTIONS]-(:Chunk) WHERE e:Norm OR e:Concept OR e:Institution
WITH e, count(m) AS mentions WHERE mentions > $limit
RETURN elementId(e) AS id
"""
SHORTEST_PATH_QUERY = f"""
MATCH (a) WHERE elementId(a) = $source
MATCH (b) WHERE elementId(b) = $target
MATCH p = shortestPath((a)-[:{'|'.join(ENTITY_RELATIONS + ['MENTIONS'])}*..{MAX_HOPS}]-(b))
WHERE none(n IN nodes(p) WHERE elementId(n) IN $blocked)
{PATH_RETURN}
"""
EXPAND_QUERY = f"""
UNWIND $ids AS sid
CALL (sid) {{
  MATCH (e) WHERE elementId(e) = sid
  MATCH p = (e)-[:{'|'.join(ENTITY_RELATIONS)}*1..2]-(o)
  WHERE none(n IN tail(nodes(p)) WHERE elementId(n) IN $blocked)
  WITH p, o,
       CASE WHEN $use_communities AND o.community = e.community THEN 1 ELSE 0 END AS same_community,
       CASE WHEN $use_pagerank THEN coalesce(o.pagerank, 0.0) ELSE 0.0 END AS centrality
  ORDER BY same_community DESC, centrality DESC, length(p)
  LIMIT $per_entity
  RETURN p
}}
{PATH_RETURN}
"""


@dataclass(frozen=True)
class Options:
    """Algoritmos de grafo usados na recuperação (todos ligados por padrão)."""
    use_shortest_path: bool = True
    use_pagerank: bool = True
    use_communities: bool = True


@dataclass
class Subgraph:
    """Nós, relações e caminhos acumulados, sem repetição."""
    nodes: dict[str, dict[str, Any]] = field(default_factory=dict)
    edges: dict[tuple[str, str, str], dict[str, str]] = field(default_factory=dict)
    paths: list[list[str]] = field(default_factory=list)

    def add(self, nodes: list[dict[str, Any]], rels: list[dict[str, str]]) -> None:
        for n in nodes:
            self.nodes.setdefault(n["id"], n)
        for r in rels:
            self.edges.setdefault((r["source"], r["target"], r["type"]), r)
        text = path_to_text(nodes, rels)
        if text not in self.paths:
            self.paths.append(text)

    def chunk_keys(self) -> list[str]:
        """Ids (`Chunk.id`) dos trechos que aparecem nos caminhos."""
        return [n["key"] for n in self.nodes.values() if n["label"] == "Chunk"]


@dataclass
class Context:
    """Tudo o que a recuperação encontrou para uma pergunta."""
    question: str
    entities: list[dict[str, Any]]
    chunks: list[ChunkRef]
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    paths: list[list[str]]
    cypher: list[str]
    confidence: Confidence
    options: Options


def path_to_text(nodes: list[dict[str, Any]], rels: list[dict[str, str]]) -> list[str]:
    """Caminho legível com a direção de cada relação:
    ["Lei 13.709/2018", "-[REGULATES]->", "privacidade", "<-[MENTIONS]-", "trecho 12:3"]."""
    def name(n):
        return f'trecho {n["key"]}' if n["label"] == "Chunk" else n["name"]

    text = [name(nodes[0])]
    for prev, rel, nxt in zip(nodes, rels, nodes[1:]):
        text.append(f'-[{rel["type"]}]->' if rel["source"] == prev["id"] else f'<-[{rel["type"]}]-')
        text.append(name(nxt))
    return text


def confidence(components: dict[str, float | None]) -> Confidence:
    """Média ponderada (WEIGHTS) dos componentes disponíveis; `None` fica de fora."""
    present = {k: min(max(v, 0.0), 1.0) for k, v in components.items() if v is not None and k in WEIGHTS}
    total = sum(WEIGHTS[k] for k in present)
    score = sum(WEIGHTS[k] * v for k, v in present.items()) / total if total else 0.0
    return Confidence(score=round(score, 4), components={k: round(v, 4) for k, v in present.items()})


def path_coverage(n_entities: int, connected_pairs: int, expanded: bool) -> float:
    """Fração dos pares de entidades ligados por caminho; com uma entidade só, 1 se houve expansão."""
    pairs = n_entities * (n_entities - 1) // 2
    if pairs:
        return connected_pairs / pairs
    return 1.0 if n_entities and expanded else 0.0


@lru_cache(maxsize=1)
def supernodes(limit: int = SUPERNODE) -> frozenset[str]:
    return frozenset(r["id"] for r in read(SUPERNODES_QUERY, limit=limit))


def shortest_path(source: str, target: str, blocked: frozenset[str]) -> dict[str, list] | None:
    rows = read(SHORTEST_PATH_QUERY, source=source, target=target, blocked=list(blocked - {source, target}))
    return rows[0] if rows else None


def expand(ids: list[str], blocked: frozenset[str], options: Options = Options(),
           per_entity: int = EXPAND_PER_ENTITY) -> list[dict[str, list]]:
    if not ids:
        return []
    return read(EXPAND_QUERY, ids=ids, blocked=list(blocked - set(ids)), per_entity=per_entity,
                use_pagerank=options.use_pagerank, use_communities=options.use_communities)


def retrieve(question: str, k: int = 8, per_article: int = 2, options: Options = Options()) -> Context:
    """Entidades -> Shortest Path + expansão 2 saltos -> trechos (caminhos + entidades + busca híbrida)."""
    vec = embed_question(question)
    entities = link_entities(question, vec)
    ids = [e["id"] for e in entities]
    blocked = supernodes()

    sub, connected = Subgraph(), 0
    if options.use_shortest_path:
        for source, target in combinations(ids, 2):
            row = shortest_path(source, target, blocked)
            if row:
                sub.add(row["nodes"], row["rels"])
                connected += 1
    expansions = expand(ids, blocked, options)
    for row in expansions:
        sub.add(row["nodes"], row["rels"])

    vector_ids, fulltext_ids = hybrid_rankings(question, vec)
    ranked = rrf(sub.chunk_keys(), entity_chunk_ids(ids, vec), vector_ids, fulltext_ids)
    rows = diversify(fetch_chunks(ranked[: k * 5], vec), k, per_article)

    chunks = [ChunkRef(id=r["id"], text=r["text"], article_title=r["title"], link=r["link"], score=r["vector_score"])
              for r in rows]
    top_scores = sorted((c.score or 0.0 for c in chunks), reverse=True)[:3]
    components = {
        "vector": sum(top_scores) / len(top_scores) if top_scores else 0.0,
        "fulltext": sum(c.id in fulltext_ids for c in chunks) / len(chunks) if chunks else 0.0,
        # sem Shortest Path não há como medir a ligação entre as entidades: o componente fica de fora
        "path": path_coverage(len(ids), connected, bool(expansions)) if options.use_shortest_path else None,
    }
    queries = ([SHORTEST_PATH_QUERY] if options.use_shortest_path else []) + [EXPAND_QUERY]
    return Context(
        question=question,
        entities=entities,
        chunks=chunks,
        nodes=[GraphNode(**{f: n[f] for f in ("id", "label", "name", "pagerank", "community")})
               for n in sub.nodes.values()],
        edges=[GraphEdge(**e) for e in sub.edges.values()],
        paths=sub.paths,
        cypher=[q.strip() for q in queries],
        confidence=confidence(components),
        options=options,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Recuperação multi-hop para uma pergunta (sem LLM).")
    parser.add_argument("question")
    parser.add_argument("--k", type=int, default=8, help="número de trechos no contexto")
    parser.add_argument("--no-shortest-path", action="store_true", help="não liga as entidades por Shortest Path")
    parser.add_argument("--no-pagerank", action="store_true", help="não prioriza vizinhos por PageRank")
    parser.add_argument("--no-communities", action="store_true", help="não prioriza vizinhos da mesma comunidade")
    args = parser.parse_args()

    options = Options(use_shortest_path=not args.no_shortest_path, use_pagerank=not args.no_pagerank,
                      use_communities=not args.no_communities)
    ctx = retrieve(args.question, k=args.k, options=options)
    print("Entidades:", "; ".join(f'{e["name"]} ({e["label"]}, {e["via"]})' for e in ctx.entities) or "nenhuma")
    print(f"\nCaminhos ({len(ctx.paths)}):")
    for path in ctx.paths:
        print("  " + " ".join(path))
    print(f"\nTrechos ({len(ctx.chunks)}):")
    for i, c in enumerate(ctx.chunks, 1):
        print(f"  [C{i}] {c.score:.3f}  {c.article_title}")
    print(f"\nSubgrafo: {len(ctx.nodes)} nós, {len(ctx.edges)} relações")
    print(f"Confiança: {ctx.confidence.score:.2f} {ctx.confidence.components}")


if __name__ == "__main__":
    main()
