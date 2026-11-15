"""Algoritmos de grafo sobre o subgrafo de entidades (Norm, Concept, Institution).

PageRank mede a centralidade de cada entidade; Louvain agrupa entidades em comunidades
temáticas. Ambos rodam em através do networkx, sem depender do plugin GDS (ausente
no Neo4j Aura Free), e os resultados são gravados nos nós como `pagerank` e `community`.

Uso:
    python -m graphrag_juridico.algorithms            # calcula e mostra o resumo (não grava)
    python -m graphrag_juridico.algorithms --write    # calcula e grava no banco configurado no .env
"""
import argparse
from collections import Counter

import networkx as nx

from .db import read, write

ENTITY_RELATIONS = ["REGULATES", "AMENDS", "REVOKES", "ISSUED_BY", "INTERPRETS", "SPECIALIZES"]
BATCH = 1000
SEED = 42

EDGES_QUERY = f"""
MATCH (a)-[r:{'|'.join(ENTITY_RELATIONS)}]->(b)
RETURN elementId(a) AS source, elementId(b) AS target, coalesce(r.times, 1) AS weight
"""
NAMES_QUERY = """
MATCH (n) WHERE elementId(n) IN $ids
RETURN elementId(n) AS id, labels(n)[0] AS label, coalesce(n.name, n.id) AS name
"""
CLEAR_QUERY = "MATCH (n) WHERE n.pagerank IS NOT NULL OR n.community IS NOT NULL REMOVE n.pagerank, n.community"
SET_QUERY = """
UNWIND $rows AS row
MATCH (n) WHERE elementId(n) = row.id
SET n.pagerank = row.pagerank, n.community = row.community
"""


def build_graph(edges: list[dict]) -> nx.DiGraph:
    """Grafo dirigido; relações repetidas entre o mesmo par somam o peso."""
    graph = nx.DiGraph()
    for e in edges:
        if graph.has_edge(e["source"], e["target"]):
            graph[e["source"]][e["target"]]["weight"] += e["weight"]
        else:
            graph.add_edge(e["source"], e["target"], weight=e["weight"])
    return graph


def compute(graph: nx.DiGraph, seed: int = SEED) -> dict[str, dict]:
    """Devolve {node: {"pagerank": float, "community": int}}.
    Comunidades são numeradas por tamanho: 0 é a maior."""
    pagerank = nx.pagerank(graph, weight="weight")
    communities = nx.community.louvain_communities(graph.to_undirected(), weight="weight", seed=seed)
    community_of = {
        node: cid
        for cid, members in enumerate(sorted(communities, key=len, reverse=True))
        for node in members
    }
    return {node: {"pagerank": pagerank[node], "community": community_of[node]} for node in graph}


def save(scores: dict[str, dict]) -> None:
    """Apaga valores antigos e grava `pagerank`/`community` nos nós (ESCRITA no banco)."""
    write(CLEAR_QUERY)
    rows = [{"id": node, **values} for node, values in scores.items()]
    for i in range(0, len(rows), BATCH):
        write(SET_QUERY, rows=rows[i:i + BATCH])


def summary(scores: dict[str, dict], top: int = 10) -> str:
    """Resumo legível: entidades mais centrais e maiores comunidades."""
    ranked = sorted(scores, key=lambda n: scores[n]["pagerank"], reverse=True)
    sizes = Counter(v["community"] for v in scores.values())
    leaders = {}
    for node in ranked:  # entidade mais central de cada comunidade, para nomeá-la
        leaders.setdefault(scores[node]["community"], []).append(node)
    shown = set(ranked[:top]) | {m for cid, _ in sizes.most_common(top) for m in leaders[cid][:3]}
    names = {r["id"]: f'{r["name"]} ({r["label"]})' for r in read(NAMES_QUERY, ids=list(shown))}

    lines = [f"{len(scores)} entidades, {len(sizes)} comunidades", "", f"Top {top} PageRank:"]
    lines += [f'  {scores[n]["pagerank"]:.4f}  {names[n]}  [comunidade {scores[n]["community"]}]' for n in ranked[:top]]
    lines += ["", f"Top {top} comunidades (tamanho: entidades mais centrais):"]
    lines += [f"  {cid:>3} ({size}): " + "; ".join(names[m] for m in leaders[cid][:3])
              for cid, size in sizes.most_common(top)]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="PageRank e Louvain sobre o grafo de entidades.")
    parser.add_argument("--write", action="store_true", help="grava pagerank/community nos nós")
    args = parser.parse_args()

    graph = build_graph(read(EDGES_QUERY))
    scores = compute(graph)
    print(summary(scores))
    if args.write:
        save(scores)
        print(f"\nGravado em {len(scores)} nós.")
    else:
        print("\nNada foi gravado (use --write para gravar).")


if __name__ == "__main__":
    main()
