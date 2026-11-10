"""Carga das entidades e relações extraídas no Neo4j (com resolução de entidades).
Pré-requisito: Article e Chunk já existem (scripts/load_structure.py <parquet do piloto/subconjunto>).
Uso: python scripts/load_entities.py <arquivo.jsonl> [--dry-run]
--dry-run só mostra as estatísticas da resolução (não lê o .env nem conecta ao Neo4j).
"""
import argparse, json, os
from collections import Counter, defaultdict
from pathlib import Path

from extraction_schema import RELATION_RULES
from resolve_entities import KEY_PROP, Resolver

ROOT = Path(__file__).resolve().parent.parent
BATCH = 500


def build(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    rows = [r for r in rows if "extraction" in r]
    res = Resolver()
    mentions = []                      # (chunk, article, ident, evidence)
    rels = {}                          # (type, s, t) -> {evidence, source_chunk, times}
    raw_counts = Counter()
    skipped = Counter()
    for r in rows:
        local = {}
        for e in r["extraction"]["entities"]:
            raw_counts[e["type"]] += 1
            ident = res.resolve(e)
            if ident is None:
                skipped[e["type"]] += 1
                continue
            local[e["name"]] = ident
            mentions.append((r["chunk_id"], r["article_id"], ident, e["evidence"]))
        for x in r["extraction"]["relations"]:
            s, t = local.get(x["source"]), local.get(x["target"])
            if not (s and t) or s == t:
                continue
            k = (x["type"], s, t)
            d = rels.setdefault(k, {"evidence": x["evidence"], "source_chunk": r["chunk_id"], "times": 0})
            d["times"] += 1
    return res, mentions, rels, raw_counts, skipped


def report(res, mentions, rels, raw_counts, skipped):
    by_label = Counter(l for (l, _k) in res.names)
    print("Entidades extraídas (menções) -> nós distintos após resolução:")
    for t in ["Norm", "Institution", "Concept", "Scholar"]:
        print(f"  {t:12} {raw_counts[t]:5} -> {by_label[t]:5}   (descartadas por ambiguidade: {skipped[t]})")
    print(f"Menções ligadas: {len(mentions)} | Relações entre entidades (únicas): {len(rels)}")
    print("Relações por tipo:", dict(Counter(k[0] for k in rels)))
    merged = sorted(((sum(c.values()), res.display(i), dict(c)) for i, c in res.names.items() if len(c) > 1), reverse=True)[:8]
    print("Maiores fusões (nome exibido <- variantes):")
    for n, disp, c in merged:
        print(f"  {disp}  <- {c}")
    multi = Counter()
    arts = defaultdict(set)
    for _c, a, ident, _e in mentions:
        arts[ident].add(a)
    shared = [(len(v), res.display(i), i[0]) for i, v in arts.items() if len(v) >= 2]
    print(f"Entidades presentes em 2+ artigos (potencial de multi-hop): {len(shared)}")
    print("Mais compartilhadas:", sorted(shared, reverse=True)[:8])


def load(res, mentions, rels):
    from dotenv import load_dotenv
    from neo4j import GraphDatabase
    load_dotenv(ROOT / ".env")
    driver = GraphDatabase.driver(os.environ["NEO4J_URI"], auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]))

    def run(q, rows):
        for i in range(0, len(rows), BATCH):
            driver.execute_query(q, rows=rows[i:i + BATCH])

    with driver:
        driver.verify_connectivity()
        for label, kp in KEY_PROP.items():
            rows = []
            for ident in [i for i in res.names if i[0] == label]:
                props = dict(res.props[ident]); props["name"] = res.display(ident)
                rows.append({"key": ident[1] if label != "Norm" else ident[1], "props": props})
            # Norm: chave = id; demais: chave = nome exibido (canônico)
            if label != "Norm":
                for r in rows: r["key"] = r["props"]["name"]
            run(f"UNWIND $rows AS row MERGE (n:{label} {{{kp}: row.key}}) SET n += row.props", rows)
        key_of = lambda ident: ident[1] if ident[0] == "Norm" else res.display(ident)

        for label, kp in KEY_PROP.items():
            rows = [{"chunk": c, "key": key_of(i), "evidence": ev} for (c, _a, i, ev) in mentions if i[0] == label]
            run(f"UNWIND $rows AS row MATCH (k:Chunk {{id: row.chunk}}) MATCH (e:{label} {{{kp}: row.key}}) "
                "MERGE (k)-[m:MENTIONS]->(e) SET m.evidence = row.evidence", rows)

        agg = {"Norm": ("CITES_NORM", "times"), "Concept": ("ABOUT", "weight"), "Scholar": ("CITES_SCHOLAR", "times")}
        for label, (rel, prop) in agg.items():
            cnt = Counter((a, i) for (_c, a, i, _e) in mentions if i[0] == label)
            rows = [{"art": a, "key": key_of(i), "n": n} for (a, i), n in cnt.items()]
            kp = KEY_PROP[label]
            run(f"UNWIND $rows AS row MATCH (a:Article {{id: row.art}}) MATCH (e:{label} {{{kp}: row.key}}) "
                f"MERGE (a)-[r:{rel}]->(e) SET r.{prop} = row.n", rows)

        for rtype, (sl, tl) in RELATION_RULES.items():
            rows = [{"s": key_of(s), "t": key_of(t), **d} for (ty, s, t), d in rels.items() if ty == rtype]
            run(f"UNWIND $rows AS row MATCH (s:{sl} {{{KEY_PROP[sl]}: row.s}}) MATCH (t:{tl} {{{KEY_PROP[tl]}: row.t}}) "
                f"MERGE (s)-[r:{rtype}]->(t) SET r.evidence = row.evidence, r.source_chunk = row.source_chunk, r.times = row.times", rows)
        n = driver.execute_query("MATCH (a:Article)-[:HAS_CHUNK]->(k:Chunk)-[:MENTIONS]->() RETURN count(DISTINCT a) AS n").records[0]["n"]
        print(f"Carga concluída. Artigos com entidades ligadas: {n}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    res, mentions, rels, raw, skipped = build(a.jsonl)
    report(res, mentions, rels, raw, skipped)
    if not a.dry_run:
        load(res, mentions, rels)
