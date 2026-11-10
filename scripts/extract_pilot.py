"""Piloto de extração NER/RE com LLM (OpenRouter).
Uso:  python scripts/extract_pilot.py [--limit-articles 5] [--max-chunks 4] [--workers 4]
Lê o piloto_ambito_juridico.parquet e grava data/extractions/<modelo>.jsonl (uma linha por chunk).
Variáveis (no .env): OPENROUTER_KEY, EXTRACTION_MODEL, EXTRACTION_REASONING (0/1).
Respostas do LLM ficam em cache (data/cache/), então reexecutar não paga de novo.
"""
import argparse, hashlib, json, os, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import ValidationError

from chunking import chunk_article
from extraction_prompt import SYSTEM, user_message
from extraction_schema import Extraction, RELATION_RULES

ROOT = Path(__file__).resolve().parent.parent
MAX_ATTEMPTS = 2  # 1 retentativa se o JSON vier inválido/truncado


def parse_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").removeprefix("json").strip()
    return json.loads(raw)


def validate(data: dict, chunk_text: str) -> dict:
    """Valida contra o schema e descarta o que não tem evidência literal ou endpoints coerentes.
    Devolve {'extraction': Extraction, 'dropped': {motivo: n}}."""
    ext = Extraction.model_validate(data)
    dropped = {"entity_sem_evidencia": 0, "relacao_sem_evidencia": 0, "relacao_endpoint": 0, "relacao_tipo": 0, "relacao_desconhecida": 0}

    def in_text(ev: str) -> bool:  # tolera diferença de espaços
        return " ".join(ev.split()).lower() in " ".join(chunk_text.split()).lower()

    entities = []
    for e in ext.entities:
        if in_text(e.evidence):
            entities.append(e)
        else:
            dropped["entity_sem_evidencia"] += 1
    by_name = {e.name: e for e in entities}

    relations = []
    for r in ext.relations:
        s, t = by_name.get(r.source), by_name.get(r.target)
        if r.type not in RELATION_RULES:
            dropped["relacao_desconhecida"] += 1
        elif not in_text(r.evidence):
            dropped["relacao_sem_evidencia"] += 1
        elif not (s and t):
            dropped["relacao_endpoint"] += 1
        elif (s.type, t.type) != RELATION_RULES[r.type]:
            dropped["relacao_tipo"] += 1
        else:
            relations.append(r)
    return {"extraction": Extraction(entities=entities, relations=relations), "dropped": dropped}


def call_llm(client, model, reasoning, title, category, chunk_text, cache_dir, fresh=False):
    key = hashlib.sha1(f"{model}|{reasoning}|{SYSTEM}|{chunk_text}".encode()).hexdigest()
    cache = cache_dir / f"{key}.json"
    if cache.exists() and not fresh:
        return json.loads(cache.read_text())
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": user_message(title, category, chunk_text)}],
        temperature=0,
        max_tokens=2500,
        response_format={"type": "json_object"},
        extra_body={"reasoning": {"enabled": reasoning}, "usage": {"include": True}},
    )
    out = {"content": resp.choices[0].message.content, "usage": resp.usage.model_dump() if resp.usage else {}}
    cache.write_text(json.dumps(out, ensure_ascii=False))
    return out


def process(job, client, model, reasoning, cache_dir, fresh=False):
    ch, title, category = job
    rec = {"chunk_id": ch["id"], "article_id": ch["id"].split(":")[0]}
    try:
        for attempt in range(MAX_ATTEMPTS):
            out = call_llm(client, model, reasoning, title, category, ch["text"], cache_dir, fresh or attempt > 0)
            try:
                v = validate(parse_json(out["content"]), ch["text"])
                break
            except (json.JSONDecodeError, ValidationError):
                if attempt == MAX_ATTEMPTS - 1:
                    raise
        rec.update(extraction=v["extraction"].model_dump(), dropped=v["dropped"], usage=out["usage"])
    except (json.JSONDecodeError, ValidationError) as e:
        rec.update(error=f"{type(e).__name__}: {str(e)[:200]}")
    except Exception as e:  # rede, rate limit, etc.
        rec.update(error=f"api: {type(e).__name__}: {str(e)[:200]}")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(ROOT / "piloto_ambito_juridico.parquet"))
    ap.add_argument("--limit-articles", type=int, default=None)
    ap.add_argument("--max-chunks", type=int, default=4, help="chunks por artigo (controla custo)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--fresh", action="store_true", help="ignora o cache (refaz as chamadas e regrava; necessário para medir custo)")
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    key, model = os.environ.get("OPENROUTER_KEY"), os.environ.get("EXTRACTION_MODEL")
    if not key or not model:
        sys.exit("Defina OPENROUTER_KEY e EXTRACTION_MODEL no .env")
    reasoning = os.environ.get("EXTRACTION_REASONING", "0") == "1"
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)

    df = pd.read_parquet(args.input)
    if args.limit_articles:
        df = df.head(args.limit_articles)
    jobs = []
    for r in df.itertuples():
        for ch in chunk_article(r.id, r.texto)[: args.max_chunks]:
            jobs.append((ch, r.titulo, r.categoria))
    print(f"{len(df)} artigos, {len(jobs)} chunks | modelo={model} reasoning={reasoning}")

    cache_dir = ROOT / "data" / "cache"; cache_dir.mkdir(parents=True, exist_ok=True)
    out_dir = ROOT / "data" / "extractions"; out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (Path(args.input).stem.removesuffix("_ambito_juridico") + "__" + model.replace("/", "_") + (".reasoning" if reasoning else "") + ".jsonl")

    with ThreadPoolExecutor(args.workers) as ex, open(out_path, "w", encoding="utf-8") as f:
        results = []
        for i, rec in enumerate(ex.map(lambda j: process(j, client, model, reasoning, cache_dir, args.fresh), jobs), 1):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); results.append(rec)
            if i % 20 == 0: print(f"  {i}/{len(jobs)}")

    ok = [r for r in results if "extraction" in r]
    ents = sum(len(r["extraction"]["entities"]) for r in ok)
    rels = sum(len(r["extraction"]["relations"]) for r in ok)
    tok_in = sum(r["usage"].get("prompt_tokens", 0) for r in ok)
    tok_out = sum(r["usage"].get("completion_tokens", 0) for r in ok)
    costs = [r["usage"]["cost"] for r in ok if r["usage"].get("cost") is not None]
    reasoning_tok = sum((r["usage"].get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0 for r in ok)
    print(f"\nOK: {len(ok)}/{len(results)} chunks | erros: {len(results) - len(ok)}")
    print(f"Entidades: {ents} | Relações: {rels} | tokens in/out: {tok_in}/{tok_out}")
    drop = {}
    for r in ok:
        for k, v in r["dropped"].items(): drop[k] = drop.get(k, 0) + v
    print("Descartado pela validação:", drop)
    print(f"Tokens de raciocínio (dentro da saída): {reasoning_tok}")
    if costs:
        total = sum(costs)
        print(f"Custo real (usage.cost): US$ {total:.4f} em {len(costs)} chunks | US$ {total/len(costs):.5f}/chunk")
        print(f"Projeção para 13.389 chunks: US$ {total/len(costs)*13389:.2f}")
    if len(costs) < len(ok):
        print(f"Atenção: {len(ok) - len(costs)} chunks sem custo (vieram do cache antigo); use --fresh para medir.")
    print(f"Saída: {out_path}")


if __name__ == "__main__":
    main()
