"""Visualização interativa do grafo (HTML autocontido com vis-network).

Visual inspirado no Graphify (github.com/Graphify-Labs/graphify): fundo escuro, layout ForceAtlas2,
nós coloridos pela comunidade (Louvain) e dimensionados pelo PageRank, arestas finas e curvas,
barra lateral com busca, legenda de comunidades e detalhes do nó clicado.

Visões:
- `knowledge_map()`: as entidades mais centrais do grafo inteiro (o "mapa do conhecimento");
- `answer_view()`: o subgrafo usado numa resposta, com as entidades da pergunta e os caminhos citados em destaque;
- `reasoning_view()`: a rota da resposta, da esquerda para a direita: pergunta -> entidades -> nós percorridos
  nos caminhos citados [P] -> resposta, com os ramos dos trechos citados [C] e da consulta gerada [Q].
"""
import html
import json
import math
import re
from collections import Counter
from typing import Any

from .algorithms import ENTITY_RELATIONS
from .contracts import AskResult
from .db import read
from .multihop import EXPAND_QUERY, SHORTEST_PATH_QUERY

VIS_NETWORK = "https://unpkg.com/vis-network@9.1.6/standalone/umd/vis-network.min.js"
PALETTE = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
           "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC"]
CHUNK_COLOR = "#8b8fa3"
HIGHLIGHT = "#FFD166"
# mesmas cores dos selos de citação na resposta
CHUNK_BRANCH, PATH_BRANCH, QUERY_BRANCH = "#4E79A7", "#F28E2B", "#59A14F"
FORCE_LAYOUT = {
    "physics": {"solver": "forceAtlas2Based",
                "forceAtlas2Based": {"gravitationalConstant": -60, "centralGravity": 0.006, "springLength": 110,
                                     "springConstant": 0.08, "damping": 0.45, "avoidOverlap": 0.6},
                "stabilization": {"iterations": 180, "fit": True}},
    "edges": {"smooth": {"type": "continuous", "roundness": 0.2}, "selectionWidth": 2.5},
}
ROUTE_LAYOUT = {
    "layout": {"hierarchical": {"direction": "LR", "levelSeparation": 320, "nodeSpacing": 140, "treeSpacing": 220,
                                "sortMethod": "directed", "shakeTowards": "roots"}},
    "physics": {"enabled": False},
    "edges": {"smooth": {"type": "cubicBezier", "forceDirection": "horizontal", "roundness": 0.5},
              "selectionWidth": 2.5,
              "font": {"color": "#e0e0e0", "size": 10, "strokeWidth": 3, "strokeColor": "#0f0f1a", "align": "top"}},
}
LABEL_SHARE = 0.04          # fração dos nós (os maiores) que mostram o rótulo sempre
MAP_SIZE = 1500
LABEL_PT = {"Norm": "Norma", "Concept": "Conceito", "Institution": "Instituição", "Chunk": "Trecho",
            "Scholar": "Doutrinador", "Article": "Artigo"}

MAP_NODES_QUERY = """
MATCH (n) WHERE n.pagerank IS NOT NULL
RETURN elementId(n) AS id, labels(n)[0] AS label, coalesce(n.name, n.id) AS name,
       n.pagerank AS pagerank, n.community AS community,
       COUNT { (n)<-[:MENTIONS]-() } AS mentions
ORDER BY n.pagerank DESC LIMIT $limit
"""
MAP_EDGES_QUERY = f"""
MATCH (a)-[r:{'|'.join(ENTITY_RELATIONS)}]->(b)
WHERE elementId(a) IN $ids AND elementId(b) IN $ids
RETURN elementId(a) AS source, elementId(b) AS target, type(r) AS type
"""


def color_of(community: int | None) -> str:
    return CHUNK_COLOR if community is None else PALETTE[community % len(PALETTE)]


def size_of(pagerank: float | None, max_pagerank: float, low: float = 8, high: float = 42) -> float:
    """Raiz do PageRank relativo: a distribuição é muito concentrada e o tamanho linear esconderia quase tudo."""
    if not pagerank or not max_pagerank:
        return low
    return round(low + (high - low) * math.sqrt(pagerank / max_pagerank), 1)


def name_communities(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Legenda: cada comunidade recebe o nome da sua entidade mais central; ordenada por tamanho."""
    counts = Counter(n["community"] for n in nodes if n.get("community") is not None)
    leader: dict[int, dict[str, Any]] = {}
    for n in nodes:
        cid = n.get("community")
        if cid is not None and (cid not in leader or (n.get("pagerank") or 0) > (leader[cid].get("pagerank") or 0)):
            leader[cid] = n
    return [{"id": cid, "name": leader[cid]["name"], "color": color_of(cid), "count": count}
            for cid, count in counts.most_common()]


def vis_nodes(nodes: list[dict[str, Any]], seeds: set[str] = frozenset()) -> list[dict[str, Any]]:
    """Nós no formato do vis-network; rótulo fixo só para os maiores, o resto aparece no tooltip."""
    max_pr = max((n.get("pagerank") or 0 for n in nodes), default=0)
    ranked = sorted((n.get("pagerank") or 0 for n in nodes), reverse=True)
    cutoff = ranked[max(0, math.ceil(len(ranked) * LABEL_SHARE) - 1)] if ranked else 0
    out = []
    for n in nodes:
        is_chunk = n["label"] == "Chunk"
        color = color_of(None if is_chunk else n.get("community"))
        is_seed = n["id"] in seeds
        pr = n.get("pagerank") or 0
        node = {
            "id": n["id"], "name": n["name"], "kind": LABEL_PT.get(n["label"], n["label"]),
            "community": n.get("community"), "pagerank": pr,
            "label": n["name"] if (is_seed or (pr and pr >= cutoff and len(nodes) > 30) or len(nodes) <= 30) else "",
            "shape": "diamond" if is_chunk else "dot",
            "size": 7 if is_chunk else size_of(pr, max_pr),
            "color": {"background": color, "border": "#ffffff" if is_seed else color,
                      "highlight": {"background": "#ffffff", "border": color}},
            "borderWidth": 3 if is_seed else 1.5,
        }
        if is_seed:
            node["shadow"] = {"enabled": True, "color": color, "size": 28, "x": 0, "y": 0}
        out.append(node)
    return out


def vis_edges(edges: list[dict[str, Any]], highlighted: set[tuple[str, str]] = frozenset()) -> list[dict[str, Any]]:
    out = []
    for i, e in enumerate(edges):
        hot = (e["source"], e["target"]) in highlighted or (e["target"], e["source"]) in highlighted
        out.append({
            "id": i, "from": e["source"], "to": e["target"], "title": e["type"], "type": e["type"],
            "arrows": {"to": {"enabled": True, "scaleFactor": 0.4}},
            "width": 3 if hot else 0.8,
            "color": {"color": HIGHLIGHT if hot else "#9aa0b4", "opacity": 1.0 if hot else 0.35},
        })
    return out


def path_names(path: list[str]) -> list[str]:
    """Nomes dos nós de um caminho legível (posições pares; as ímpares são as relações)."""
    return path[0::2]


def cited_paths(answer: str) -> set[int]:
    return {int(n) for n in re.findall(r"\[P(\d+)\]", answer)}


def _short(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "…"


def generated_cypher(result: AskResult) -> str | None:
    """A Cypher gerada pelo LLM (as outras são consultas estruturais da recuperação)."""
    structural = {SHORTEST_PATH_QUERY.strip(), EXPAND_QUERY.strip()}
    return next((q for q in result.cypher if q not in structural), None)


def reasoning_view(result: AskResult, max_paths: int = 8) -> str:
    """Rota da resposta (esquerda -> direita), só com o que a resposta citou:
    Pergunta -> entidades -> nós dos caminhos [P] -> Resposta; Pergunta -> trechos [C] -> Resposta;
    Pergunta -> consulta [Q] -> Resposta. Sem caminhos citados, mostra os primeiros caminhos."""
    answer = result.answer
    graph = {(f'trecho {n.name}' if n.label == "Chunk" else n.name): n for n in result.nodes}
    max_pr = max((n.pagerank or 0 for n in result.nodes), default=0)
    cited = sorted(i for i in cited_paths(answer) if 1 <= i <= len(result.paths))
    chosen = cited or list(range(1, min(len(result.paths), max_paths) + 1))[:max_paths]

    nodes: dict[str, dict[str, Any]] = {}
    edges: dict[tuple[str, str, str], dict[str, Any]] = {}

    def add_node(nid: str, level: int, **attrs: Any) -> None:
        if nid in nodes:
            nodes[nid]["level"] = min(nodes[nid]["level"], level)
        else:
            nodes[nid] = {"id": nid, "level": level, "community": None, "pagerank": 0, **attrs}

    def add_edge(source: str, target: str, label: str, color: str, dashed: bool = False) -> None:
        key = (source, target, label)
        if key not in edges:
            edges[key] = {"id": len(edges), "from": source, "to": target, "label": label, "type": label,
                          "arrows": {"to": {"enabled": True, "scaleFactor": 0.5}}, "width": 2,
                          "dashes": dashed, "color": {"color": color, "opacity": 0.9}}

    def box(color: str) -> dict[str, Any]:
        # sem o contorno escuro dos outros rótulos: dentro da caixa ele "engorda" o texto escuro até borrar
        return {"shape": "box", "margin": 12, "borderWidth": 0,
                "font": {"color": "#0f0f1a", "size": 15, "strokeWidth": 0, "face": "Segoe UI, sans-serif"},
                "color": {"background": color, "border": color, "highlight": {"background": "#ffffff", "border": color}}}

    def graph_node(name: str) -> tuple[str, dict[str, Any]]:
        n = graph.get(name)
        nid = n.id if n else f"name:{name}"
        is_chunk = name.startswith("trecho ")
        color = color_of(None if is_chunk or not n else n.community)
        return nid, {"name": name, "label": name, "kind": LABEL_PT.get(n.label, n.label) if n else "Entidade",
                     "community": None if is_chunk or not n else n.community, "pagerank": (n.pagerank or 0) if n else 0,
                     "shape": "diamond" if is_chunk else "dot",
                     "size": 9 if is_chunk else size_of(n.pagerank if n else None, max_pr, 12, 26),
                     "color": {"background": color, "border": color, "highlight": {"background": "#ffffff", "border": color}}}

    add_node("question", 0, name="Pergunta", label="Pergunta", kind="Pergunta", detail=result.question,
             **box("#e0e0e0"))
    last_level = 1
    for number in chosen:
        names = path_names(result.paths[number - 1])
        rels = result.paths[number - 1][1::2]
        ids = []
        for pos, name in enumerate(names):
            nid, attrs = graph_node(name)
            add_node(nid, 1 + pos, **attrs)
            ids.append(nid)
            last_level = max(last_level, 1 + pos)
        add_edge("question", ids[0], "entidade da pergunta", "#e0e0e0", dashed=True)
        for (a, b), rel in zip(zip(ids, ids[1:]), rels):
            label = rel.strip("<->[]") or rel
            source, target = (b, a) if rel.startswith("<-") else (a, b)
            add_edge(source, target, label, PATH_BRANCH)
        add_edge(ids[-1], "answer", f"P{number}" + ("" if cited else " (não citado)"), PATH_BRANCH, dashed=not cited)

    for number in sorted({int(n) for n in re.findall(r"\[C(\d+)\]", answer)}):
        if not 1 <= number <= len(result.chunks):
            continue
        chunk = result.chunks[number - 1]
        nid, attrs = graph_node(f"trecho {chunk.id}")
        if nid in nodes:   # o trecho já está num caminho: só ganha o rótulo [C]
            nodes[nid]["label"] = f"C{number}"
        else:
            add_node(nid, last_level, **{**attrs, "label": f"C{number}", "kind": "Trecho",
                                         "color": {"background": CHUNK_BRANCH, "border": CHUNK_BRANCH,
                                                   "highlight": {"background": "#ffffff", "border": CHUNK_BRANCH}}})
            add_edge("question", nid, "busca híbrida", CHUNK_BRANCH, dashed=True)
        nodes[nid]["name"] = f"[C{number}] {chunk.article_title}"
        nodes[nid]["detail"] = _short(chunk.text, 420)
        add_edge(nid, "answer", f"C{number}", CHUNK_BRANCH)

    query = generated_cypher(result)
    if query and "[Q]" in answer:
        add_node("query", 1, name="Consulta gerada [Q]", label="Q · Cypher", kind="Text-to-Cypher",
                 detail=query, shape="hexagon", size=16,
                 color={"background": QUERY_BRANCH, "border": QUERY_BRANCH,
                        "highlight": {"background": "#ffffff", "border": QUERY_BRANCH}})
        add_edge("question", "query", "Text-to-Cypher", QUERY_BRANCH, dashed=True)
        add_edge("query", "answer", "Q", QUERY_BRANCH)

    add_node("answer", max(n["level"] for n in nodes.values()) + 1, name="Resposta", label="Resposta",
             kind="Resposta", detail=_short(answer, 600), **box(HIGHLIGHT))
    entity_nodes = [n for n in nodes.values() if n.get("community") is not None]
    return render(list(nodes.values()), list(edges.values()), name_communities(entity_nodes),
                  title="Rota da resposta", height=640, options=ROUTE_LAYOUT)


def answer_view(result: AskResult) -> str:
    """HTML do subgrafo de uma resposta: entidades da pergunta brilhando, caminhos citados em amarelo."""
    nodes = [n.model_dump() for n in result.nodes]
    by_name: dict[str, str] = {}
    for n in nodes:
        by_name[f'trecho {n["name"]}' if n["label"] == "Chunk" else n["name"]] = n["id"]
    seeds = {by_name[p[0]] for p in result.paths if p and p[0] in by_name}

    hot: set[tuple[str, str]] = set()
    for number in cited_paths(result.answer):
        if 1 <= number <= len(result.paths):
            ids = [by_name.get(name) for name in path_names(result.paths[number - 1])]
            hot |= {(a, b) for a, b in zip(ids, ids[1:]) if a and b}

    return render(vis_nodes(nodes, seeds), vis_edges([e.model_dump() for e in result.edges], hot),
                  name_communities(nodes), title="Subgrafo da resposta", height=620)


def load_map(limit: int = MAP_SIZE) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Entidades mais centrais (PageRank) e as relações entre elas."""
    nodes = read(MAP_NODES_QUERY, limit=limit)
    edges = read(MAP_EDGES_QUERY, ids=[n["id"] for n in nodes])
    return nodes, edges


def knowledge_map(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> str:
    return render(vis_nodes(nodes), vis_edges(edges), name_communities(nodes),
                  title="Mapa do conhecimento", height=760)


def render(nodes: list[dict], edges: list[dict], communities: list[dict], title: str, height: int,
           options: dict | None = None) -> str:
    """Página HTML autocontida; os dados entram como JSON escapado para não fechar o <script>."""
    data = json.dumps({"nodes": nodes, "edges": edges, "communities": communities}, ensure_ascii=False)
    data = data.replace("</", "<\\/")
    layout = json.dumps(options or FORCE_LAYOUT)
    return (TEMPLATE.replace("__TITLE__", html.escape(title)).replace("__HEIGHT__", str(height))
            .replace("__VIS__", VIS_NETWORK).replace("__OPTIONS__", layout).replace("__DATA__", data))


TEMPLATE = """<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8">
<script src="__VIS__"></script>
<style>
  :root { --bg:#0f0f1a; --panel:#1a1a2e; --border:#2a2a4e; --text:#e0e0e0; --muted:#8a8aa3; --accent:#4E79A7; }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { background:var(--bg); color:var(--text); font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
         display:flex; height:__HEIGHT__px; overflow:hidden; border-radius:10px; }
  #graph { flex:1; position:relative; }
  #hud { position:absolute; left:14px; top:12px; font-size:12px; color:var(--muted); pointer-events:none; letter-spacing:.04em; }
  #hud b { color:var(--text); font-size:14px; letter-spacing:.08em; text-transform:uppercase; }
  #center { position:absolute; left:14px; bottom:12px; z-index:5; background:var(--panel); color:var(--text);
            border:1px solid var(--border); border-radius:6px; padding:5px 10px; font-size:12px; cursor:pointer; }
  #center:hover { border-color:var(--accent); }
  #side { width:280px; background:var(--panel); border-left:1px solid var(--border); display:flex; flex-direction:column; }
  .sec { padding:12px 14px; border-bottom:1px solid var(--border); }
  h3 { font-size:11px; letter-spacing:.12em; color:var(--muted); margin-bottom:8px; }
  #search { width:100%; background:var(--bg); border:1px solid var(--border); color:var(--text); padding:7px 10px;
            border-radius:6px; font-size:13px; outline:none; }
  #search:focus { border-color:var(--accent); }
  #results div, .nb { padding:4px 6px; border-radius:4px; cursor:pointer; font-size:12px; }
  #results div:hover, .nb:hover, .lg:hover { background:var(--border); }
  #info { font-size:12px; line-height:1.6; }
  #info .name { font-size:15px; font-weight:600; margin-bottom:4px; }
  #info .detail { margin-top:6px; padding:6px 8px; background:var(--bg); border-radius:6px; max-height:180px;
                  overflow-y:auto; white-space:pre-wrap; font-size:11.5px; color:#cfcfe0; }
  #info .tag { display:inline-block; padding:1px 7px; border-radius:10px; font-size:11px; margin-right:4px; color:#0f0f1a; }
  #legend { flex:1; overflow-y:auto; padding:8px 14px; }
  .lg { display:flex; align-items:center; gap:8px; padding:4px; border-radius:4px; cursor:pointer; font-size:12px; }
  .lg.off { opacity:.3; }
  .dot { width:10px; height:10px; border-radius:50%; flex-shrink:0; }
  .lg .n { margin-left:auto; color:var(--muted); }
  .lg .t { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  #all { font-size:11px; color:var(--accent); cursor:pointer; float:right; }
  div.vis-tooltip { background:var(--panel)!important; color:var(--text)!important; border:1px solid var(--border)!important;
                    border-radius:6px!important; font-family:inherit!important; font-size:12px!important; }
</style></head>
<body>
<div id="graph"><div id="hud"><b>__TITLE__</b><br><span id="stats"></span></div>
  <button id="center" title="Enquadrar o grafo inteiro">⌖ Centralizar</button></div>
<div id="side">
  <div class="sec"><input id="search" placeholder="Buscar entidade..." autocomplete="off"><div id="results"></div></div>
  <div class="sec" id="info"><span style="color:var(--muted)">Clique num nó para ver os detalhes.</span></div>
  <div class="sec" style="border-bottom:none;padding-bottom:0"><h3>COMUNIDADES <span id="all">mostrar todas</span></h3></div>
  <div id="legend"></div>
</div>
<script>
const DATA = __DATA__;
const nodes = new vis.DataSet(DATA.nodes.map(n => ({...n, title: `${n.name}\\n${n.kind}` +
  (n.community !== null && n.community !== undefined ? ` · comunidade ${n.community}` : "") +
  (n.pagerank ? `\\nPageRank ${n.pagerank.toFixed(5)}` : "")})));
const edges = new vis.DataSet(DATA.edges);
const byId = Object.fromEntries(DATA.nodes.map(n => [n.id, n]));
const commOf = Object.fromEntries(DATA.communities.map(c => [c.id, c]));
document.getElementById("stats").textContent = `${DATA.nodes.length} nós · ${DATA.edges.length} relações · ${DATA.communities.length} comunidades`;

const OPTIONS = __OPTIONS__;
const network = new vis.Network(document.getElementById("graph"), {nodes, edges}, {
  interaction: { hover: true, tooltipDelay: 90, hideEdgesOnDrag: true, keyboard: false },
  nodes: { font: { color: "#ffffff", size: 13, strokeWidth: 3, strokeColor: "#0f0f1a" } },
  ...OPTIONS,
});
network.once("stabilizationIterationsDone", () => network.setOptions({ physics: { enabled: false } }));

// Enquadramento: o Streamlit monta abas escondidas com tamanho zero, e o primeiro "fit" sai deslocado.
// Reenquadra quando o quadro muda de tamanho (ex.: ao abrir a aba), enquanto o usuário não mexeu no zoom.
let userMoved = false;
const fitAll = () => network.fit({ animation: { duration: 300, easingFunction: "easeInOutQuad" } });
network.on("dragStart", () => { userMoved = true; });
network.on("zoom", () => { userMoved = true; });
network.once("afterDrawing", () => network.fit());
new ResizeObserver(() => { network.redraw(); if (!userMoved) network.fit(); })
  .observe(document.getElementById("graph"));
document.getElementById("center").onclick = () => { userMoved = false; fitAll(); };

function focus(id) {
  userMoved = true;
  network.selectNodes([id]);
  network.focus(id, { scale: 1.2, animation: { duration: 600, easingFunction: "easeInOutQuad" } });
  showInfo(id);
}
function showInfo(id) {
  const n = byId[id]; if (!n) return;
  const nbIds = network.getConnectedNodes(id);
  const rels = network.getConnectedEdges(id).map(e => edges.get(e)).filter(Boolean);
  const comm = commOf[n.community];
  const color = (n.color && n.color.background) || "#888";
  let h = `<div class="name">${esc(n.name)}</div><span class="tag" style="background:${color}">${n.kind}</span>`;
  if (comm) h += `<span class="tag" style="background:${comm.color}">comunidade: ${esc(comm.name)}</span>`;
  if (n.pagerank) h += `<div>PageRank: <b>${n.pagerank.toFixed(5)}</b></div>`;
  if (n.detail) h += `<div class="detail">${esc(n.detail)}</div>`;
  h += `<div style="margin-top:6px;color:var(--muted)">${nbIds.length} vizinhos</div>`;
  h += nbIds.slice(0, 12).map(v => {
    const r = rels.find(e => e.from === v || e.to === v);
    const arrow = r ? (r.from === id ? `→ ${r.type}` : `← ${r.type}`) : "";
    return `<div class="nb" data-id="${esc(v)}"><span style="color:var(--muted)">${arrow}</span> ${esc(byId[v] ? byId[v].name : v)}</div>`;
  }).join("");
  document.getElementById("info").innerHTML = h;
  document.querySelectorAll("#info .nb").forEach(el => el.onclick = () => focus(el.dataset.id));
}
function esc(s) { return String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c])); }
network.on("click", p => { if (p.nodes.length) showInfo(p.nodes[0]); });

const search = document.getElementById("search"), results = document.getElementById("results");
search.oninput = () => {
  const q = search.value.trim().toLowerCase();
  if (q.length < 2) { results.innerHTML = ""; return; }
  results.innerHTML = DATA.nodes.filter(n => n.name.toLowerCase().includes(q)).slice(0, 8)
    .map(n => `<div data-id="${esc(n.id)}">${esc(n.name)} <span style="color:var(--muted)">${n.kind}</span></div>`).join("");
  results.querySelectorAll("div").forEach(el => el.onclick = () => { focus(el.dataset.id); results.innerHTML = ""; search.value = ""; });
};

const hidden = new Set();
function applyFilter() {
  nodes.update(DATA.nodes.map(n => ({ id: n.id, hidden: n.community !== null && hidden.has(n.community) })));
  document.querySelectorAll(".lg").forEach(el => el.classList.toggle("off", hidden.has(+el.dataset.c)));
}
document.getElementById("legend").innerHTML = DATA.communities.map(c =>
  `<div class="lg" data-c="${c.id}"><div class="dot" style="background:${c.color}"></div>` +
  `<span class="t">${esc(c.name)}</span><span class="n">${c.count}</span></div>`).join("");
document.querySelectorAll(".lg").forEach(el => el.onclick = () => {
  const c = +el.dataset.c; hidden.has(c) ? hidden.delete(c) : hidden.add(c); applyFilter();
});
document.getElementById("all").onclick = () => { hidden.clear(); applyFilter(); };
</script>
</body></html>
"""
