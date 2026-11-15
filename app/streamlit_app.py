"""Interface do GraphRAG Jurídico.

Rodar a partir da raiz do repositório:
    streamlit run app/streamlit_app.py

Modo demonstração (sem Neo4j nem LLM): ative na barra lateral ou rode com GRAPHRAG_MOCK=1.
"""
import os
import re
import time

import altair as alt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from graphrag_juridico.ask import ask, is_no_info
from graphrag_juridico.config import get_settings
from graphrag_juridico.contracts import AskResult
from graphrag_juridico.evaluation import METHOD_COLORS, METHODS, SUMMARY_LABELS, find_question, run_method, summary
from graphrag_juridico.evaluation import score as score_runs
from graphrag_juridico.graph_view import (MAP_SIZE, answer_view, cited_paths, knowledge_map, load_map,
                                          reasoning_view)
from graphrag_juridico.mock import DEMO_QUESTIONS as EXAMPLES
from graphrag_juridico.metrics import METRIC_HELP, evaluate, glossary_entries, norm_label
from graphrag_juridico.mock import is_recorded, recorded_comparison
from graphrag_juridico.multihop import EXPAND_QUERY, SHORTEST_PATH_QUERY, Options

COST = {"llm": "1 requisição", "vector": "1 requisição", "graphrag": "2 a 3 requisições"}
COMPONENTS = {"vector": "Similaridade vetorial", "fulltext": "Concordância fulltext",
              "path": "Cobertura de caminhos", "citations": "Citações válidas"}
STRUCTURAL = {SHORTEST_PATH_QUERY.strip(), EXPAND_QUERY.strip()}

st.set_page_config(page_title="GraphRAG Jurídico", page_icon="⚖️", layout="wide")
st.markdown("""<style>
  .block-container { padding-top: 2rem; }
  .stTabs [data-baseweb="tab"] { font-size: 1.05rem; }
  iframe { border-radius: 10px; }
</style>""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False, max_entries=64)
def cached_ask(question: str, shortest_path: bool, pagerank: bool, communities: bool,
               use_cypher: bool, k: int, mock: bool) -> dict:
    """Mesma pergunta com as mesmas opções não gasta outra requisição ao LLM."""
    options = Options(use_shortest_path=shortest_path, use_pagerank=pagerank, use_communities=communities)
    start = time.perf_counter()
    result = ask(question, mock=mock, options=options, use_cypher=use_cypher, k=k).model_dump()
    return {"result": result, "latency": round(time.perf_counter() - start, 2)}


@st.cache_data(show_spinner=False, max_entries=64)
def cached_baseline(method: str, question: str, mock: bool) -> dict | None:
    """Resposta de um método de comparação (no modo demonstração, só a gravada pela avaliação).
    Cada método é guardado em cache separadamente: se um falhar, o outro não é perdido nem refeito.
    Erros não ficam no cache, então a próxima tentativa roda de novo."""
    if mock:
        return (recorded_comparison(question) or {}).get(method)
    return run_method(method, question)


def explain(error: Exception) -> str:
    """Motivo de uma falha em português, para o aviso na tela."""
    status = getattr(error, "status_code", None)
    if status in (402, 429):
        return "cota do LLM esgotada (nas chaves configuradas); tente novamente mais tarde ou amanhã"
    if "resposta vazia" in str(error):
        return "o provedor do LLM devolveu uma resposta vazia; tente novamente"
    return f"{type(error).__name__}: {str(error)[:200]}"


def bar_chart(frame: pd.DataFrame, height: int) -> None:
    """Barras horizontais agrupadas por método, estáticas (sem zoom), escala fixa de 0 a 100%."""
    data = frame.reset_index(names="Métrica").melt("Métrica", var_name="Método", value_name="Valor")
    names = list(frame.columns)
    colors = [METHOD_COLORS[m] for m in METHODS if METHODS[m][0] in names]
    chart = alt.Chart(data).mark_bar(cornerRadiusEnd=3).encode(
        y=alt.Y("Métrica:N", sort=list(frame.index), title=None, axis=alt.Axis(labelLimit=260)),
        yOffset=alt.YOffset("Método:N", sort=names),
        x=alt.X("Valor:Q", scale=alt.Scale(domain=[0, 1]), title=None, axis=alt.Axis(format="%")),
        color=alt.Color("Método:N", sort=names, scale=alt.Scale(domain=names, range=colors),
                        legend=None),
        tooltip=["Método", "Métrica", alt.Tooltip("Valor:Q", format=".0%")],
    ).properties(height=height)
    # legenda fora do gráfico: dentro dele ela fica presa à largura das barras e corta os nomes
    st.markdown(" &nbsp; ".join(f'<span style="color:{c}">■</span> {n}' for n, c in zip(names, colors)),
                unsafe_allow_html=True)
    st.altair_chart(chart, width="stretch")


def short_norms(norms: list[str], limit: int = 3) -> str:
    """" · sem suporte: CF, CDC, CPP +4" (a lista completa fica na comparação e no relatório)."""
    if not norms:
        return ""
    shown = ", ".join(norm_label(n) for n in norms[:limit])
    return f" · sem suporte: {shown}" + (f" +{len(norms) - limit}" if len(norms) > limit else "")


def legend(labels) -> None:
    """Dois dropdowns fechados: siglas e leis que aparecem na tela, e como ler as métricas."""
    entries = glossary_entries(labels)
    if entries:
        with st.expander(f"📖 Siglas e leis citadas ({len(entries)})"):
            for col, part in zip(st.columns(2, gap="medium"), (entries[::2], entries[1::2])):
                with col:
                    for label, description in part:
                        st.markdown(f":orange-background[**{label}**] &nbsp; {description}")
    with st.expander("ℹ️ Como ler as métricas"):
        for name, description in METRIC_HELP.items():
            st.markdown(f"**{name}:** {description}")


@st.cache_data(show_spinner=False, ttl=3600)
def cached_map(limit: int) -> str:
    return knowledge_map(*load_map(limit))


def database_name() -> str:
    """Qual Neo4j o .env aponta (sem mostrar endereço nem credenciais)."""
    try:
        uri = get_settings().neo4j_uri
    except RuntimeError:
        return "não configurado (.env)"
    return "Neo4j Aura" if "databases.neo4j.io" in uri else "Neo4j local" if "localhost" in uri else "Neo4j"


def describe(mock: bool, options: Options, use_cypher: bool, k: int) -> str:
    if mock:
        return "modo demonstração"
    algos = [name for on, name in [(options.use_shortest_path, "Shortest Path"), (options.use_pagerank, "PageRank"),
                                   (options.use_communities, "Louvain")] if on] or ["nenhum algoritmo"]
    return f"{', '.join(algos)} · Cypher do LLM: {'sim' if use_cypher else 'não'} · {k} trechos"


def pct(value: float) -> str:
    return "-" if value != value else f"{value:.0%}"


def render_comparison(runs: dict[str, dict], question: str) -> None:
    """Três colunas (LLM puro, RAG vetorial, GraphRAG) com as métricas automáticas e um quadro comparativo."""
    reference = find_question(question)
    st.markdown("#### ⚖️ Comparação dos métodos")
    st.caption("Mesmo LLM nos três. O LLM puro responde sem contexto; o RAG vetorial e o GraphRAG usam o mesmo "
               "prompt e as mesmas regras de citação, mudando só o contexto (trechos da busca vetorial x grafo)."
               + (" Esta pergunta tem gabarito no conjunto de avaliação." if reference else ""))
    methods = [m for m in METHODS if m in runs]
    evaluated = {m: evaluate(AskResult.model_validate(runs[m]["result"]), reference and reference.reference,
                             runs[m].get("context_norms")) for m in methods}
    for col, m in zip(st.columns(len(methods), gap="medium"), methods):
        result, metric = AskResult.model_validate(runs[m]["result"]), evaluated[m]
        with col:
            st.markdown(f"**{METHODS[m][0]}**")
            with st.container(height=300):
                st.markdown(badges(result.answer))
            if metric.correct is not None:
                st.markdown("✅ **correta** (gabarito)" if metric.correct else "❌ **incorreta** (gabarito)")
            if metric.abstained:
                st.markdown("🛑 se absteve")
            st.markdown(f"Citações válidas: **{pct(metric.citations_valid)}** · "
                        f"frases com fonte: **{pct(metric.grounded_sentences)}**")
            if metric.unsupported_norms:
                st.markdown("⚠️ normas sem suporte no contexto: **"
                            + ", ".join(norm_label(n) for n in metric.unsupported_norms) + "**")
            else:
                st.markdown("✓ nenhuma norma sem suporte")
            latency = runs[m].get("latency")
            st.caption(f"⏱ {latency:.1f} s · {COST[m]}" if latency else COST[m])

    table = {METHODS[m][0]: {"Citações válidas": evaluated[m].citations_valid,
                             "Frases com fonte": evaluated[m].grounded_sentences,
                             "Normas sem suporte (qtde.)": len(evaluated[m].unsupported_norms)} for m in methods}
    if reference and not reference.should_abstain:
        for m in methods:
            table[METHODS[m][0]]["Itens do gabarito encontrados"] = evaluated[m].term_recall
    frame = pd.DataFrame(table)
    chart, grid = st.columns([1.3, 1], gap="large")
    with chart:
        bar_chart(frame.drop(index="Normas sem suporte (qtde.)"), height=200)
    with grid:
        st.dataframe(frame.map(lambda v: f"{v:.0%}" if isinstance(v, float) else v), width="stretch")
        legend(norm_label(n) for m in methods for n in evaluated[m].unsupported_norms)
    st.divider()


def badges(answer: str) -> str:
    """Citações viram selos coloridos: trechos em azul, caminhos em laranja, consulta em verde."""
    answer = re.sub(r"\[C(\d+)\]", r" :blue-background[C\1]", answer)
    answer = re.sub(r"\[P(\d+)\]", r" :orange-background[P\1]", answer)
    return answer.replace("[Q]", " :green-background[Q]")


# ---------------------------------------------------------------- barra lateral
with st.sidebar:
    st.markdown("## ⚖️ GraphRAG Jurídico")
    st.caption("Perguntas sobre ~1.000 artigos do Âmbito Jurídico, respondidas com um grafo de conhecimento "
               "no Neo4j e um LLM que cita as fontes.")
    mock = st.toggle("Modo demonstração", value=os.getenv("GRAPHRAG_MOCK") == "1",
                     help="Não usa o banco nem o LLM. As perguntas de exemplo reproduzem execuções reais gravadas; "
                          "outras perguntas recebem uma resposta genérica fixa. Serve para testar sem gastar a cota.")
    st.caption(f"Banco: **{database_name()}**" + (" (não usado no modo demonstração)" if mock else ""))
    st.markdown("#### Algoritmos de grafo")
    use_sp = st.checkbox("Shortest Path entre entidades", value=True)
    use_pr = st.checkbox("PageRank na expansão", value=True)
    use_comm = st.checkbox("Comunidades (Louvain) na expansão", value=True)
    st.markdown("#### Geração")
    use_cypher = st.checkbox("Cypher gerada pelo LLM", value=True,
                             help="Desligue para economizar uma requisição por pergunta.")
    compare = st.toggle("Comparar com LLM puro e RAG vetorial", value=False,
                        help="Responde também sem RAG (LLM puro) e com RAG só vetorial, e compara as métricas "
                             "automáticas. Custa +2 requisições por pergunta (no modo demonstração, mostra as "
                             "comparações gravadas na avaliação).")
    k = st.slider("Trechos no contexto", 4, 12, 8,
                  help="Quantos trechos de artigos ([C1]...[Cn]) vão para o LLM. Os candidatos vêm dos caminhos do "
                       "grafo, das entidades da pergunta, da busca vetorial e da fulltext, fundidos por RRF, com no "
                       "máximo 2 por artigo. Menos trechos: resposta mais focada e rápida, mas pode faltar "
                       "informação. Mais trechos: mais cobertura, porém mais ruído e um prompt maior. "
                       "Não afeta as respostas gravadas do modo demonstração.")
    st.caption("Cada pergunta usa 2 a 3 requisições ao LLM (4 a 5 comparando). Perguntas repetidas vêm do cache.")

tab_ask, tab_map, tab_eval = st.tabs(["💬 Perguntar", "🌌 Mapa do conhecimento", "📊 Avaliação"])

# ---------------------------------------------------------------- perguntar
with tab_ask:
    st.markdown("### Pergunte ao grafo")
    # A pergunta só é enviada por Enter/botão ou por um exemplo; mudar as opções da barra lateral
    # não refaz a pergunta (evita gastar a cota do LLM sem querer).
    cols = st.columns(len(EXAMPLES))
    for col, example in zip(cols, EXAMPLES):
        if col.button(example, width="stretch"):
            st.session_state.question_input = example
            st.session_state.pending = example
    with st.form("ask_form", border=False):
        field, send = st.columns([6, 1], vertical_alignment="bottom")
        question = field.text_input("Pergunta", key="question_input", label_visibility="collapsed",
                                    placeholder="Digite a pergunta e tecle Enter")
        submitted = send.form_submit_button("Perguntar", type="primary", width="stretch", key="submit")

    to_ask = st.session_state.pop("pending", None) or (question.strip() if submitted else None)
    if to_ask:
        options = Options(use_shortest_path=use_sp, use_pagerank=use_pr, use_communities=use_comm)
        data = None
        try:
            with st.spinner("Buscando no grafo e gerando a resposta... (a 1ª pergunta carrega o modelo de embedding)"):
                data = cached_ask(to_ask, use_sp, use_pr, use_comm, use_cypher, k, mock)
        except Exception as error:  # banco fora do ar, cota do LLM esgotada etc.
            st.error(f"Não foi possível responder: {explain(error)}")
        if data is not None:
            comparison, failures = None, []
            if compare:
                # cada método de comparação falha sozinho: a resposta do GraphRAG (já paga) nunca se perde
                baselines = {}
                with st.spinner("Respondendo também com o LLM puro e o RAG vetorial..."):
                    for method in ("llm", "vector"):
                        try:
                            run = cached_baseline(method, to_ask, mock)
                        except Exception as error:
                            failures.append(f"{METHODS[method][0]}: {explain(error)}")
                            continue
                        if run:
                            baselines[method] = run
                if baselines:
                    graphrag = dict(data)
                    if mock:   # latência real da gravação, não a do cache
                        graphrag["latency"] = (recorded_comparison(to_ask) or {}).get("graphrag", {}).get("latency")
                    comparison = {**baselines, "graphrag": graphrag}
                elif mock:
                    failures.append("comparação não gravada para esta pergunta no modo demonstração: use uma "
                                    "pergunta da aba 📊 Avaliação ou desligue o modo demonstração")
            st.session_state.last = {"result": data["result"], "settings": describe(mock, options, use_cypher, k),
                                     "mock": mock, "comparison": comparison, "failures": failures}

    last = st.session_state.get("last")
    if last:
        result = AskResult.model_validate(last["result"])
        st.caption(f"Resposta para: **{result.question}** · gerada com: {last['settings']}")
        if last.get("failures"):
            st.warning("Comparação incompleta. A resposta do GraphRAG abaixo não foi afetada.\n\n"
                       + "\n".join(f"- {f}" for f in last["failures"]), icon="⚖️")
        if last.get("comparison"):
            render_comparison(last["comparison"], result.question)
            st.markdown("#### Detalhes do GraphRAG/KAG")
        left, right = st.columns([1, 1.25], gap="large")
        with left:
            if last["mock"] and is_recorded(result.question):
                st.info("Modo demonstração: execução real gravada anteriormente (banco + LLM), reproduzida sem "
                        "gastar a cota. Os textos dos trechos estão resumidos.", icon="🧪")
            elif last["mock"]:
                st.info("Modo demonstração: resposta genérica FIXA, porque esta pergunta não está entre os exemplos "
                        "gravados. Não veio do LLM nem do banco.", icon="🧪")
            if is_no_info(result.answer):
                st.warning("O sistema se absteve: o corpus não tem informação suficiente.", icon="🛑")
            st.markdown(badges(result.answer))
            st.divider()
            score = result.confidence.score
            st.metric("Nível de confiança", f"{score:.0%}")
            for key, value in result.confidence.components.items():
                st.progress(value, text=f"{COMPONENTS.get(key, key)}: {value:.0%}")
        with right:
            components.html(answer_view(result), height=630, scrolling=False)
            st.caption("Brilho: entidades da pergunta · amarelo: caminhos citados na resposta · "
                       "cor: comunidade (Louvain) · tamanho: PageRank · ◆: trechos")

        t_chunks, t_paths, t_cypher, t_nodes, t_json = st.tabs(
            [f"📄 Trechos ({len(result.chunks)})", f"🔗 Caminhos ({len(result.paths)})",
             f"🧮 Cypher ({len(result.cypher)})", f"🔵 Nós ({len(result.nodes)})", "{ } JSON"])
        with t_chunks:
            for i, c in enumerate(result.chunks, 1):
                with st.expander(f"[C{i}] {c.article_title}  ·  cosseno {c.score or 0:.2f}"):
                    st.write(c.text)
                    if c.link:
                        st.markdown(f"[Abrir o artigo ↗]({c.link})")
        with t_paths:
            view = st.segmented_control("Visualização", ["Rota da resposta", "Lista"], default="Rota da resposta",
                                        key="paths_view", label_visibility="collapsed")
            if view == "Rota da resposta":
                st.caption("O percurso da pergunta até a resposta, só com o que a resposta citou: entidades ligadas → "
                           "nós percorridos nos caminhos :orange-background[P] · trechos :blue-background[C] · "
                           "consulta :green-background[Q]. Clique num nó para ver o detalhe.")
                components.html(reasoning_view(result), height=650, scrolling=False)
            else:
                cited = cited_paths(result.answer)
                for i, p in enumerate(result.paths, 1):
                    tag = " · :orange-background[citado na resposta]" if i in cited else ""
                    st.markdown(f"**[P{i}]** " + " ".join(f"`{x}`" if j % 2 else x for j, x in enumerate(p)) + tag)
        with t_cypher:
            for q in result.cypher:
                st.caption("Gerada pelo LLM" if q not in STRUCTURAL else "Consulta estrutural da recuperação")
                st.code(q, language="cypher")
                if q not in STRUCTURAL and result.cypher_rows:
                    st.caption(f"Resultado [Q]: {len(result.cypher_rows)} linhas")
                    st.dataframe(result.cypher_rows, width="stretch", hide_index=True)
        with t_nodes:
            st.dataframe([n.model_dump() for n in result.nodes], width="stretch", hide_index=True)
        with t_json:
            st.json(result.model_dump(), expanded=False)

# ---------------------------------------------------------------- avaliação
with tab_eval:
    st.markdown("### Avaliação: LLM puro x RAG vetorial x GraphRAG/KAG")
    st.caption("Conjunto de perguntas com gabarito (estruturadas, multi-hop e armadilhas sem resposta no corpus). "
               "As respostas foram gravadas com `python -m graphrag_juridico.evaluation run`; as métricas são "
               "calculadas aqui, sem LLM. Clique numa pergunta da aba Perguntar com a comparação ligada para ver "
               "as três respostas lado a lado.")
    rows = score_runs()
    if not rows:
        st.info("Nenhuma resposta gravada ainda. Rode `python -m graphrag_juridico.evaluation run`.")
    else:
        agg = summary(rows)
        methods = [m for m in METHODS if m in agg]
        frame = pd.DataFrame({METHODS[m][0]: {label: agg[m][key] for key, label in SUMMARY_LABELS.items()}
                              for m in methods})
        chart, grid = st.columns([1, 1.1], gap="large")
        with chart:
            bar_chart(frame.drop(index=SUMMARY_LABELS["latency"]), height=330)
        with grid:
            shown = frame.copy().astype(object)
            for label in shown.index:
                shown.loc[label] = [f"{v:.1f} s" if label == SUMMARY_LABELS["latency"] else pct(v)
                                    for v in frame.loc[label]]
            st.dataframe(shown, width="stretch")
            st.caption("Menor é melhor em \"abstenção indevida\" e \"normas sem suporte\".")
        per_question = pd.DataFrame([{"Pergunta": f"{r['id']}: {r['question']}", "Tipo": r["type"],
                                      "Método": METHODS[r["method"]][0],
                                      "Resultado": ("✅" if r["correct"] else "❌") + (" · absteve" if r["abstained"] else "")
                                      + short_norms(r["unsupported_norms"])}
                                     for r in rows])
        order = list(dict.fromkeys(per_question["Pergunta"]))           # E1..A3, como no conjunto
        table = (per_question.pivot(index=["Pergunta", "Tipo"], columns="Método", values="Resultado")
                 .reindex(columns=[METHODS[m][0] for m in methods]).reset_index())
        table = table.set_index("Pergunta").loc[order].reset_index()
        st.dataframe(table, width="stretch", hide_index=True, column_config={
            "Pergunta": st.column_config.TextColumn(width="large"),
            "Tipo": st.column_config.TextColumn(width="small"),
            **{METHODS[m][0]: st.column_config.TextColumn(width="medium") for m in methods}})
        legend(norm_label(n) for r in rows for n in r["unsupported_norms"])

# ---------------------------------------------------------------- mapa
# Executado por último: na 1ª vez lê ~1.500 nós do banco, e as abas anteriores não precisam esperar.
with tab_map:
    st.markdown("### Mapa do conhecimento")
    st.caption("As entidades mais centrais do grafo (PageRank). As cores são as comunidades encontradas pelo "
               "Louvain, sem supervisão: repare como elas correspondem às áreas do Direito. "
               "Use a busca, clique num nó para ver os vizinhos e filtre comunidades na legenda.")
    if mock:
        st.info("O mapa lê o Neo4j: desligue o modo demonstração para vê-lo.")
    else:
        size = st.select_slider("Entidades no mapa", options=[300, 800, 1500, 2500], value=MAP_SIZE)
        try:
            with st.spinner("Carregando o grafo... (só na primeira vez; depois fica em cache)"):
                page = cached_map(size)
            components.html(page, height=770, scrolling=False)
        except Exception as error:
            st.error(f"Não foi possível carregar o mapa: {type(error).__name__}: {error}")
