"""Interface do GraphRAG Jurídico.

Rodar a partir da raiz do repositório:
    streamlit run app/streamlit_app.py

Modo demonstração (sem Neo4j nem LLM): ative na barra lateral ou rode com GRAPHRAG_MOCK=1.
"""
import os
import re

import streamlit as st
import streamlit.components.v1 as components

from graphrag_juridico.ask import ask, is_no_info
from graphrag_juridico.config import get_settings
from graphrag_juridico.contracts import AskResult
from graphrag_juridico.graph_view import (MAP_SIZE, answer_view, cited_paths, knowledge_map, load_map,
                                          reasoning_view)
from graphrag_juridico.mock import DEMO_QUESTIONS as EXAMPLES
from graphrag_juridico.mock import is_recorded
from graphrag_juridico.multihop import EXPAND_QUERY, SHORTEST_PATH_QUERY, Options
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
    return ask(question, mock=mock, options=options, use_cypher=use_cypher, k=k).model_dump()


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
    k = st.slider("Trechos no contexto", 4, 12, 8,
                  help="Quantos trechos de artigos ([C1]...[Cn]) vão para o LLM. Os candidatos vêm dos caminhos do "
                       "grafo, das entidades da pergunta, da busca vetorial e da fulltext, fundidos por RRF, com no "
                       "máximo 2 por artigo. Menos trechos: resposta mais focada e rápida, mas pode faltar "
                       "informação. Mais trechos: mais cobertura, porém mais ruído e um prompt maior. "
                       "Não afeta as respostas gravadas do modo demonstração.")
    st.caption("Cada pergunta usa 2 a 3 requisições ao LLM. Perguntas repetidas vêm do cache.")

tab_ask, tab_map = st.tabs(["💬 Perguntar", "🌌 Mapa do conhecimento"])

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
        try:
            with st.spinner("Buscando no grafo e gerando a resposta... (a 1ª pergunta carrega o modelo de embedding)"):
                data = cached_ask(to_ask, use_sp, use_pr, use_comm, use_cypher, k, mock)
            st.session_state.last = {"result": data, "settings": describe(mock, options, use_cypher, k), "mock": mock}
        except Exception as error:  # banco fora do ar, cota do LLM esgotada etc.
            st.error(f"Não foi possível responder: {type(error).__name__}: {error}")

    last = st.session_state.get("last")
    if last:
        result = AskResult.model_validate(last["result"])
        st.caption(f"Resposta para: **{result.question}** · gerada com: {last['settings']}")
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
        with t_nodes:
            st.dataframe([n.model_dump() for n in result.nodes], width="stretch", hide_index=True)
        with t_json:
            st.json(result.model_dump(), expanded=False)

# ---------------------------------------------------------------- mapa
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
