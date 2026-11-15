import json
import re

from graphrag_juridico.ask import ask
from graphrag_juridico.graph_view import (PALETTE, answer_view, cited_paths, color_of, knowledge_map, name_communities,
                                          path_names, reasoning_view, render, size_of, vis_edges, vis_nodes)
from graphrag_juridico.mock import DEMO_QUESTIONS, mock_ask


def node(nid, name, community, pagerank, label="Concept"):
    return {"id": nid, "name": name, "label": label, "community": community, "pagerank": pagerank}


def embedded_data(page: str) -> dict:
    return json.loads(re.search(r"const DATA = (\{.*?\});\n", page, re.DOTALL).group(1).replace("<\\/", "</"))


def test_colors_repeat_the_palette_by_community_and_chunks_are_grey():
    assert color_of(0) == PALETTE[0] and color_of(len(PALETTE)) == PALETTE[0]
    assert color_of(None) not in PALETTE


def test_size_grows_with_pagerank():
    assert size_of(0.01, 0.01) > size_of(0.0025, 0.01) > size_of(None, 0.01)
    assert size_of(0.01, 0.01) == 42


def test_communities_are_named_by_their_most_central_entity_and_sorted_by_size():
    nodes = [node("a", "direitos fundamentais", 0, 0.02), node("b", "liberdade", 0, 0.001),
             node("c", "dignidade", 0, 0.009), node("d", "responsabilidade civil", 1, 0.008)]

    legend = name_communities(nodes)

    assert [(c["id"], c["name"], c["count"]) for c in legend] == [(0, "direitos fundamentais", 3),
                                                                   (1, "responsabilidade civil", 1)]


def test_seed_nodes_glow_and_only_big_nodes_have_fixed_labels():
    nodes = [node(f"n{i}", f"conceito {i}", 0, 1 / (i + 1)) for i in range(100)]

    out = {n["id"]: n for n in vis_nodes(nodes, seeds={"n50"})}

    assert out["n0"]["label"] == "conceito 0" and out["n99"]["label"] == ""
    assert out["n50"]["label"] and out["n50"]["shadow"]["enabled"] and out["n50"]["borderWidth"] == 3


def test_highlighted_edges_ignore_direction():
    edges = [{"source": "a", "target": "b", "type": "REGULATES"}, {"source": "b", "target": "c", "type": "AMENDS"}]

    out = vis_edges(edges, highlighted={("b", "a")})

    assert out[0]["width"] > out[1]["width"]


def test_path_names_skip_relations():
    assert path_names(["LGPD", "-[REGULATES]->", "privacidade", "<-[MENTIONS]-", "trecho x:1"]) == [
        "LGPD", "privacidade", "trecho x:1"]


def test_render_escapes_closing_script_tags():
    page = render([{"id": "x", "name": "</script><script>alert(1)</script>"}], [], [], title="t", height=100)

    assert page.count("</script>") == 2          # o do vis-network e o do nosso script, nada injetado
    assert embedded_data(page)["nodes"][0]["name"].startswith("</script>")


def test_answer_view_highlights_cited_paths_and_question_entities():
    result = ask("O que a LGPD diz sobre consentimento?", mock=True)
    result.answer += " Caminho [P1]."

    data = embedded_data(answer_view(result))
    nodes = {n["id"]: n for n in data["nodes"]}
    hot = [e for e in data["edges"] if e["width"] == 3]

    assert nodes["lei:13709:2018"].get("shadow")      # início de um caminho = entidade da pergunta
    assert {(e["from"], e["to"]) for e in hot} == {("lei:13709:2018", "consentimento")}
    assert data["communities"][0]["name"] == "consentimento"   # maior PageRank da maior comunidade do exemplo



def recorded_lgpd():
    return mock_ask(DEMO_QUESTIONS[0])      # execução real gravada: cita [P], [C] e [Q]


def test_reasoning_view_goes_from_question_to_answer_through_cited_items():
    result = recorded_lgpd()
    data = embedded_data(reasoning_view(result))
    nodes = {n["id"]: n for n in data["nodes"]}
    levels = {nid: n["level"] for nid, n in nodes.items()}

    assert levels["question"] == 0 and levels["answer"] == max(levels.values())
    cited_chunks = {int(c) for c in re.findall(r"\[C(\d+)\]", result.answer)}
    assert {n["label"] for n in nodes.values() if n["label"].startswith("C")} == {f"C{c}" for c in cited_chunks}
    assert "query" in nodes and nodes["query"]["detail"].startswith("MATCH")
    assert all(e["to"] == "answer" for e in data["edges"] if e["label"] in {"Q"} | {f"C{c}" for c in cited_chunks})


def test_reasoning_view_follows_path_order_and_relation_direction():
    result = recorded_lgpd()
    first = sorted(cited_paths(result.answer))[0]
    path = result.paths[first - 1]
    data = embedded_data(reasoning_view(result))
    by_name = {n["name"]: n for n in data["nodes"]}

    assert by_name[path[0]]["level"] == 1                     # 1º nó do caminho = entidade da pergunta
    assert any(e["from"] == "question" and e["to"] == by_name[path[0]]["id"] for e in data["edges"])
    assert any(e["label"] == f"P{first}" and e["to"] == "answer" for e in data["edges"])


def test_reasoning_view_without_cited_paths_shows_the_first_ones_as_not_cited():
    result = recorded_lgpd().model_copy(update={"answer": "Resposta sem citar caminhos [C1]."})

    data = embedded_data(reasoning_view(result, max_paths=3))

    assert sum("(não citado)" in e["label"] for e in data["edges"]) == min(3, len(result.paths))
    assert "query" not in {n["id"] for n in data["nodes"]}     # [Q] não foi citada


def test_reasoning_view_uses_hierarchical_layout_and_the_map_uses_force_layout():
    assert '"direction": "LR"' in reasoning_view(recorded_lgpd())
    assert "forceAtlas2Based" in knowledge_map([], [])
