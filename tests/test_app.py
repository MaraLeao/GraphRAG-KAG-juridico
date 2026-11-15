"""Testes de fumaça da interface: abrem o app em modo demonstração (sem Neo4j nem LLM)."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from graphrag_juridico import graph_view
from graphrag_juridico.mock import DEMO_QUESTIONS

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
QUESTION = "O que a LGPD diz sobre consentimento?"


@pytest.fixture
def app(monkeypatch):
    """Modo demonstração; o mapa (que lê o Neo4j quando o modo é desligado) usa um grafo falso."""
    monkeypatch.setenv("GRAPHRAG_MOCK", "1")
    fake_map = ([{"id": "a", "label": "Concept", "name": "dignidade", "pagerank": 0.01, "community": 0, "mentions": 9}],
                [])
    monkeypatch.setattr(graph_view, "load_map", lambda limit: fake_map)
    test = AppTest.from_file(str(APP), default_timeout=30)
    test.run()
    return test


def ask_by_enter(app, question=QUESTION):
    app.text_input(key="question_input").input(question)
    app.button(key="submit").click().run()


def test_app_starts_in_demo_mode(app):
    assert not app.exception
    assert app.sidebar.toggle[0].value is True
    assert any("desligue o modo demonstração" in i.value for i in app.info)


def test_typing_alone_does_not_ask(app):
    app.text_input(key="question_input").input(QUESTION).run()

    assert not app.metric                         # nada foi perguntado sem Enter/botão


def test_question_shows_answer_confidence_and_sources(app):
    ask_by_enter(app)

    assert not app.exception
    assert any("consentimento" in m.value for m in app.markdown)
    assert any("genérica FIXA" in i.value for i in app.info)      # pergunta fora dos exemplos gravados
    assert app.metric[0].label == "Nível de confiança"
    assert any("Trechos (2)" in t.label for t in app.tabs)


def test_example_button_asks_immediately_with_its_recorded_run(app):
    app.button[0].click().run()

    assert not app.exception
    assert app.metric and app.text_input(key="question_input").value == DEMO_QUESTIONS[0]
    assert any("execução real gravada" in i.value for i in app.info)


def test_each_example_has_its_own_answer(app):
    answers = []
    for i in range(len(DEMO_QUESTIONS)):
        app.button[i].click().run()
        answers.append(app.session_state["last"]["result"]["answer"])

    assert len(set(answers)) == len(DEMO_QUESTIONS)


def test_changing_settings_keeps_last_answer_without_asking_again(app):
    ask_by_enter(app)

    app.sidebar.toggle[0].set_value(False).run()   # sair do modo demonstração NÃO dispara o LLM

    assert not app.exception and not app.error
    assert app.metric                               # a última resposta continua na tela
    assert any("modo demonstração" in c.value for c in app.caption)
    assert not any("Mapa" in e.value for e in app.error)   # o mapa carregou sozinho (grafo falso)



def test_paths_tab_switches_between_route_and_list(app):
    app.button[0].click().run()
    assert not app.exception and app.session_state["paths_view"] == "Rota da resposta"

    app.session_state["paths_view"] = "Lista"
    app.run()

    assert not app.exception
    assert any(m.value.startswith("**[P1]**") for m in app.markdown)
