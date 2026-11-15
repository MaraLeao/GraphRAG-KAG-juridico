"""Métricas automáticas de uma resposta (sem LLM nem banco), usadas pela interface e pela avaliação.

Sem gabarito (valem para qualquer pergunta):
- `citations_valid`: fração das citações [C]/[P]/[Q] que existem no contexto (citação inventada = alucinação);
- `grounded_sentences`: fração das frases da resposta que têm ao menos uma citação;
- `unsupported_norms`: normas citadas na resposta ("Lei 13.709/2018", "LGPD"...) que não aparecem em nenhum
  trecho, caminho ou resultado de consulta do contexto (indício forte de alucinação);
- `abstained`: a resposta diz que não há informação suficiente.

Com gabarito (perguntas do conjunto de avaliação):
- `term_recall`: fração dos itens esperados encontrados na resposta (cada item aceita sinônimos);
- `correct`: pergunta respondível com recall >= 0,5 e sem abstenção, ou pergunta armadilha com abstenção.
"""
import re
from dataclasses import asdict, dataclass

from .ask import check_citations, is_no_info
from .contracts import AskResult
from .entities import NORM_ALIASES, norm_ids_in, normalize
from .graph_view import generated_cypher

CITATION = re.compile(r"\[(?:C\d+|P\d+|Q)\]")
SENTENCE_END = re.compile(r"(?<=[.!?:])\s+|\n+")
MIN_SENTENCE = 25          # frases mais curtas (títulos, "Fontes:") não contam
RECALL_TO_PASS = 0.5
METRIC_HELP = {
    "Acerto (gabarito)": "os itens esperados estão na resposta (perguntas respondíveis) ou houve abstenção "
                         "(perguntas armadilha). Só para as perguntas do conjunto de avaliação.",
    "Abstenção": "a resposta diz que o corpus não tem informação suficiente. Correta nas armadilhas, indevida "
                 "nas perguntas que têm resposta.",
    "Citações válidas": "fração das citações [C], [P] e [Q] que apontam para algo que existe no contexto. "
                        "Citação inventada é sinal de alucinação.",
    "Frases com fonte": "fração das frases da resposta que têm ao menos uma citação.",
    "Normas sem suporte": "leis citadas na resposta que não aparecem em nenhum trecho, caminho ou resultado de "
                          "consulta do contexto: não verificáveis no corpus, indício de alucinação. Menor é melhor.",
}
NORM_NAMES = {"codigo:civil:2002": "Código Civil", "codigo:penal:1940": "Código Penal"}   # sem sigla curta


@dataclass
class Reference:
    """Gabarito de uma pergunta: itens esperados (cada um com sinônimos) ou abstenção esperada."""
    expected: list[list[str]]
    should_abstain: bool = False


@dataclass
class Metrics:
    abstained: bool
    citations_valid: float
    grounded_sentences: float
    unsupported_norms: list[str]
    term_recall: float | None = None
    correct: bool | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def sentences(answer: str) -> list[str]:
    return [s.strip() for s in SENTENCE_END.split(answer) if len(s.strip()) >= MIN_SENTENCE]


def grounded_fraction(answer: str) -> float:
    """Fração das frases com citação. A frase de abstenção não é uma afirmação e fica de fora;
    uma abstenção sem outras afirmações conta como totalmente fundamentada."""
    parts = [s for s in sentences(answer) if not is_no_info(s)]
    if not parts:
        return 1.0 if is_no_info(answer) else 0.0
    return sum(bool(CITATION.search(s)) for s in parts) / len(parts)


def context_text(result: AskResult) -> str:
    """Tudo o que o método recebeu como contexto: trechos, caminhos, nós e resultado da consulta [Q]."""
    return " ".join([*(c.text for c in result.chunks), *(" ".join(p) for p in result.paths),
                     *(n.name for n in result.nodes), *(str(r) for r in result.cypher_rows)])


def context_norms(result: AskResult) -> list[str]:
    """Normas presentes no contexto. Calcule sobre o contexto COMPLETO: as gravações cortam os trechos."""
    return sorted(set(norm_ids_in(context_text(result))))


def unsupported_norms(result: AskResult, known_context_norms: list[str] | None = None) -> list[str]:
    """Normas que a resposta cita e que não estão no contexto (ignora as que a própria pergunta menciona).
    `known_context_norms`: normas do contexto completo, calculadas antes de cortar os trechos para gravar."""
    in_context = known_context_norms if known_context_norms is not None else context_norms(result)
    allowed = set(in_context) | set(norm_ids_in(result.question))
    return [n for n in norm_ids_in(result.answer) if n not in allowed]


def norm_label(norm_id: str) -> str:
    """Nome legível de um id canônico: 'lei:8078:1990' -> 'Lei 8.078/1990', 'codigo:cdc:1990' -> 'CDC'."""
    kind, *rest = norm_id.split(":")
    if kind in {"lei", "lc", "decreto"} and len(rest) == 2 and rest[0].isdigit():
        label = {"lei": "Lei", "lc": "Lei Complementar", "decreto": "Decreto"}[kind]
        return f"{label} {int(rest[0]):,}/{rest[1]}".replace(",", ".")
    if norm_id in NORM_NAMES:
        return NORM_NAMES[norm_id]
    aliases = sorted((a for a, target in NORM_ALIASES.items() if target == norm_id), key=len)
    return aliases[0].upper() if aliases else norm_id


# Siglas e leis que aparecem nas respostas, para a legenda da interface e do relatório.
NORM_GLOSSARY = {
    "CF": "Constituição Federal de 1988",
    "CDC": "Código de Defesa do Consumidor (Lei 8.078/1990)",
    "CPC": "Código de Processo Civil (Lei 13.105/2015)",
    "CPP": "Código de Processo Penal (Decreto-Lei 3.689/1941)",
    "CTN": "Código Tributário Nacional (Lei 5.172/1966)",
    "CLT": "Consolidação das Leis do Trabalho (Decreto-Lei 5.452/1943)",
    "ECA": "Estatuto da Criança e do Adolescente (Lei 8.069/1990)",
    "LEP": "Lei de Execução Penal (Lei 7.210/1984)",
    "LRF": "Lei de Responsabilidade Fiscal (Lei Complementar 101/2000)",
    "LINDB": "Lei de Introdução às Normas do Direito Brasileiro (Decreto-Lei 4.657/1942)",
    "ADCT": "Ato das Disposições Constitucionais Transitórias",
    "Código Civil": "Lei 10.406/2002",
    "Código Penal": "Decreto-Lei 2.848/1940",
    "Lei 13.709/2018": "LGPD, Lei Geral de Proteção de Dados Pessoais",
    "Lei 12.965/2014": "Marco Civil da Internet",
    "Lei 10.406/2002": "Código Civil",
    "Lei 8.078/1990": "Código de Defesa do Consumidor",
    "Lei 8.069/1990": "Estatuto da Criança e do Adolescente",
    "Lei 13.853/2019": "altera a LGPD e cria a ANPD",
    "Lei 9.507/1997": "Lei do Habeas Data",
    "Lei 12.527/2011": "Lei de Acesso à Informação",
    "Lei 12.850/2013": "Lei das Organizações Criminosas",
    "Lei 9.296/1996": "Lei de Interceptação Telefônica",
    "Lei 12.737/2012": "Lei de crimes informáticos (Lei Carolina Dieckmann)",
    "Lei 8.906/1994": "Estatuto da Advocacia e da OAB",
}


def glossary_entries(labels) -> list[tuple[str, str]]:
    """(sigla ou lei, descrição) das conhecidas, na ordem em que aparecem, sem repetir."""
    return [(label, NORM_GLOSSARY[label]) for label in dict.fromkeys(labels) if label in NORM_GLOSSARY]


def glossary(labels) -> list[str]:
    """Legenda em texto ("CF: Constituição Federal de 1988"), usada no relatório."""
    return [f"{label}: {description}" for label, description in glossary_entries(labels)]


def term_recall(answer: str, expected: list[list[str]]) -> float:
    text = normalize(answer)
    found = sum(any(normalize(option) in text for option in item) for item in expected)
    return found / len(expected) if expected else 0.0


def evaluate(result: AskResult, reference: Reference | None = None,
             known_context_norms: list[str] | None = None) -> Metrics:
    has_query = generated_cypher(result) is not None
    citations, _ = check_citations(result.answer, len(result.chunks), len(result.paths), has_query)
    metrics = Metrics(abstained=is_no_info(result.answer), citations_valid=round(citations, 4),
                      grounded_sentences=round(grounded_fraction(result.answer), 4),
                      unsupported_norms=unsupported_norms(result, known_context_norms))
    if reference is not None:
        if reference.should_abstain:
            metrics.correct = metrics.abstained
        else:
            metrics.term_recall = round(term_recall(result.answer, reference.expected), 4)
            metrics.correct = not metrics.abstained and metrics.term_recall >= RECALL_TO_PASS
    return metrics
