"""Ligação de entidades: encontra no grafo as normas, conceitos e instituições citados na pergunta.

As entidades são o ponto de partida da travessia multi-hop. Quatro estratégias, em ordem de confiança:
1. apelidos e números de normas ("LGPD", "CDC", "Lei 13.709/2018") -> `Norm.id` canônico;
2. siglas de instituições ("STF", "ANPD") -> `Institution.acronym`;
3. similaridade de cosseno com os conceitos (`vec_concept`);
4. fulltext nos nomes das entidades (`ft_entities`).

As notas de cosseno entre a pergunta e nomes curtos ficam muito próximas (0,75 a 0,90), então nenhum
corte fixo separa conceito citado de conceito apenas parecido. Por isso, do vetor entra sempre o mais
próximo; os demais candidatos (vetor e fulltext) só entram se todas as palavras do nome estão na pergunta.
"""
import re
import unicodedata
from typing import Any

from .db import read
from .retrieval import lucene

MAX_ENTITIES = 4
VECTOR_MIN_SCORE = 0.6       # cosseno mínimo entre a pergunta e o nome do conceito

# Apelidos usuais -> id canônico (mesmas chaves usadas na resolução de entidades da ingestão).
NORM_ALIASES = {
    "lgpd": "lei:13709:2018", "lei geral de protecao de dados": "lei:13709:2018",
    "cf": "const:1988", "cf/88": "const:1988", "constituicao federal": "const:1988", "constituicao de 1988": "const:1988",
    "cdc": "codigo:cdc:1990", "codigo de defesa do consumidor": "codigo:cdc:1990",
    "clt": "codigo:clt:1943", "consolidacao das leis do trabalho": "codigo:clt:1943",
    "cpc": "codigo:cpc:2015", "codigo de processo civil": "codigo:cpc:2015",
    "cpp": "codigo:cpp:1941", "codigo de processo penal": "codigo:cpp:1941",
    "ctn": "codigo:ctn:1966", "codigo tributario nacional": "codigo:ctn:1966",
    "eca": "lei:estatuto-eca", "estatuto da crianca e do adolescente": "lei:estatuto-eca",
    "codigo civil": "codigo:civil:2002", "codigo penal": "codigo:penal:1940",
    "marco civil da internet": "lei:12965:2014", "lei maria da penha": "lei:11340:2006",
    "lep": "lei:7210:1984", "lei de execucao penal": "lei:7210:1984",
    "lrf": "lc:101:2000", "lei de responsabilidade fiscal": "lc:101:2000",
    "lindb": "lei:lindb", "adct": "const:adct", "lei de improbidade": "lei:8429:1992",
    "estatuto da cidade": "lei:10257:2001", "codigo eleitoral": "lei:4737:1965",
}
NORM_KINDS = {"lei complementar": "lc", "lei": "lei", "decreto": "decreto"}
NORM_NUMBER_RX = re.compile(r"\b(lei complementar|lei|decreto)\s*(?:n[o°.]*\s*)?(\d{1,3}(?:\.\d{3})+|\d+)\s*/\s*(\d{4})\b")
ACRONYM_RX = re.compile(r"\b[A-Z]{2,6}\b")
STOPWORDS = set("""
qual quais como onde quando porque sobre entre para pela pelo pelas pelos com sem uma umas uns que quem
relacao existe existem esta estao seja sao diz dizem trata tratam segundo acordo direito direitos lei leis
""".split())

NORMS_QUERY = "MATCH (n:Norm) WHERE n.id IN $ids RETURN elementId(n) AS id, n.id AS key, 'Norm' AS label, n.name AS name"
INSTITUTIONS_QUERY = """
MATCH (i:Institution) WHERE i.acronym IN $acronyms
RETURN elementId(i) AS id, 'Institution' AS label, i.name AS name
"""
CONCEPTS_QUERY = """
CALL db.index.vector.queryNodes('vec_concept', $n, $vec) YIELD node, score
WHERE score >= $min_score
RETURN elementId(node) AS id, 'Concept' AS label, node.name AS name, score
"""
FULLTEXT_QUERY = """
CALL db.index.fulltext.queryNodes('ft_entities', $q, {limit: $n}) YIELD node, score
RETURN elementId(node) AS id, labels(node)[0] AS label, node.name AS name, score
"""


def normalize(text: str) -> str:
    """Minúsculas e sem acentos (mantém pontuação, para números como 13.709/2018)."""
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def norm_ids_in(question: str) -> list[str]:
    """Ids canônicos de normas citadas por apelido ou por número/ano, na ordem em que aparecem."""
    text = normalize(question)
    found = []
    for alias, norm_id in NORM_ALIASES.items():
        match = re.search(rf"(?<![\w/]){re.escape(alias)}(?![\w/])", text)
        if match:
            found.append((match.start(), norm_id))
    for match in NORM_NUMBER_RX.finditer(text):
        kind, number, year = match.groups()
        found.append((match.start(), f"{NORM_KINDS[kind]}:{int(number.replace('.', ''))}:{year}"))
    return list(dict.fromkeys(norm_id for _, norm_id in sorted(found)))


def institution_acronyms(question: str) -> list[str]:
    """Siglas em maiúsculas (STF, ANPD...); as que forem apelidos de normas ficam de fora."""
    return [a for a in dict.fromkeys(ACRONYM_RX.findall(question)) if a.lower() not in NORM_ALIASES]


def keywords(question: str) -> list[str]:
    """Palavras com conteúdo (4+ letras, fora da lista de palavras vazias) para a busca fulltext."""
    words = re.findall(r"\w+", question.lower())
    return [w for w in words if len(w) >= 4 and normalize(w) not in STOPWORDS]


def mentioned(name: str, question: str) -> bool:
    """Todas as palavras do nome (3+ letras) aparecem na pergunta, sem acento e sem diferenciar maiúsculas."""
    asked = set(re.findall(r"\w+", normalize(question)))
    words = [w for w in re.findall(r"\w+", normalize(name)) if len(w) >= 3]
    return bool(words) and all(w in asked for w in words)


def _overlaps(a: str, b: str) -> bool:
    """Um nome contém o outro como palavras inteiras ("consentimento" x "consentimento informado")."""
    a, b = f" {normalize(a)} ", f" {normalize(b)} "
    return a in b or b in a


def merge(*groups: list[dict[str, Any]], limit: int = MAX_ENTITIES) -> list[dict[str, Any]]:
    """Junta os grupos na ordem dada (o mais confiável primeiro), sem repetir nós nem variações
    do mesmo nome, que ocupariam as vagas com quase o mesmo conceito."""
    merged: list[dict[str, Any]] = []
    for group in groups:
        for entity in group:
            if not any(entity["id"] == m["id"] or _overlaps(entity["name"], m["name"]) for m in merged):
                merged.append(entity)
    return merged[:limit]


def link_entities(question: str, vec: list[float], limit: int = MAX_ENTITIES) -> list[dict[str, Any]]:
    """Entidades do grafo citadas na pergunta: [{id, label, name, via}], `id` = elementId."""
    norm_ids = norm_ids_in(question)
    norms = read(NORMS_QUERY, ids=norm_ids) if norm_ids else []
    norms.sort(key=lambda r: norm_ids.index(r.pop("key")))
    acronyms = institution_acronyms(question)
    institutions = read(INSTITUTIONS_QUERY, acronyms=acronyms) if acronyms else []
    # pede mais candidatos do que vagas: variações do mesmo nome são descartadas no merge
    concepts = read(CONCEPTS_QUERY, n=limit * 3, vec=vec, min_score=VECTOR_MIN_SCORE)
    words = keywords(question)
    fulltext = read(FULLTEXT_QUERY, q=lucene(" ".join(words)), n=limit * 3) if words else []
    concepts = concepts[:1] + [r for r in concepts[1:] if mentioned(r["name"], question)]
    fulltext = [r for r in fulltext if mentioned(r["name"], question)]

    tagged = [[{**r, "via": via} for r in rows] for via, rows in
              [("apelido", norms), ("sigla", institutions), ("vetor", concepts), ("fulltext", fulltext)]]
    return merge(*tagged, limit=limit)
