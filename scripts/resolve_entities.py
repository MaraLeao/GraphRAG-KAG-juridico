"""Resolução de entidades: nomes extraídos pelo LLM -> chave canônica (evita nós duplicados).
Estratégia: dicionários de aliases (constituições, códigos, tribunais), chave numérica para leis
(tipo:número:ano) e normalização de texto para conceitos. Fusão por embedding fica para depois.
"""
import re
import unicodedata
from collections import Counter

KEY_PROP = {"Norm": "id", "Institution": "name", "Concept": "name", "Scholar": "name"}


def key(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    return re.sub(r"^(a|o|as|os) ", "", s)


# ---------- Norm ----------
OLD_CONSTITUTIONS = re.compile(r"\b(1824|1891|1934|1937|1946|1967|1969)\b")
# (regex sobre a chave, id, nome canônico) -- ordem importa (específicos primeiro)
CODES = [
    (r"codigo de processo civil.*1973", "codigo:cpc:1973", "Código de Processo Civil de 1973"),
    (r"codigo de processo civil|\bcpc\b", "codigo:cpc:2015", "Código de Processo Civil"),
    (r"codigo de processo penal|\bcpp\b", "codigo:cpp:1941", "Código de Processo Penal"),
    (r"codigo civil.*1916", "codigo:civil:1916", "Código Civil de 1916"),
    (r"codigo civil", "codigo:civil:2002", "Código Civil"),
    (r"codigo penal militar", "codigo:penal-militar", "Código Penal Militar"),
    (r"codigo penal.*1890", "codigo:penal:1890", "Código Penal de 1890"),
    (r"codigo penal", "codigo:penal:1940", "Código Penal"),
    (r"codigo de defesa do consumidor|\bcdc\b", "codigo:cdc:1990", "Código de Defesa do Consumidor"),
    (r"consolidacao das leis do trabalho|\bclt\b", "codigo:clt:1943", "Consolidação das Leis do Trabalho"),
    (r"codigo tributario nacional|\bctn\b", "codigo:ctn:1966", "Código Tributário Nacional"),
    (r"estatuto da crianca e (do )?adolescente|\beca\b", "lei:estatuto-eca", "Estatuto da Criança e do Adolescente"),
    (r"declaracao universal d[eo]s? direitos (humanos|do homem)", "intl:dudh", "Declaração Universal dos Direitos Humanos"),
    (r"convencao americana (de|sobre) direitos humanos|pacto de sa[no] jose", "intl:cadh", "Convenção Americana sobre Direitos Humanos"),
    (r"pacto internacional (dos|sobre) direitos civis e politicos", "intl:pidcp", "Pacto Internacional dos Direitos Civis e Políticos"),
    (r"declaracao dos direitos do homem e do cidadao", "intl:ddhc", "Declaração dos Direitos do Homem e do Cidadão"),
    (r"lei de introducao (as normas do direito brasileiro|ao codigo civil)|\blindb\b", "lei:lindb", "Lei de Introdução às Normas do Direito Brasileiro"),
    (r"lei de execucoes? penais?|execucao penal|\blep\b", "lei:7210:1984", "Lei 7.210/1984"),
    (r"lei de responsabilidade fiscal|\blrf\b", "lc:101:2000", "Lei Complementar 101/2000"),
    (r"estatuto da cidade", "lei:10257:2001", "Lei 10.257/2001"),
    (r"codigo eleitoral", "lei:4737:1965", "Lei 4.737/1965"),
    (r"ato das disposicoes constitucionais transitorias|\badct\b", "const:adct", "Ato das Disposições Constitucionais Transitórias"),
    (r"codigo de etica medica", "codigo:etica-medica", "Código de Ética Médica"),
    (r"codigo de transito", "codigo:transito:1997", "Código de Trânsito Brasileiro"),
    (r"marco civil da internet", "lei:12965:2014", "Lei 12.965/2014"),
    (r"lei geral de protecao de dados|\blgpd\b", "lei:13709:2018", "Lei 13.709/2018"),
    (r"lei maria da penha", "lei:11340:2006", "Lei 11.340/2006"),
    (r"lei de improbidade", "lei:8429:1992", "Lei 8.429/1992"),
]
LAW_RX = re.compile(r"(\d{1,3}(?:\.\d{3})+|\d+)\s*/\s*(\d{4}|\d{2})\b")
EC_RX = re.compile(r"\bemendas? constitu\w+ (?:n[o]? )?(\d{1,3})\b")
SUMULA_RX = re.compile(r"sumula(?: vinculante)?\s*(?:n[o]?\s*)?(\d+)")


def _fmt_number(n: str) -> str:
    return f"{int(n):,}".replace(",", ".")


def _year(y: str) -> int:
    y = int(y)
    return y if y > 100 else (1900 + y if y >= 30 else 2000 + y)


def resolve_norm(e: dict):
    k = key(e["name"])
    m_old = OLD_CONSTITUTIONS.search(k)
    if "constituicao" in k and "estadual" not in k and m_old:
        return f"const:{m_old.group(1)}", f"Constituição de {m_old.group(1)}", {"type": "constituição", "year": int(m_old.group(1))}
    m_ec = EC_RX.search(k)
    if m_ec:
        return f"ec:{int(m_ec.group(1))}", f"Emenda Constitucional {int(m_ec.group(1))}", {"type": "emenda constitucional", "number": m_ec.group(1)}
    if "constituicao" in k and "estadual" not in k and not OLD_CONSTITUTIONS.search(k) and (
            "federal" in k or "republica" in k or "1988" in k or k == "constituicao"):
        return "const:1988", "Constituição Federal", {"type": "constituição", "year": 1988}
    for rx, nid, name in CODES:
        if re.search(rx, k):
            return nid, name, {"type": "código"}
    m = SUMULA_RX.search(k)
    if m:
        court = "stj" if "stj" in k else "stf" if "stf" in k else "vinculante" if "vinculante" in k else "outra"
        vinc = " Vinculante" if "vinculante" in k else ""
        return f"sumula:{court}:{m.group(1)}{'v' if vinc else ''}", f"Súmula{vinc} {m.group(1)} do {court.upper()}", {"type": "súmula", "number": m.group(1)}
    number, year = e.get("number"), e.get("year")
    if not (number and year):
        m = LAW_RX.search(e["name"])
        if m:
            number, year = m.group(1), _year(m.group(2))
    if number and year:
        digits = re.sub(r"\D", "", str(number))
        if digits:
            kind = ("lc" if "complementar" in k else "decreto" if k.startswith("decreto") else
                    "mp" if "medida provisoria" in k else "resolucao" if k.startswith("resolucao") else
                    "ec" if "emenda" in k else "lei")
            label = {"lc": "Lei Complementar", "decreto": "Decreto", "mp": "Medida Provisória",
                     "resolucao": "Resolução", "ec": "Emenda Constitucional", "lei": "Lei"}[kind]
            return f"{kind}:{int(digits)}:{int(year)}", f"{label} {_fmt_number(digits)}/{int(year)}", {
                "type": e.get("norm_type") or label.lower(), "number": _fmt_number(digits), "year": int(year)}
    return f"norm:{k.replace(' ', '-')}", e["name"], {"type": e.get("norm_type")}


# ---------- Institution ----------
# sigla -> nome canônico
INSTITUTIONS = {
    "stf": "Supremo Tribunal Federal", "stj": "Superior Tribunal de Justiça", "tst": "Tribunal Superior do Trabalho",
    "tse": "Tribunal Superior Eleitoral", "cnj": "Conselho Nacional de Justiça", "cfm": "Conselho Federal de Medicina",
    "anvisa": "Agência Nacional de Vigilância Sanitária", "onu": "Organização das Nações Unidas",
    "oms": "Organização Mundial da Saúde", "omc": "Organização Mundial do Comércio", "sus": "Sistema Único de Saúde",
    "ibama": "IBAMA", "cade": "CADE", "cvm": "Comissão de Valores Mobiliários", "ue": "União Europeia",
    "mp": "Ministério Público", "oab": "Ordem dos Advogados do Brasil", "inss": "INSS", "bacen": "Banco Central do Brasil",
}
_STOP = {"de", "do", "da", "dos", "das", "estado", "e"}
_NAME2ACR = {" ".join(w for w in key(v).split() if w not in _STOP): a for a, v in INSTITUTIONS.items()}
_NAME2ACR.update({"uniao europeia": "ue", "ministerio publico": "mp", "banco central": "bacen"})


def resolve_institution(e: dict):
    reduced = " ".join(w for w in key(e["name"]).split() if w not in _STOP)
    acr = key(e.get("acronym") or "")
    group = _NAME2ACR.get(reduced) or (acr if acr in INSTITUTIONS else None)
    if group:
        return INSTITUTIONS[group], INSTITUTIONS[group], {"acronym": group.upper() if group not in ("ibama", "cade") else INSTITUTIONS[group]}
    return reduced, e["name"], {"acronym": e.get("acronym")}


# ---------- Concept ----------
SYNONYMS = {
    "dignidade humana": "dignidade da pessoa humana", "principio da dignidade da pessoa humana": "dignidade da pessoa humana",
    "direito a saude": "direito à saúde", "direitos humanos": "direitos humanos",
    "principio da legalidade": "princípio da legalidade", "devido processo": "devido processo legal",
}


def resolve_concept(e: dict):
    k = key(e["name"])
    k = SYNONYMS_KEY.get(k, k)
    return k, e["name"].strip().lower(), {}


SYNONYMS_KEY = {key(a): key(b) for a, b in SYNONYMS.items()}


# ---------- Scholar ----------
def resolve_scholar(e: dict):
    if len(e["name"].split()) < 2:  # sobrenome solto é ambíguo ("Silva", "Santos")
        return None
    k = key(e["name"])
    return k, e["name"].strip(), {}


RESOLVERS = {"Norm": resolve_norm, "Institution": resolve_institution, "Concept": resolve_concept, "Scholar": resolve_scholar}


class Resolver:
    """Acumula nomes por chave canônica e escolhe o nome de exibição (mais frequente, empate = mais longo)."""

    def __init__(self):
        self.names = {}   # (label, key) -> Counter(nome)
        self.props = {}   # (label, key) -> dict

    def resolve(self, e: dict):
        out = RESOLVERS[e["type"]](e)
        if out is None:
            return None
        k, name, props = out
        ident = (e["type"], k)
        self.names.setdefault(ident, Counter())[name] += 1
        self.props.setdefault(ident, {}).update({p: v for p, v in props.items() if v is not None})
        return ident

    def display(self, ident):
        c = self.names[ident]
        return max(c, key=lambda n: (c[n], len(n)))
