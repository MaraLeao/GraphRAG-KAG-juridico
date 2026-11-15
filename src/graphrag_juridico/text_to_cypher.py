"""Text-to-Cypher: o LLM traduz a pergunta em UMA consulta Cypher de leitura sobre o grafo.

Camadas de proteção, da mais barata para a mais cara:
1. prompt com o schema, regras e exemplos (few-shot); normas citadas por apelido ("LGPD") viram dicas de id;
2. validação estática: somente leitura, só labels e relações do schema, caminhos variáveis com limite
   (até 4 saltos), uma única consulta e `LIMIT` obrigatório (acrescentado ou reduzido se preciso);
3. `EXPLAIN` no Neo4j antes de executar (pega erros de sintaxe sem rodar a consulta);
4. execução em transação somente leitura (`db.read`): o Neo4j recusa escrita mesmo que algo escape.

Se a consulta falha em 2 ou 3, o erro volta para o LLM em uma nova tentativa. Se ainda assim falha,
o resultado vem sem Cypher (`ok = False`) e quem chamou usa a busca híbrida.

Uso:
    python -m graphrag_juridico.text_to_cypher "Quais instituições interpretam a LGPD?"
"""
import argparse
import re
from dataclasses import dataclass, field
from typing import Any

from .config import chat, get_settings
from .db import read
from .entities import norm_ids_in

MAX_ATTEMPTS = 2
DEFAULT_LIMIT = 25
MAX_LIMIT = 100
MAX_HOPS = 4

LABELS = {"Article", "Chunk", "Category", "Norm", "Institution", "Concept", "Scholar"}
RELATIONS = {"IN_CATEGORY", "HAS_CHUNK", "MENTIONS", "CITES_NORM", "CITES_SCHOLAR", "ABOUT", "REGULATES",
             "AMENDS", "REVOKES", "ISSUED_BY", "INTERPRETS", "SPECIALIZES"}
# procedimentos de leitura permitidos (índices); qualquer outro CALL é bloqueado
ALLOWED_PROCEDURES = {"db.index.fulltext.querynodes", "db.index.vector.querynodes"}
WRITE_CLAUSES = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|FOREACH|LOAD\s+CSV|GRANT|REVOKE|DENY|USE|ALTER|RENAME|"
    r"START|STOP|TERMINATE|ENABLE|DISABLE)\b", re.IGNORECASE)

SCHEMA = """
Nós (label {propriedades}):
- Article {id, title, summary, link, word_count}: artigo jurídico do corpus
- Chunk {id, text, order, section}: trecho de um artigo
- Category {name}: área do Direito do artigo (ex.: 'Direito Penal')
- Norm {id, name, type, number, year}: norma (lei, decreto, súmula, constituição, código...).
  Ids canônicos: 'lei:13709:2018' (LGPD), 'const:1988' (Constituição Federal), 'codigo:civil:2002',
  'codigo:cdc:1990', 'codigo:clt:1943'; name legível, ex.: 'Lei 13.709/2018'
- Institution {name, acronym}: órgão ou tribunal (ex.: name 'Supremo Tribunal Federal', acronym 'STF')
- Concept {name}: conceito jurídico em minúsculas (ex.: 'consentimento', 'dignidade da pessoa humana')
- Scholar {name}: doutrinador citado

Relações (origem)-[TIPO {propriedades}]->(destino):
- (Article)-[:IN_CATEGORY]->(Category)
- (Article)-[:HAS_CHUNK]->(Chunk)
- (Chunk)-[:MENTIONS {evidence}]->(Norm|Institution|Concept|Scholar)
- (Article)-[:CITES_NORM {times}]->(Norm)
- (Article)-[:CITES_SCHOLAR {times}]->(Scholar)
- (Article)-[:ABOUT {weight}]->(Concept)
- (Norm)-[:REGULATES {evidence, times}]->(Concept)
- (Norm)-[:AMENDS]->(Norm), (Norm)-[:REVOKES]->(Norm)
- (Norm)-[:ISSUED_BY]->(Institution)
- (Institution)-[:INTERPRETS]->(Norm)
- (Concept)-[:SPECIALIZES]->(Concept)
""".strip()

SYSTEM = f"""Você converte perguntas em português para UMA consulta Cypher de leitura no Neo4j.

Schema do grafo:
{SCHEMA}

Regras:
- Somente leitura: nunca use CREATE, MERGE, DELETE, SET, REMOVE, DROP, LOAD CSV nem CALL de procedimentos.
- Use apenas os labels e relações do schema, na direção indicada.
- Compare nomes com toLower(x.name) CONTAINS 'termo em minúsculas'; para normas conhecidas, prefira o id.
- Caminhos variáveis precisam de limite máximo, no máximo {MAX_HOPS} saltos (ex.: -[:SPECIALIZES*1..2]->).
- Retorne nomes legíveis (name, title) com DISTINCT e termine com LIMIT {DEFAULT_LIMIT}.
- Responda apenas com a consulta, num bloco ```cypher.

Exemplos:
Pergunta: Quais doutrinadores são citados por artigos da categoria Direito Penal?
```cypher
MATCH (c:Category)<-[:IN_CATEGORY]-(:Article)-[:CITES_SCHOLAR]->(s:Scholar)
WHERE toLower(c.name) CONTAINS 'penal'
RETURN DISTINCT s.name AS doutrinador LIMIT 25
```
Pergunta: Quais normas foram revogadas pelo Código Civil?
```cypher
MATCH (n:Norm {{id: 'codigo:civil:2002'}})-[:REVOKES]->(r:Norm)
RETURN DISTINCT r.name AS norma LIMIT 25
```
Pergunta: Quais artigos tratam de conceitos regulados pela LGPD?
```cypher
MATCH (:Norm {{id: 'lei:13709:2018'}})-[:REGULATES]->(c:Concept)<-[:ABOUT]-(a:Article)
RETURN DISTINCT a.title AS artigo, c.name AS conceito LIMIT 25
```"""

CODE_BLOCK = re.compile(r"```(?:cypher)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
STRING_OR_COMMENT = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"|//[^\n]*|/\*.*?\*/", re.DOTALL)
NODE_LABELS = re.compile(r"\(\s*\w*\s*((?::\s*`?\w+`?\s*)+)")
REL_TYPES = re.compile(r"\[\s*\w*\s*:\s*([\w|`:\s!&]+?)\s*(?:\*|\{|\]|WHERE\b)", re.IGNORECASE)
VAR_LENGTH = re.compile(r"\[[^\[\]]*?\*\s*(\d*)\s*(\.\.)?\s*(\d*)[^\[\]]*\]")  # só dentro de -[...]-, não count(*)
PROCEDURE = re.compile(r"\bCALL\s+([\w.]+)\s*\(", re.IGNORECASE)
LIMIT_AT_END = re.compile(r"\bLIMIT\s+(\d+)\s*;?\s*$", re.IGNORECASE)


class CypherValidationError(ValueError):
    """Consulta recusada pela validação estática (o motivo vai para o LLM na nova tentativa)."""


@dataclass
class CypherResult:
    question: str
    cypher: str | None = None                       # última consulta executada com sucesso
    rows: list[dict[str, Any]] = field(default_factory=list)
    attempts: list[dict[str, str]] = field(default_factory=list)  # [{cypher, error}] das tentativas que falharam

    @property
    def ok(self) -> bool:
        return self.cypher is not None


def extract_cypher(text: str) -> str:
    """Tira a consulta do bloco ```cypher (ou usa o texto todo, se não houver bloco)."""
    match = CODE_BLOCK.search(text)
    return (match.group(1) if match else text).strip()


def _code_only(cypher: str) -> str:
    """Remove strings e comentários, para que 'set' dentro de um texto não pareça uma cláusula."""
    return STRING_OR_COMMENT.sub("''", cypher)


def validate(cypher: str) -> str:
    """Devolve a consulta pronta para executar ou levanta `CypherValidationError`."""
    cypher = cypher.strip().rstrip(";").strip()
    code = _code_only(cypher)
    if not cypher:
        raise CypherValidationError("consulta vazia")
    if ";" in code:
        raise CypherValidationError("apenas uma consulta é permitida")
    if match := WRITE_CLAUSES.search(code):
        raise CypherValidationError(f"cláusula proibida (somente leitura): {match.group(1).upper()}")
    for proc in PROCEDURE.findall(code):
        if proc.lower() not in ALLOWED_PROCEDURES:
            raise CypherValidationError(f"procedimento não permitido: {proc}")
    if not re.search(r"\bRETURN\b", code, re.IGNORECASE):
        raise CypherValidationError("a consulta precisa de RETURN")

    for group in NODE_LABELS.findall(code):
        for label in re.findall(r"\w+", group):
            if label not in LABELS:
                raise CypherValidationError(f"label inexistente no schema: {label}")
    for group in REL_TYPES.findall(code):
        for rel in re.findall(r"\w+", group):
            if rel not in RELATIONS:
                raise CypherValidationError(f"relação inexistente no schema: {rel}")
    for low, dots, high in VAR_LENGTH.findall(code):
        upper = high if dots else low
        if not upper or int(upper) > MAX_HOPS:
            raise CypherValidationError(f"caminho variável precisa de limite de até {MAX_HOPS} saltos")

    if match := LIMIT_AT_END.search(cypher):
        if int(match.group(1)) > MAX_LIMIT:
            cypher = cypher[:match.start()] + f"LIMIT {MAX_LIMIT}"
    else:
        cypher = f"{cypher}\nLIMIT {DEFAULT_LIMIT}"
    return cypher


def hints(question: str) -> str:
    """Ids canônicos das normas citadas por apelido ou número, para o LLM não adivinhar nomes."""
    ids = norm_ids_in(question)
    return f"\nDica: normas citadas na pergunta e seus ids: {', '.join(ids)}" if ids else ""


def build_messages(question: str, failed: list[dict[str, str]]) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Pergunta: {question}{hints(question)}"}]
    for attempt in failed:
        messages += [{"role": "assistant", "content": f"```cypher\n{attempt['cypher']}\n```"},
                     {"role": "user", "content": f"Essa consulta falhou: {attempt['error']}\n"
                                                 "Corrija e responda só com a nova consulta."}]
    return messages


def generate(messages: list[dict[str, str]]) -> str:
    return extract_cypher(chat(messages, model=get_settings().cypher_model))


def run(question: str, max_attempts: int = MAX_ATTEMPTS) -> CypherResult:
    """Gera, valida, confere com EXPLAIN e executa; o erro de cada falha alimenta a tentativa seguinte."""
    result = CypherResult(question=question)
    for _ in range(max_attempts):
        cypher = generate(build_messages(question, result.attempts))
        try:
            cypher = validate(cypher)
            read(f"EXPLAIN {cypher}")
            rows = read(cypher)
        except Exception as error:  # validação, sintaxe ou execução: tudo vira dica para a nova tentativa
            result.attempts.append({"cypher": cypher, "error": str(error).splitlines()[0][:300]})
            continue
        result.cypher, result.rows = cypher, rows
        break
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera e executa Cypher a partir de uma pergunta (usa o LLM).")
    parser.add_argument("question")
    args = parser.parse_args()

    result = run(args.question)
    for i, attempt in enumerate(result.attempts, 1):
        print(f"Tentativa {i} falhou: {attempt['error']}\n{attempt['cypher']}\n")
    if not result.ok:
        print("Sem Cypher válida: use a busca híbrida.")
        return
    print(result.cypher, f"\n\n{len(result.rows)} linhas:")
    for row in result.rows[:15]:
        print("  ", row)


if __name__ == "__main__":
    main()
