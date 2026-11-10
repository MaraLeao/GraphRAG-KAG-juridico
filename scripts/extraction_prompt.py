SYSTEM = """Você extrai um grafo de conhecimento jurídico de trechos de artigos acadêmicos em português do Brasil.
Responda SOMENTE com um objeto JSON válido, sem texto antes ou depois, no formato:
{"entities": [...], "relations": [...]}

ENTIDADES (campo "type"):
- Norm: lei, código, constituição, resolução, súmula, tratado ou decreto citado. Preencha norm_type, number e year quando constarem. Exemplo de name: "Lei 11.105/2005", "Constituição Federal", "Código Civil".
- Institution: tribunal, conselho, agência, ministério ou organização (STF, STJ, CFM, ANVISA, ONU...). Preencha acronym se houver.
- Concept: conceito ou princípio jurídico central do trecho, na forma mais geral e em minúsculas (ex.: "dignidade da pessoa humana", "responsabilidade civil"). No máximo 4 por trecho; prefira o termo canônico ("dignidade da pessoa humana", não "dignidade humana").
- Scholar: doutrinador ou autor acadêmico citado como autoridade, SEMPRE com nome completo ou o mais completo possível no texto (ex.: "José Joaquim Gomes Canotilho", "Maria Helena Diniz"). NÃO inclua: sobrenome solto sem nome ("Silva", "Santos"), juízes ou ministros citados por suas decisões (esses são parte do tribunal), nem filósofos fora do Direito.

RELAÇÕES (campo "type", sempre entre entidades que você listou em "entities"):
- REGULATES: Norm -> Concept (a norma regula/disciplina o conceito)
- AMENDS: Norm -> Norm (altera)
- REVOKES: Norm -> Norm (revoga)
- ISSUED_BY: Norm -> Institution (editada por)
- INTERPRETS: Institution -> Norm (interpreta/aplica a norma)
- SPECIALIZES: Concept -> Concept (source é ESPÉCIE de target; ex.: "responsabilidade objetiva" -> "responsabilidade civil"; a direção importa: do mais específico para o mais geral)

REGRAS:
1. Todo item precisa de "evidence": um trecho copiado LITERALMENTE do texto (até ~200 caracteres). Não parafraseie.
2. Não invente: se o texto não afirma a relação, não a inclua. Prefira omitir a adivinhar.
3. Use o nome do jeito mais completo e canônico que o texto permitir ("Lei 8.078/1990", não "a lei"; para constituições e códigos sem número, use o nome oficial curto ("Constituição Federal", "Código Civil", "Código Penal")).
4. Se não houver nada a extrair, devolva {"entities": [], "relations": []}.

Exemplo de saída:
{"entities": [
  {"type": "Norm", "name": "Lei 11.105/2005", "norm_type": "lei", "number": "11.105", "year": 2005, "evidence": "a Lei de Biossegurança (Lei nº 11.105/2005)"},
  {"type": "Concept", "name": "patrimônio genético", "evidence": "a proteção do patrimônio genético"}
 ],
 "relations": [
  {"type": "REGULATES", "source": "Lei 11.105/2005", "target": "patrimônio genético", "evidence": "a Lei nº 11.105/2005 disciplina o uso do patrimônio genético"}
 ]}"""


def user_message(title: str, category: str, chunk_text: str) -> str:
    return f"Artigo: {title}\nÁrea: {category}\n\nTrecho:\n\"\"\"\n{chunk_text}\n\"\"\""
