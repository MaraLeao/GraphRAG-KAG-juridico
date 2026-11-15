# Modelagem do Grafo de Conhecimento Jurídico (LPG)

Este documento descreve o modelo **Labeled Property Graph (LPG)** implementado no Neo4j para o sistema GraphRAG/KAG sobre artigos jurídicos. Ele cobre os nós, relações, constraints e índices definidos em `schema.cypher` e `schema_vector.cypher`.

```mermaid
flowchart LR
    Category((Category))
    Article((Article))
    Chunk((Chunk))
    Norm((Norm))
    Institution((Institution))
    Concept((Concept))
    Scholar((Scholar))

    Article -->|IN_CATEGORY| Category
    Article -->|HAS_CHUNK| Chunk
    Article -->|CITES_NORM| Norm
    Article -->|CITES_SCHOLAR| Scholar
    Article -->|ABOUT| Concept

    Chunk -->|MENTIONS| Norm
    Chunk -->|MENTIONS| Institution
    Chunk -->|MENTIONS| Concept
    Chunk -->|MENTIONS| Scholar

    Norm -->|REGULATES| Concept
    Norm -->|AMENDS| Norm
    Norm -->|REVOKES| Norm
    Norm -->|ISSUED_BY| Institution
    Institution -->|INTERPRETS| Norm
    Concept -->|SPECIALIZES| Concept
``` 

## 1. Visão Geral

O grafo representa o conhecimento extraído de ~1.000 artigos do [Âmbito Jurídico](https://huggingface.co/datasets/celsowm/ambito_juridico_artigos) em 17 áreas do Direito. A modelagem conecta trechos de texto (`Chunk`) a entidades jurídicas (`Norm`, `Institution`, `Concept`, `Scholar`), permitindo tanto busca semântica por embedding quanto travessia estruturada via Cypher.

Nomes de labels e tipos de relação estão em inglês; valores e conteúdo em português.

**Estatísticas do banco (snapshot de 14/11):**

| Métrica | Valor |
|---|---|
| Total de nós | 29.705 |
| Total de relações | 111.547 |
| Chunks com embedding | 14.048 |
| Comunidades Louvain | 367 |
| Constraints | 7 |

---

## 2. Nós (Labels)

| Label | Constraint de Unicidade | Propriedades Principais | Descrição |
|---|---|---|---|
| `Article` | `article_id` → `id` | `id`, `titulo`, `resumo`, `categoria`, `link` | Artigo jurídico completo do dataset. Ponto de entrada do grafo. |
| `Chunk` | `chunk_id` → `id` | `id`, `text`, `embedding` (1024 dim) | Trecho de texto de um artigo, unidade básica de recuperação. Contém o embedding vetorial usado na busca semântica. |
| `Category` | `cat_name` → `name` | `name` | Área do Direito (ex: Direito Penal, Direito do Trabalho). Agrupa artigos por domínio. |
| `Norm` | `norm_id` → `id` | `id`, `name`, `year`, `aliases` | Lei, decreto, portaria ou outro ato normativo (ex: "Lei 13.709/2018 - LGPD"). O campo `aliases` armazena siglas e formas alternativas de referência. |
| `Institution` | `inst_name` → `name` | `name` | Órgão, tribunal ou entidade jurídica (ex: STF, ANPD, TST). |
| `Concept` | `concept_name` → `name` | `name`, `embedding` (1024 dim) | Conceito jurídico abstrato (ex: "dignidade da pessoa humana", "proteção de dados"). Possui embedding para busca semântica. |
| `Scholar` | `scholar_name` → `name` | `name` | Doutrinador ou jurista citado nos artigos (ex: "Norberto Bobbio"). |

---

## 3. Relações (Relationship Types)

| Tipo | Padrão | Propriedades | Descrição |
|---|---|---|---|
| `IN_CATEGORY` | `Article → Category` | — | Classifica o artigo em uma área do Direito. |
| `HAS_CHUNK` | `Article → Chunk` | — | Liga o artigo aos seus trechos de texto. |
| `MENTIONS` | `Chunk → Norm \| Institution \| Concept \| Scholar` | `evidence` (trecho literal) | Indica que o chunk menciona a entidade. A propriedade `evidence` guarda o trecho exato do texto que justifica a relação (extraído por NER/RE com LLM). |
| `CITES_NORM` | `Article → Norm` | `count` (frequência) | Artigo cita uma norma; `count` indica quantas vezes. |
| `CITES_SCHOLAR` | `Article → Scholar` | `count` (frequência) | Artigo cita um doutrinador. |
| `ABOUT` | `Article → Concept` | `weight` | Artigo trata de um conceito jurídico; `weight` indica relevância. |
| `REGULATES` | `Norm → Concept` | — | A norma regula ou disciplina o conceito. |
| `AMENDS` | `Norm → Norm` | — | A norma altera outra norma. |
| `REVOKES` | `Norm → Norm` | — | A norma revoga outra norma. |
| `ISSUED_BY` | `Norm → Institution` | — | A norma foi emitida pelo órgão. |
| `INTERPRETS` | `Institution → Norm` | — | O órgão (ex: tribunal) interpreta ou regulamenta a norma. |
| `SPECIALIZES` | `Concept → Concept` | — | Um conceito é mais específico que outro (hierarquia de conceitos). |

---

## 4. Constraints e Índices

### 4.1 Constraints de Unicidade

Garantem que não existam nós duplicados para a mesma entidade real.

| Constraint | Label | Propriedade |
|---|---|---|
| `article_id` | `Article` | `id` |
| `chunk_id` | `Chunk` | `id` |
| `cat_name` | `Category` | `name` |
| `norm_id` | `Norm` | `id` |
| `inst_name` | `Institution` | `name` |
| `concept_name` | `Concept` | `name` |
| `scholar_name` | `Scholar` | `name` |

### 4.2 Índice BTree

| Índice | Label | Propriedade | Finalidade |
|---|---|---|---|
| `norm_year` | `Norm` | `year` | Filtros e ordenações por ano de publicação da norma. |

### 4.3 Índices Fulltext

Permitem busca textual eficiente com análise linguística (tokenização, stemming).

| Índice | Labels | Propriedades | Finalidade |
|---|---|---|---|
| `ft_chunk` | `Chunk` | `text` | Busca por palavras-chave nos trechos de texto dos artigos. |
| `ft_entities` | `Norm`, `Concept`, `Institution` | `name`, `aliases` | Ligação de entidades da pergunta do usuário a nós do grafo; suporta siglas e apelidos (ex: "LGPD" → Norm com id `norm:lei-13709-2018`). |

### 4.4 Índices Vetoriais

Calculados com o modelo `BAAI/bge-m3` (1024 dimensões), similaridade por cosseno. Devem ser aplicados após a geração dos embeddings (`scripts/embed.py`).

| Índice | Label | Propriedade | Dimensões | Similaridade |
|---|---|---|---|---|
| `vec_chunk` | `Chunk` | `embedding` | 1024 | Cosseno |
| `vec_concept` | `Concept` | `embedding` | 1024 | Cosseno |

> Para usar `text-embedding-3-small` (OpenAI), ajuste `vector.dimensions` para 1536 em `schema_vector.cypher` e recalcule os embeddings.

---

## 5. Diagrama Simplificado

```
(Category)
    ▲
    │ IN_CATEGORY
    │
(Article) ──HAS_CHUNK──► (Chunk) ──MENTIONS──► (Norm)
    │                                │               │
    │ CITES_NORM                     │               ├─ REGULATES ──► (Concept)
    │ CITES_SCHOLAR                  │               ├─ AMENDS ──────► (Norm)
    │ ABOUT                          │               ├─ REVOKES ─────► (Norm)
    ▼                                │               └─ ISSUED_BY ───► (Institution)
 (Norm)                              │
 (Scholar)                           ├─ MENTIONS ──► (Institution)
 (Concept)                           ├─ MENTIONS ──► (Concept) ──SPECIALIZES──► (Concept)
                                     └─ MENTIONS ──► (Scholar)

(Institution) ──INTERPRETS──► (Norm)
```

---

## 6. Decisões de Design

**Por que LPG e não RDF/triple store?** O Neo4j LPG permite armazenar propriedades diretamente nos nós e relações (ex: `evidence` na relação `MENTIONS`, `pagerank` e `community` nos nós de entidade), o que simplifica as queries Cypher de recuperação híbrida sem joins adicionais.

**Por que `aliases` em `Norm`?** Normas jurídicas são referenciadas de múltiplas formas nos textos ("LGPD", "Lei Geral de Proteção de Dados", "Lei 13.709"). O campo `aliases` (indexado em `ft_entities`) permite que a ligação de entidades da pergunta encontre o nó correto independente da forma usada.

**Por que embeddings em `Chunk` e `Concept`, mas não em `Norm` ou `Institution`?** Chunks são a unidade de recuperação semântica — faz sentido buscá-los por similaridade com a pergunta. Concepts têm semântica difusa o suficiente para se beneficiar de busca vetorial. Norms e Institutions têm identificadores canônicos (nome exato, sigla, número da lei) mais bem servidos por busca fulltext e BTree.

**PageRank e Comunidades Louvain** são calculados em Python (networkx) sobre o subgrafo de entidades e gravados como propriedades (`pagerank`, `community`) nos nós, sem depender do plugin GDS. Isso permite rodar no Neo4j Aura Free.