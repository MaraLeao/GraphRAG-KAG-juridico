# GraphRAG/KAG sobre artigos jurídicos (Neo4j)

Projeto final de *Tendências em Ciência da Computação* (Bacharelado em Ciência da Computação): um sistema de
recuperação aumentada por grafo de conhecimento (GraphRAG/KAG) que combina um banco de grafos (Neo4j) com LLMs.

## Dataset

[`celsowm/ambito_juridico_artigos`](https://huggingface.co/datasets/celsowm/ambito_juridico_artigos) (Hugging Face):
11.373 artigos do Âmbito Jurídico em 17 áreas do Direito, com `titulo`, `resumo`, `categoria`, `texto` e `link`.
O notebook de análise remove duplicatas (resta ~10,6 mil artigos únicos) e define um subconjunto estratificado por
categoria (~1.000 artigos) para a construção do grafo. Confira a licença do dataset antes de redistribuir trechos dos textos.

## Conteúdo do repositório

| Arquivo | Descrição |
|---|---|
| `analise_ambito_juridico.ipynb` | Análise exploratória: escopo, duplicatas, tamanho dos textos, frequência de normas/instituições/conceitos, co-ocorrência e amostragem |
| `schema.cypher` | Constraints de unicidade, índice BTree e índices fulltext |
| `schema_vector.cypher` | Índices vetoriais (rodar só depois de escolher o modelo de embedding; ajuste `vector.dimensions`) |
| `.env.example` | Variáveis de ambiente esperadas (copie para `.env`, que **não** vai para o Git) |
| `requirements.txt` | Dependências Python |

## Modelo do grafo (LPG)

Nomes do schema em inglês; valores e conteúdo em português.

**Nós:** `Article`, `Chunk`, `Category`, `Norm`, `Institution`, `Concept`, `Scholar`.

**Relações:**

| Relação | Padrão |
|---|---|
| `IN_CATEGORY` | `Article → Category` |
| `HAS_CHUNK` | `Article → Chunk` |
| `MENTIONS` | `Chunk → Norm \| Institution \| Concept \| Scholar` (propriedade `evidence`) |
| `CITES_NORM`, `CITES_SCHOLAR`, `ABOUT` | `Article → Norm \| Scholar \| Concept` (com contagem/peso) |
| `REGULATES` | `Norm → Concept` |
| `AMENDS`, `REVOKES` | `Norm → Norm` |
| `ISSUED_BY` | `Norm → Institution` |
| `INTERPRETS` | `Institution → Norm` |
| `SPECIALIZES` | `Concept → Concept` |

Entidades e relações são extraídas por LLM (NER/RE) com exigência de trecho literal de evidência, validadas
contra o texto do chunk e unificadas por resolução de entidades (chaves canônicas para normas, instituições e conceitos).

## Como reproduzir

### Requisitos

- Python 3.11+
- Docker (ou uma instalação do Neo4j **5.11 ou superior**, necessário para índices vetoriais)
- Opcional: plugin **Graph Data Science** (PageRank, Louvain)

### 1. Ambiente Python

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env     # edite com suas credenciais (nunca faça commit do .env)
```

### 2. Neo4j com Docker

```bash
docker run -d --name neo4j-kag \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/SUA_SENHA_AQUI \
  -e 'NEO4J_PLUGINS=["graph-data-science"]' \
  -v neo4j_kag_data:/data \
  neo4j:latest
```

A senha precisa ter 8 caracteres ou mais e deve ser a mesma do `.env`. O Neo4j Browser fica em http://localhost:7474.

### 3. Criar o schema

```bash
docker exec -i neo4j-kag cypher-shell -u neo4j -p SUA_SENHA_AQUI < schema.cypher
```

Os índices vetoriais ficam em `schema_vector.cypher`; aplique-os do mesmo modo, depois de ajustar a dimensão do embedding
(1024 para `BAAI/bge-m3`, 1536 para `text-embedding-3-small`).

### 3.1 Embeddings e busca GraphRAG

```bash
python scripts/embed.py                                  # embeddings de Chunk e Concept (baai/bge-m3, 1024 dim)
python scripts/apply_schema.py schema_vector.cypher      # índices vetoriais
python scripts/graphrag.py "Qual a relação entre a LGPD e a dignidade da pessoa humana?"
```

`graphrag.py` funde busca vetorial e fulltext (RRF), expande no grafo (entidades citadas e relações
`REGULATES`/`AMENDS`/`INTERPRETS`/... ignorando super-nós como a CF/88) e gera a resposta citando trechos e relações.
Use `--no-answer` para ver só a recuperação.

### 4. Análise exploratória

Abra `analise_ambito_juridico.ipynb` e execute todas as células. Ele baixa o dataset do Hugging Face e gera
`processed_ambito_juridico.parquet`, `piloto_ambito_juridico.parquet` e `subconjunto_ambito_juridico.parquet`
(arquivos `.parquet` não são versionados).

## Segurança

- Chaves e senhas ficam no `.env`, ignorado pelo Git. Use `.env.example` como modelo.
- Não exponha a porta 7687 do Neo4j na internet; para compartilhar o banco, use um usuário somente leitura.

## Créditos

- Dataset: [celsowm/ambito_juridico_artigos](https://huggingface.co/datasets/celsowm/ambito_juridico_artigos).
- Neo4j, Neo4j GDS e o SDK `neo4j` para Python.
