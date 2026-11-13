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
| `pyproject.toml` | Empacotamento do código em `src/` e configuração do `pytest` |
| `src/graphrag_juridico/` | Sistema de perguntas (ver abaixo) |
| `tests/` | Testes automatizados (`python -m pytest`) |

### Pacote `graphrag_juridico`

| Módulo | Papel |
|---|---|
| `config.py` | Lê o `.env`; cria o driver Neo4j (`get_driver`) e o cliente LLM (`get_llm`, OpenRouter por padrão, trocável via `LLM_BASE_URL`) |
| `db.py` | `read()`: executa Cypher em transação **somente leitura** (o Neo4j recusa escrita) |
| `contracts.py` | Contrato `AskResult` devolvido por `ask()`, consumido pela UI e pela avaliação |
| `ask.py` | Orquestrador `ask(question)`; `ask(..., mock=True)` ou `GRAPHRAG_MOCK=1` devolve dados de exemplo |
| `mock.py` | Resposta de exemplo no formato do contrato |

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
pip install -e .         # instala o pacote src/graphrag_juridico em modo editável
cp .env.example .env     # edite com suas credenciais (nunca faça commit do .env)
python -m pytest         # roda os testes
```

> No WSL, se `python -m venv` falhar por falta do `ensurepip`, use o [uv](https://docs.astral.sh/uv/):
> `uv venv --python 3.12 .venv` e troque `pip install` por `uv pip install`. Para um PyTorch menor (só CPU),
> rode antes `uv pip install torch --index-url https://download.pytorch.org/whl/cpu`.

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

### 3.2 Carregar o banco pronto (dump)

Para não repetir a extração por LLM e os embeddings, o grafo completo (artigos, chunks, entidades, relações e
embeddings) pode ser restaurado de um dump do Neo4j (`neo4j.dump`, ~318 MB, compartilhado fora do Git porque o
GitHub limita arquivos a 100 MB). Coloque o arquivo numa pasta, por exemplo `./dump/`, e rode:

```bash
# 1. restaurar o dump num volume novo (o container do Neo4j não pode estar usando o volume)
docker run --rm -v neo4j_kag_data:/data -v "$PWD/dump":/backups neo4j:latest \
  neo4j-admin database load neo4j --from-path=/backups --overwrite-destination=true

# 2. subir o Neo4j com esse volume
docker run -d --name neo4j-kag -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/SUA_SENHA_AQUI -e 'NEO4J_PLUGINS=["graph-data-science"]' \
  -v neo4j_kag_data:/data neo4j:latest
```

O dump contém o texto dos artigos do Âmbito Jurídico: não publique em repositório aberto sem conferir a licença do
dataset. Ele não inclui senhas nem chaves. Para gerar um dump novo: pare o container e rode
`neo4j-admin database dump neo4j --to-path=/backups` num container temporário com o mesmo volume.

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
