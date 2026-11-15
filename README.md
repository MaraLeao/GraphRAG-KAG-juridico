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
| `app/streamlit_app.py` | Interface web (Streamlit): ponto de entrada fino sobre o pacote |
| `.streamlit/config.toml` | Tema escuro da interface (mesmas cores do grafo) |
| `tests/` | Testes automatizados (`python -m pytest`), inclusive da interface em modo demonstração |

### Pacote `graphrag_juridico`

| Módulo | Papel |
|---|---|
| `config.py` | Lê o `.env`; cria o driver Neo4j (`get_driver`) e o cliente LLM (`get_llm`, OpenRouter por padrão, trocável via `LLM_BASE_URL`) |
| `db.py` | `read()`: executa Cypher em transação **somente leitura** (o Neo4j recusa escrita) |
| `contracts.py` | Contrato `AskResult` devolvido por `ask()`, consumido pela UI e pela avaliação |
| `ask.py` | Orquestrador `ask(question)`: recuperação multi-hop + Text-to-Cypher + resposta do LLM citando `[Cn]`, `[Pn]` e `[Q]`, com verificação das citações; `ask(..., mock=True)` ou `GRAPHRAG_MOCK=1` devolve dados de exemplo |
| `mock.py` | Modo demonstração: reproduz execuções reais gravadas em `resources/demo_answers.json` (sem banco nem LLM) |
| `graph_view.py` | Visualizações interativas (vis-network): mapa do conhecimento, subgrafo da resposta e rota da resposta |
| `baselines.py` | Métodos de comparação: LLM puro (sem contexto) e RAG vetorial (mesmo prompt do GraphRAG, só busca vetorial) |
| `metrics.py` | Métricas automáticas de uma resposta: citações válidas, frases com fonte, normas sem suporte, abstenção e acerto |
| `evaluation.py` | Avaliação comparativa no conjunto com gabarito (`resources/evaluation_questions.json`), relatório e gráfico |
| `algorithms.py` | PageRank e Louvain (networkx) sobre o grafo de entidades, gravados nos nós como `pagerank` e `community` |
| `retrieval.py` | Busca híbrida de trechos: vetor (`vec_chunk`, bge-m3 local) + fulltext (`ft_chunk`) + trechos das entidades, fundidos por RRF |
| `entities.py` | Ligação das entidades da pergunta: apelidos de normas ("LGPD"), siglas ("STF"), vetor (`vec_concept`) e fulltext |
| `multihop.py` | Shortest Path entre as entidades, expansão de 2 saltos priorizada por comunidade e PageRank, subgrafo e confiança |
| `text_to_cypher.py` | Cypher gerada pelo LLM, validada (somente leitura, schema, `LIMIT`), conferida com `EXPLAIN` e com 1 nova tentativa |

#### Algoritmos de grafo

```bash
python -m graphrag_juridico.algorithms           # calcula e mostra as entidades mais centrais e as comunidades
python -m graphrag_juridico.algorithms --write   # calcula e grava pagerank/community no banco do .env
```

Rodam em Python (networkx) para funcionar também no Neo4j Aura Free, que não tem o plugin GDS.
O `--write` altera o banco: no Aura (compartilhado), combine com o grupo antes.

#### Recuperação multi-hop

```bash
python -m graphrag_juridico.multihop "Qual a relação entre a LGPD e a dignidade da pessoa humana?"
python -m graphrag_juridico.multihop "pergunta" --no-communities --no-pagerank   # compara sem os algoritmos
```

Mostra as entidades ligadas, os caminhos no grafo, os trechos escolhidos (`[C1]...`) e a confiança. Não usa o LLM.
Na primeira execução baixa o modelo bge-m3 (~2 GB). Entidades citadas por mais de 300 trechos (como a CF/88) não
servem de ponte nos caminhos. Requer `pagerank`/`community` gravados para priorizar a expansão.

#### Text-to-Cypher

```bash
python -m graphrag_juridico.text_to_cypher "Quais instituições interpretam normas que regulam o consentimento?"
```

Usa o `CYPHER_MODEL` do `.env` (uma requisição ao LLM por tentativa). Consultas com escrita, labels ou relações fora
do schema e caminhos sem limite são recusadas antes de chegar ao banco; a execução é sempre em transação somente leitura.

#### Perguntas (pipeline completo)

```bash
python -m graphrag_juridico.ask "Qual a relação entre a LGPD e a dignidade da pessoa humana?"
python -m graphrag_juridico.ask "pergunta" --no-cypher    # sem a Cypher do LLM (uma requisição a menos)
```

```python
from graphrag_juridico.ask import ask
from graphrag_juridico.multihop import Options

result = ask("Quais instituições interpretam a LGPD?", options=Options(use_pagerank=False))
result.answer, result.chunks, result.paths, result.cypher, result.confidence
```

O LLM (`ANSWER_MODEL`) recebe só o contexto recuperado: trechos `[C1]...`, caminhos do grafo `[P1]...` e o resultado
da Cypher `[Q]`. Ele deve citar cada afirmação e, se o contexto não bastar, responder "Não há informação suficiente no
corpus". As citações são conferidas: citar algo fora do contexto reduz o componente `citations` da confiança.
Cada pergunta usa 2 a 3 requisições ao LLM.

#### Interface (Streamlit)

```bash
streamlit run app/streamlit_app.py      # rode da raiz do repositório; abre em http://localhost:8501
```

- **Perguntar:** resposta com as citações destacadas (trechos `[C]`, caminhos `[P]`, consulta `[Q]`), aviso de
  abstenção, nível de confiança com seus componentes e o **subgrafo** recuperado. Abaixo: trechos com link para o
  artigo, a **rota da resposta** (pergunta → entidades → nós percorridos → resposta, só com o que foi citado),
  a Cypher executada, os nós e o JSON completo.
- **Mapa do conhecimento:** as entidades mais centrais do grafo (PageRank), coloridas pelas comunidades do Louvain,
  com busca, filtro por comunidade e vizinhos do nó clicado.
- **Barra lateral:** liga/desliga Shortest Path, PageRank, Louvain e a Cypher do LLM, para comparar o efeito de cada
  um. A pergunta só é enviada com Enter (ou clicando num exemplo): mudar as opções não gasta requisições.
- **Comparar métodos** (barra lateral, +2 requisições): responde também com o LLM puro e com o RAG vetorial e mostra
  as três respostas lado a lado com as métricas automáticas, tabela e gráfico.
- **Avaliação:** o resultado consolidado do conjunto de perguntas com gabarito (ver abaixo).
- **Modo demonstração** (barra lateral ou `GRAPHRAG_MOCK=1`): não usa o Neo4j nem o LLM. As perguntas de exemplo
  reproduzem execuções reais gravadas, úteis para apresentar sem internet. Para regravar:
  `python -m graphrag_juridico.mock --record` (2 a 3 requisições por pergunta).

#### Avaliação (LLM puro x RAG vetorial x GraphRAG/KAG)

```bash
python -m graphrag_juridico.evaluation status   # o que já foi gravado
python -m graphrag_juridico.evaluation run      # roda o que falta (~45 requisições ao LLM; lê o banco)
python -m graphrag_juridico.evaluation report   # gera evaluation/report.md e evaluation/metrics.png
```

Dez perguntas com gabarito: estruturadas (gabarito vindo de Cypher de referência), multi-hop e armadilhas que o
corpus não responde (inclusive uma lei inexistente). Os três métodos usam o mesmo LLM; o RAG vetorial e o GraphRAG,
o mesmo prompt e as mesmas regras de citação. As métricas (`metrics.py`) não usam LLM: acerto pelo gabarito,
abstenção correta nas armadilhas, abstenção indevida, citações válidas, frases com fonte e **normas sem suporte**
(leis citadas que não estão no contexto, indício de alucinação). Cada resposta é gravada assim que chega: se a cota
acabar, rodar de novo completa só o que falta. As respostas gravadas também alimentam a comparação no modo
demonstração da interface.

**Cota do LLM:** o plano gratuito do OpenRouter permite 50 requisições por dia **por conta**. Com uma
`FALLBACK_OPENROUTER_KEY` (de outra conta) no `.env`, as chamadas passam para ela automaticamente quando a principal
recebe 429/402.

O visual dos grafos (fundo escuro, layout ForceAtlas2, cores por comunidade, barra lateral com busca e legenda) é
inspirado no [Graphify](https://github.com/Graphify-Labs/graphify); o template é próprio.

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
- [networkx](https://networkx.org/) (PageRank e Louvain).
- Modelo de embedding [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) via `sentence-transformers`.
- LLMs via [OpenRouter](https://openrouter.ai/) (modelos definidos em `CYPHER_MODEL` e `ANSWER_MODEL` no `.env`).
- [Streamlit](https://streamlit.io/) (interface) e [vis-network](https://visjs.github.io/vis-network/) (grafos interativos).
- Visual dos grafos inspirado no [Graphify](https://github.com/Graphify-Labs/graphify) (Apache-2.0); nenhum código copiado.
- Parte do código foi desenvolvida com apoio de assistente de IA (Claude Code), com revisão do grupo.
