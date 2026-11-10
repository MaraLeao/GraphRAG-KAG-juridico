// Constraints (unicidade) -- modelo LPG do plano_grafo_neo4j.md, seção 2.3
CREATE CONSTRAINT article_id    IF NOT EXISTS FOR (a:Article)     REQUIRE a.id IS UNIQUE;
CREATE CONSTRAINT chunk_id      IF NOT EXISTS FOR (c:Chunk)       REQUIRE c.id IS UNIQUE;
CREATE CONSTRAINT cat_name      IF NOT EXISTS FOR (c:Category)    REQUIRE c.name IS UNIQUE;
CREATE CONSTRAINT norm_id       IF NOT EXISTS FOR (n:Norm)        REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT inst_name     IF NOT EXISTS FOR (i:Institution) REQUIRE i.name IS UNIQUE;
CREATE CONSTRAINT concept_name  IF NOT EXISTS FOR (c:Concept)     REQUIRE c.name IS UNIQUE;
CREATE CONSTRAINT scholar_name  IF NOT EXISTS FOR (s:Scholar)     REQUIRE s.name IS UNIQUE;

// Índice BTree
CREATE INDEX norm_year IF NOT EXISTS FOR (n:Norm) ON (n.year);

// Índices fulltext
CREATE FULLTEXT INDEX ft_chunk IF NOT EXISTS FOR (c:Chunk) ON EACH [c.text];
CREATE FULLTEXT INDEX ft_entities IF NOT EXISTS FOR (n:Norm|Concept|Institution) ON EACH [n.name, n.aliases];
