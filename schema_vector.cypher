// Índices vetoriais: rodar só depois de escolher o modelo de embedding.
// Ajustar vector.dimensions (bge-m3 = 1024, text-embedding-3-small = 1536).
CREATE VECTOR INDEX vec_chunk IF NOT EXISTS FOR (c:Chunk) ON (c.embedding)
  OPTIONS {indexConfig: {`vector.dimensions`: 1024, `vector.similarity_function`: 'cosine'}};
CREATE VECTOR INDEX vec_concept IF NOT EXISTS FOR (c:Concept) ON (c.embedding)
  OPTIONS {indexConfig: {`vector.dimensions`: 1024, `vector.similarity_function`: 'cosine'}};
