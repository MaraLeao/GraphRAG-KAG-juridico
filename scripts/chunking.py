"""Divisão de artigos em chunks (usada na carga estrutural e na extração)."""
import re

CHUNK_WORDS = 350  # ~500 tokens em português
SECTIONS = ["INTRODUÇÃO", "DESENVOLVIMENTO", "METODOLOGIA", "RESULTADOS", "CONSIDERAÇÕES FINAIS", "CONCLUSÃO", "REFERÊNCIAS"]
SECTION_RX = re.compile("|".join(re.escape(s) for s in SECTIONS))


def chunk_article(article_id: str, text: str):
    """Janelas de CHUNK_WORDS palavras; 'section' = último cabeçalho conhecido visto até o início da janela."""
    words = text.split()
    section, chunks = None, []
    for order, start in enumerate(range(0, len(words), CHUNK_WORDS)):
        piece = " ".join(words[start:start + CHUNK_WORDS])
        found = SECTION_RX.findall(piece)
        if found:
            section = found[-1]
        chunks.append({"id": f"{article_id}:{order}", "order": order, "section": section, "text": piece})
    return chunks
