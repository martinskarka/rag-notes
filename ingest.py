"""
INGEST — krok 1 RAG pipeline.

Načte všechny poznámky ze složky notes/, rozseká je na malé kousky (chunks),
pro každý kousek vytvoří embedding (vektor čísel vyjadřující význam) a všechno
uloží do index.json. Tenhle index pak používá ask.py při odpovídání.

Spuštění:  ./.venv/bin/python ingest.py
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Načte OPENAI_API_KEY ze souboru .env do prostředí.
load_dotenv()

# Vytvoří klienta. Automaticky si vezme klíč z OPENAI_API_KEY.
client = OpenAI()

# Model na embeddingy. "small" je levný a pro tohle úplně stačí.
EMBEDDING_MODEL = "text-embedding-3-small"

# Složka s poznámkami a kam uložíme hotový index.
NOTES_DIR = Path("notes")
INDEX_FILE = Path("index.json")

# Jak velké mají být kousky (ve znacích) a jak moc se mají překrývat.
# Překryv zajistí, že se nám myšlenka nerozpůlí přesně na hranici kousku.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def chunk_text(text: str) -> list[str]:
    """Rozseká dlouhý text na překrývající se kousky po CHUNK_SIZE znacích."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        # Posuneme se dopředu, ale o kousek zpět kvůli překryvu.
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def embed(texts: list[str]) -> list[list[float]]:
    """Pošle seznam textů do OpenAI a vrátí jejich embeddingy."""
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    # response.data je ve stejném pořadí jako vstup.
    return [item.embedding for item in response.data]


def main() -> None:
    # 1) Posbíráme všechny kousky ze všech souborů + zapamatujeme si zdroj.
    all_chunks: list[str] = []
    sources: list[str] = []

    note_files = sorted(NOTES_DIR.glob("*.md")) + sorted(NOTES_DIR.glob("*.txt"))
    if not note_files:
        print(f"V složce {NOTES_DIR}/ nejsou žádné poznámky (.md nebo .txt).")
        return

    for file in note_files:
        text = file.read_text(encoding="utf-8")
        for chunk in chunk_text(text):
            all_chunks.append(chunk)
            sources.append(file.name)  # odkud kousek pochází → citace

    print(f"Načteno {len(note_files)} souborů, rozsekáno na {len(all_chunks)} kousků.")

    # 2) Uděláme embedding pro všechny kousky najednou (jedno volání API).
    print("Volám OpenAI pro embeddingy...")
    vectors = embed(all_chunks)

    # 3) Uložíme index: pro každý kousek jeho text, zdroj a embedding.
    index = [
        {"text": chunk, "source": source, "embedding": vector}
        for chunk, source, vector in zip(all_chunks, sources, vectors)
    ]
    INDEX_FILE.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    print(f"Hotovo! Index uložen do {INDEX_FILE} ({len(index)} kousků).")


if __name__ == "__main__":
    main()
