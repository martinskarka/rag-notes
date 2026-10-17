"""
ASK — kroky 2, 3 a 4 RAG pipeline.

2) RETRIEVE: tvoji otázku převede na embedding a najde v index.json
             nejpodobnější kousky poznámek (cosine similarity).
3) AUGMENT:  nalezené kousky přilepí k otázce jako kontext.
4) GENERATE: pošle to modelu s instrukcí "odpověz JEN podle kontextu
             a uveď zdroj" a vypíše odpověď i s citacemi.

Spuštění:  ./.venv/bin/python ask.py "Jaké háčky mi fungují?"
      nebo: ./.venv/bin/python ask.py        (zeptá se interaktivně)
"""

import json
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

EMBEDDING_MODEL = "text-embedding-3-small"  # musí být stejný jako v ingest.py!
CHAT_MODEL = "gpt-4o-mini"                   # levný a chytrý na odpovídání
INDEX_FILE = Path("index.json")
TOP_K = 3                                    # kolik nejlepších kousků použít

# Práh: když je i nejlepší shoda pod touhle hranicí, odpověď v poznámkách
# nejspíš není → rovnou řekneme "nevím" a ušetříme volání modelu.
SIMILARITY_THRESHOLD = 0.45


def embed_query(text: str) -> np.ndarray:
    """Převede otázku na embedding (jako numpy pole kvůli výpočtu podobnosti)."""
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=[text])
    return np.array(response.data[0].embedding)


def cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Spočítá podobnost otázky ke všem kouskům naráz.

    Cosine similarity = úhel mezi vektory. 1.0 = stejný význam, 0 = nesouvisí.
    """
    query_norm = query / np.linalg.norm(query)
    matrix_norm = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix_norm @ query_norm


def retrieve(question: str, index: list[dict]) -> list[dict]:
    """Najde TOP_K kousků nejbližších otázce."""
    question_vec = embed_query(question)
    matrix = np.array([item["embedding"] for item in index])
    scores = cosine_similarity(question_vec, matrix)
    # Seřadíme od nejpodobnějšího a vezmeme nejlepších TOP_K.
    best = np.argsort(scores)[::-1][:TOP_K]
    return [{**index[i], "score": float(scores[i])} for i in best]


def answer(question: str, chunks: list[dict]) -> str:
    """Pošle otázku + nalezený kontext modelu a vrátí odpověď."""
    # Poskládáme kontext: každý kousek označíme jeho zdrojem kvůli citaci.
    context = "\n\n".join(
        f"[Zdroj: {c['source']}]\n{c['text']}" for c in chunks
    )

    system_prompt = (
        "Jsi asistent, který odpovídá na otázky POUZE na základě poskytnutých "
        "poznámek. Pokud odpověď v poznámkách není, řekni to a nic si nevymýšlej. "
        "Na konci odpovědi uveď, ze kterých zdrojů (názvů souborů) jsi čerpal."
    )
    user_prompt = f"Poznámky:\n{context}\n\nOtázka: {question}"

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content


def main() -> None:
    if not INDEX_FILE.exists():
        print("Index neexistuje. Nejdřív spusť:  ./.venv/bin/python ingest.py")
        return

    index = json.loads(INDEX_FILE.read_text(encoding="utf-8"))

    # Otázka z příkazové řádky, nebo se zeptáme interaktivně.
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = input("Na co se chceš zeptat svých poznámek? ")

    chunks = retrieve(question, index)

    print("\n--- Nalezené kousky (seřazené podle podobnosti) ---")
    for c in chunks:
        print(f"  [{c['score']:.2f}] {c['source']}: {c['text'][:70]}...")

    print("\n--- Odpověď ---")

    # PRÁH: když je i nejlepší kousek málo podobný, odpověď v poznámkách není.
    best_score = chunks[0]["score"]
    if best_score < SIMILARITY_THRESHOLD:
        print(
            "Na tohle nemám ve svých poznámkách dost relevantní informace, "
            "tak si radši nic nevymýšlím."
        )
        print(f"(nejlepší shoda byla jen {best_score:.2f}, práh je {SIMILARITY_THRESHOLD})")
        return  # model vůbec nevoláme → šetříme peníze i čas

    print(answer(question, chunks))


if __name__ == "__main__":
    main()
