"""
WEB — chatovací okno pro RAG appku v prohlížeči (přes Streamlit).

Dělá to samé co ask.py, jen hezky v prohlížeči: napíšeš otázku do políčka
(nebo klikneš na ukázkovou otázku) a appka odpoví z tvých poznámek.
Pod každou odpovědí je rozklikávací sekce "📄 Zdroje" s přesnými pasážemi.

Spuštění:  ./.venv/bin/streamlit run web.py
"""

import json
import os
from pathlib import Path

import streamlit as st

# V cloudu (Streamlit Community Cloud) je API klíč uložený v "Secrets".
# Načteme ho do prostředí JEŠTĚ PŘED importem ask.py (ask.py si při importu
# vytváří OpenAI klienta, takže klíč musí být nastavený dřív).
try:
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass  # lokálně žádné secrets nejsou → klíč se vezme z .env (řeší ask.py)

# Použijeme stejnou logiku jako ask.py — neopakujeme kód.
from ask import retrieve, answer, SIMILARITY_THRESHOLD

INDEX_FILE = Path("index.json")

# Ukázkové otázky, co se nabídnou, když appku někdo otevře poprvé.
EXAMPLE_QUESTIONS = [
    "Komu chci pomáhat?",
    "Jaký je můj dlouhodobý cíl?",
    "Kdo mě inspiruje?",
]

# --- Nastavení stránky (název v záložce + nadpis) ---
st.set_page_config(page_title="Chat s mými poznámkami", page_icon="🧠")
st.title("🧠 Chat s mými poznámkami")
st.caption("Ptej se na cokoliv ze svých poznámek. Odpovídám jen z nich a uvádím zdroj.")


# --- Načtení indexu (jen jednou, díky cache) ---
@st.cache_data
def load_index() -> list[dict]:
    return json.loads(INDEX_FILE.read_text(encoding="utf-8"))


def show_sources(sources: list[dict]) -> None:
    """Vykreslí rozklikávací sekci se zdroji (odkud odpověď pochází)."""
    with st.expander("📄 Zdroje — odkud to je"):
        for s in sources:
            st.markdown(f"**{s['source']}** · shoda {s['score']:.2f}")
            st.caption(s["text"])


def handle_question(question: str) -> None:
    """Zpracuje otázku: najde kousky, rozhodne o odpovědi a uloží do historie."""
    st.session_state.messages.append({"role": "user", "content": question})

    chunks = retrieve(question, index)
    best_score = chunks[0]["score"]

    if best_score < SIMILARITY_THRESHOLD:
        reply = (
            "Na tohle nemám ve svých poznámkách dost relevantní informace, "
            "tak si radši nic nevymýšlím.\n\n"
            f"*(nejlepší shoda jen {best_score:.2f}, práh je {SIMILARITY_THRESHOLD})*"
        )
        sources = None
    else:
        reply = answer(question, chunks)
        sources = [
            {"source": c["source"], "score": c["score"], "text": c["text"]}
            for c in chunks
        ]

    st.session_state.messages.append(
        {"role": "assistant", "content": reply, "sources": sources}
    )


if not INDEX_FILE.exists():
    st.error("Index neexistuje. Nejdřív spusť v terminálu:  ./.venv/bin/python ingest.py")
    st.stop()

index = load_index()

# --- Paměť konverzace (ať zůstane historie zpráv na obrazovce) ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Vykreslíme dosavadní zprávy (včetně zdrojů, pokud u zprávy jsou).
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            show_sources(msg["sources"])

# Otázku můžeme dostat dvěma způsoby: z políčka, nebo z ukázkového tlačítka.
pending_question = st.chat_input("Zeptej se svých poznámek…")

# Ukázková tlačítka ukážeme jen na začátku (když ještě není žádná zpráva).
if not st.session_state.messages:
    st.markdown("**Zkus třeba:**")
    columns = st.columns(len(EXAMPLE_QUESTIONS))
    for column, example in zip(columns, EXAMPLE_QUESTIONS):
        if column.button(example):
            pending_question = example

# Pokud přišla otázka (z políčka nebo z tlačítka), zpracujeme ji.
if pending_question:
    with st.spinner("Hledám v poznámkách…"):
        handle_question(pending_question)
    st.rerun()  # překreslíme stránku, ať se nová zpráva hezky zobrazí v historii
