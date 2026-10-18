"""
WEB — chatovací okno pro RAG appku v prohlížeči (přes Streamlit).

Dělá to samé co ask.py, jen hezky v prohlížeči: napíšeš otázku do políčka
a appka odpoví z tvých poznámek (nebo řekne "nevím", když to tam není).

Spuštění:  ./.venv/bin/streamlit run web.py
"""

import json
from pathlib import Path

import streamlit as st

# Použijeme stejnou logiku jako ask.py — neopakujeme kód.
from ask import retrieve, answer, SIMILARITY_THRESHOLD

INDEX_FILE = Path("index.json")

# --- Nastavení stránky (název v záložce + nadpis) ---
st.set_page_config(page_title="Chat s mými poznámkami", page_icon="🧠")
st.title("🧠 Chat s mými poznámkami")
st.caption("Ptej se na cokoliv ze svých poznámek. Odpovídám jen z nich a uvádím zdroj.")


# --- Načtení indexu (jen jednou, díky cache) ---
@st.cache_data
def load_index() -> list[dict]:
    return json.loads(INDEX_FILE.read_text(encoding="utf-8"))


if not INDEX_FILE.exists():
    st.error("Index neexistuje. Nejdřív spusť v terminálu:  ./.venv/bin/python ingest.py")
    st.stop()

index = load_index()

# --- Paměť konverzace (ať zůstane historie zpráv na obrazovce) ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Vykreslíme dosavadní zprávy.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Políčko na otázku dole ---
question = st.chat_input("Zeptej se svých poznámek…")

if question:
    # 1) Ukážeme otázku uživatele.
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # 2) Spočítáme a ukážeme odpověď.
    with st.chat_message("assistant"):
        with st.spinner("Hledám v poznámkách…"):
            chunks = retrieve(question, index)
            best_score = chunks[0]["score"]

            if best_score < SIMILARITY_THRESHOLD:
                reply = (
                    "Na tohle nemám ve svých poznámkách dost relevantní informace, "
                    "tak si radši nic nevymýšlím.\n\n"
                    f"*(nejlepší shoda jen {best_score:.2f}, práh je {SIMILARITY_THRESHOLD})*"
                )
            else:
                reply = answer(question, chunks)

        st.markdown(reply)

    # Uložíme odpověď do historie.
    st.session_state.messages.append({"role": "assistant", "content": reply})
