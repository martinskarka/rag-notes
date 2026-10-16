# RAG Notes — Chat with your own notes

A small Retrieval-Augmented Generation (RAG) app in Python. You drop your notes
into a folder, and then ask questions about them in natural language. The app
answers **using only your notes** and **cites the source file** — so there are
no hallucinations and every answer is traceable.

Built as a hands-on learning project to understand how RAG actually works under
the hood (no heavy frameworks — just the OpenAI API and numpy).

## How it works

```
1. INGEST    notes → split into chunks → embed each chunk → store in index.json
2. RETRIEVE  your question → embed it → find the most similar chunks (cosine similarity)
3. AUGMENT   attach the retrieved chunks to the question as context
4. GENERATE  ask the model to answer ONLY from that context, and cite the source
```

- `ingest.py` — step 1 (build the index)
- `ask.py` — steps 2–4 (retrieve + answer)

## Tech

- Python 3
- OpenAI API — `text-embedding-3-small` (embeddings) + `gpt-4o-mini` (answers)
- numpy — cosine similarity

## Setup

```bash
# 1. Create a virtual environment and install dependencies
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt

# 2. Add your OpenAI API key
cp .env.example .env
# then edit .env and paste your key

# 3. Put some .md or .txt notes into the notes/ folder
```

## Usage

```bash
# Build the index from your notes
./.venv/bin/python ingest.py

# Ask a question
./.venv/bin/python ask.py "What is my mission and who do I want to help?"
```

The app prints the most relevant chunks (with similarity scores) and then an
answer grounded in your notes, with the source file cited.

## Example

```
--- Answer ---
Your mission is to help people orient themselves in the age of AI ...
(Source: moje-vize.md)
```
