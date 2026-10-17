"""
EVALS — automatické "vysvědčení" pro RAG appku.

Projde sadu testovacích otázek, u kterých dopředu víme, jak se má appka
zachovat, a zkontroluje, jestli to tak opravdu udělala. Na konci vypíše,
kolik testů prošlo.

Spuštění:  ./.venv/bin/python evals.py
"""

import json
from pathlib import Path

# Použijeme funkce a práh přímo z naší appky (ask.py) — neopakujeme kód.
from ask import retrieve, answer, SIMILARITY_THRESHOLD

# Načteme stejný index, jako používá appka.
INDEX = json.loads(Path("index.json").read_text(encoding="utf-8"))

# Testovací sada. U každé otázky víme, co čekáme:
#   expect_answer=True  → appka MÁ odpovědět, a odpověď má obsahovat must_contain
#   expect_answer=False → appka MÁ říct "nevím" (otázka není v poznámkách)
TEST_CASES = [
    {"question": "Komu chci svým obsahem pomáhat?", "expect_answer": True, "must_contain": "35"},
    {"question": "Jaký je můj dlouhodobý cíl?", "expect_answer": True, "must_contain": "firm"},
    {"question": "Kdo mě inspiruje k tvorbě obsahu?", "expect_answer": True, "must_contain": "chris"},
    {"question": "Jaký styl vystupování chci mít?", "expect_answer": True, "must_contain": "autentick"},
    {"question": "Jaký mikrofon mám koupit?", "expect_answer": False},
    {"question": "Jak upéct čokoládový dort?", "expect_answer": False},
    {"question": "Jaké je hlavní město Francie?", "expect_answer": False},
]


def check(case: dict) -> tuple[bool, str]:
    """Spustí jeden test a vrátí (prošel?, krátký popis)."""
    chunks = retrieve(case["question"], INDEX)
    best = chunks[0]["score"]
    answered = best >= SIMILARITY_THRESHOLD  # rozhodne appka odpovědět, nebo ne?

    if case["expect_answer"]:
        # Čekáme odpověď.
        if not answered:
            return False, f"čekal jsem odpověď, ale appka řekla 'nevím' (shoda {best:.2f})"
        text = answer(case["question"], chunks).lower()
        if case["must_contain"].lower() not in text:
            return False, f"odpověď neobsahuje očekávané slovo '{case['must_contain']}'"
        return True, f"odpovězeno a obsahuje '{case['must_contain']}' (shoda {best:.2f})"
    else:
        # Čekáme "nevím".
        if answered:
            return False, f"čekal jsem 'nevím', ale appka odpověděla (shoda {best:.2f})"
        return True, f"správně řekla 'nevím' (shoda {best:.2f})"


def main() -> None:
    passed = 0
    print(f"Spouštím {len(TEST_CASES)} testů...\n")
    for case in TEST_CASES:
        ok, detail = check(case)
        mark = "✅ PASS" if ok else "❌ FAIL"
        print(f"{mark}  {case['question']}")
        print(f"         {detail}")
        if ok:
            passed += 1

    print(f"\n==============================")
    print(f"Výsledek: {passed}/{len(TEST_CASES)} testů prošlo")
    print(f"==============================")


if __name__ == "__main__":
    main()
