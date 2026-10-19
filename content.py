"""
CONTENT — generátor nápadů a scénářů na obsah, tvým hlasem (Creator Mato).

Používá tvůj profil hlasu ze souboru voice.md (soukromý, v .gitignore),
takže všechno, co vygeneruje, zní jako ty a drží tvoje hranice.

Režimy:
  ./.venv/bin/python content.py ideas             # vymyslí nápady na obsah
  ./.venv/bin/python content.py ideas "AI"        # nápady na dané téma
  ./.venv/bin/python content.py reel "<téma>"     # b-roll reel (text na obrazovku)
  ./.venv/bin/python content.py script "<téma>"   # mluvený scénář do kamery
"""

import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()
CHAT_MODEL = "gpt-4o-mini"

VOICE_FILE = Path("voice.md")


def load_voice() -> str:
    """Načte profil hlasu. Ten je pro model "návod", jak má psát jako ty."""
    if not VOICE_FILE.exists():
        print("Chybí voice.md (profil hlasu). Vytvoř ho ve složce projektu.")
        sys.exit(1)
    return VOICE_FILE.read_text(encoding="utf-8")


def ask_model(voice: str, task: str) -> str:
    """Pošle modelu tvůj hlas (jako systémový návod) + konkrétní zadání."""
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": voice},
            {"role": "user", "content": task},
        ],
        temperature=0.9,  # trochu kreativity, ať nápady nejsou nudné
    )
    return response.choices[0].message.content


def generate_ideas(voice: str, topic: str | None) -> str:
    tema = f" na téma: {topic}" if topic else ""
    return ask_model(
        voice,
        f"Vymysli 6 nápadů na krátký obsah (reels){tema}. Každý nápad je jeden "
        "úderný koncept (ne celé video), buď vzdělávací nebo zábavný, přesně v mojí "
        "nice a mým hlasem. Vrať jako prostý seznam, jeden nápad na řádek, krátce "
        "a konkrétně. Nic mimo moje témata.",
    )


def make_reel(voice: str, topic: str) -> str:
    return ask_model(
        voice,
        f"Vymysli jeden b-roll reel na téma: {topic}. Vrať:\n"
        "- TEXT: velký úderný text na obrazovku (1 myšlenka, pár krátkých řádků, "
        "česky, mým hlasem)\n"
        "- TITULKY: anglický překlad toho textu\n"
        "- B-ROLL: 2–3 návrhy záběrů z mého světa (fitko, MacBook, káva, město, "
        "škola, stůl)\n"
        "Žádný voiceover, žádný pozdrav.",
    )


def make_script(voice: str, topic: str) -> str:
    return ask_model(
        voice,
        f"Napiš mluvený scénář do kamery (~20–40 sekund) na téma: {topic}. "
        "Struktura: silný HOOK hned v první větě (žádné 'ahoj'), pak krátké tělo "
        "z vlastní zkušenosti nebo záměru, a úderné zakončení. Jen text scénáře, "
        "mým hlasem, česky. Nic vymyšleného.",
    )


def main() -> None:
    if len(sys.argv) < 2:
        print('Použití: content.py [ideas|reel|script] "<téma>"')
        return

    voice = load_voice()
    mode = sys.argv[1]
    topic = " ".join(sys.argv[2:])

    if mode == "ideas":
        print(generate_ideas(voice, topic or None))
    elif mode == "reel" and topic:
        print(make_reel(voice, topic))
    elif mode == "script" and topic:
        print(make_script(voice, topic))
    elif mode in ("reel", "script"):
        print(f'Zadej téma: content.py {mode} "tvoje téma"')
    else:
        print(f"Neznámý režim '{mode}'. Použij: ideas, reel, nebo script.")


if __name__ == "__main__":
    main()
