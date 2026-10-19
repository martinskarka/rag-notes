"""
AGENT — malý AI agent, který umí SÁM POUŽÍVAT NÁSTROJE (function calling).

Na rozdíl od ask.py (které jen odpoví) tenhle agent dostane sadu "nástrojů"
(obyčejné Python funkce) a model se SÁM rozhodne, který zavolat, aby ti
odpověděl. Tomuhle se říká "tool calling" nebo "function calling" a je to
základ všech AI agentů.

Spuštění:  ./.venv/bin/python agent.py "Jaký je můj cíl a kolik je hodin?"
"""

import ast
import json
import operator
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Znovu použijeme vyhledávání v poznámkách z našeho RAG (ask.py).
from ask import retrieve

load_dotenv()
client = OpenAI()
CHAT_MODEL = "gpt-4o-mini"
INDEX = json.loads(Path("index.json").read_text(encoding="utf-8"))


# ========== 1) NÁSTROJE (obyčejné funkce, které agent může zavolat) ==========

def search_notes(query: str) -> str:
    """Prohledá osobní poznámky a vrátí nejrelevantnější pasáže."""
    chunks = retrieve(query, INDEX)
    return "\n\n".join(f"[{c['source']}] {c['text']}" for c in chunks)


def current_datetime() -> str:
    """Vrátí aktuální datum a čas."""
    return datetime.now().strftime("%d.%m.%Y %H:%M")


# Bezpečný výpočet matematického výrazu (jen čísla a + - * / ** %, nic víc).
_MATH_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
}


def _eval_node(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp):
        return _MATH_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        return _MATH_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("nepodporovaný výraz")


def calculator(expression: str) -> str:
    """Spočítá matematický výraz, např. '0.15 * 2400' nebo '(10+5)*3'."""
    tree = ast.parse(expression, mode="eval")
    return str(_eval_node(tree.body))


# Mapa jméno → funkce, ať nástroj umíme zavolat podle jména, které vrátí model.
AVAILABLE_TOOLS = {
    "search_notes": search_notes,
    "current_datetime": current_datetime,
    "calculator": calculator,
}


# ========== 2) POPIS NÁSTROJŮ PRO MODEL (co umí a co berou za parametry) =====
# Tohle je "menu", které dáme modelu, aby věděl, jaké nástroje má k dispozici.

TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "search_notes",
            "description": (
                "Prohledá osobní poznámky uživatele a vrátí relevantní pasáže. "
                "Použij vždy, když se otázka týká uživatele — jeho cílů, vize, "
                "obsahu, cílové skupiny apod."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Co hledat v poznámkách"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "current_datetime",
            "description": "Vrátí aktuální datum a čas. Použij, když se ptá na dnešek nebo čas.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Spočítá matematický výraz (čísla a + - * / ** %). Použij pro JAKÝKOLIV výpočet.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Výraz k výpočtu, např. '0.15 * 2400'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
]


# ========== 3) SMYČKA AGENTA (model → nástroj → model → odpověď) ============

def run_agent(question: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "Jsi užitečný český asistent. Pro výpočty, informace z poznámek "
                "nebo aktuální čas VŽDY použij dostupné nástroje (nepočítej zpaměti). "
                "Odpovídej česky."
            ),
        },
        {"role": "user", "content": question},
    ]

    # Model může chtít zavolat nástroj. Pak mu dáme výsledek a zeptáme se znovu.
    # Opakujeme, dokud nemá finální odpověď (max 5x jako pojistka).
    for _ in range(5):
        response = client.chat.completions.create(
            model=CHAT_MODEL, messages=messages, tools=TOOLS_SPEC
        )
        msg = response.choices[0].message

        # Žádný požadavek na nástroj → model má finální odpověď.
        if not msg.tool_calls:
            return msg.content

        # Jinak přidáme zprávu modelu (s požadavky) a zavoláme nástroje.
        messages.append(msg)
        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments or "{}")
            print(f"   🔧 agent se rozhodl použít nástroj: {name}({args})")

            result = AVAILABLE_TOOLS[name](**args)  # skutečně zavoláme funkci

            # Výsledek nástroje vrátíme modelu, ať s ním dál pracuje.
            messages.append(
                {"role": "tool", "tool_call_id": tool_call.id, "content": str(result)}
            )

    return "(agent se zacyklil — zkus otázku jinak)"


def main() -> None:
    question = " ".join(sys.argv[1:]) or input("Zeptej se agenta: ")
    print(f"\n👤 {question}\n")
    print(f"🤖 {run_agent(question)}")


if __name__ == "__main__":
    main()
