"""Minimal RAG: retrieve relevant entries from data/knowledge_base.json."""
import json
import re
from pathlib import Path
from typing import Dict, List

KB_PATH = Path(__file__).resolve().parent.parent / "data" / "knowledge_base.json"

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "is", "are",
    "i", "my", "me", "do", "what", "should", "how", "for", "it", "with",
}


def _tokens(text: str) -> set:
    return {
        w for w in re.findall(r"[a-z0-9']+", text.lower()) if w not in STOPWORDS
    }


def load_knowledge_base(path: Path = KB_PATH) -> List[Dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def retrieve(query: str, k: int = 3, path: Path = KB_PATH) -> List[Dict]:
    """Return up to k entries ranked by shared keywords with the query."""
    q = _tokens(query)
    if not q:
        return []
    scored = []
    for entry in load_knowledge_base(path):
        doc = _tokens(entry.get("title", "") + " " + entry.get("text", ""))
        score = len(q & doc)
        if score > 0:
            scored.append((score, entry))
    scored.sort(key=lambda s: s[0], reverse=True)
    return [e for _, e in scored[:k]]


def format_snippets(entries: List[Dict]) -> str:
    if not entries:
        return "No relevant knowledge-base entries found."
    return "\n".join(f"- [{e['title']}] {e['text']}" for e in entries)