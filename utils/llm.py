"""Configuration and Gemini API interactions."""
import json
import os
from typing import Dict, List, Optional

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # reads .env locally; harmless on Streamlit Cloud


def get_config(key: str, default: Optional[str] = None) -> Optional[str]:
    """Streamlit secrets first (cloud), then environment/.env (local)."""
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass  # no secrets file locally
    return os.getenv(key, default)


API_KEY = get_config("GEMINI_API_KEY")
MODEL_NAME = get_config("GEMINI_MODEL", "gemini-2.5-flash")
BASE_URL = (
    "https://generativelanguage.googleapis.com/v1beta/"
    f"models/{MODEL_NAME}:generateContent"
)


class GeminiClient:
    def __init__(self, api_key: str, base_url: str = BASE_URL):
        self.api_key = api_key
        self.base_url = base_url

    def chat(
        self,
        messages: List[Dict[str, str]],
        system_instruction: str,
        temperature: float = 0.3,
    ) -> str:
        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
        }
        contents = []
        for m in messages:
            role = "model" if m["role"] == "assistant" else m["role"]
            if role not in ("user", "model"):
                continue
            contents.append({"role": role, "parts": [{"text": m["content"]}]})

        payload = {
            "systemInstruction": {"parts": [{"text": system_instruction}]},
            "contents": contents,
            "generationConfig": {"temperature": temperature},
        }
        response = requests.post(
            self.base_url, headers=headers, json=payload, timeout=60
        )
        if not response.ok:
            raise RuntimeError(
                f"Gemini API error {response.status_code}: {response.text}"
            )

        candidates = response.json().get("candidates", [])
        if not candidates:
            raise RuntimeError("Gemini returned no answer (possibly blocked).")
        parts = candidates[0].get("content", {}).get("parts", [])
        answer = "\n".join(p["text"] for p in parts if "text" in p)
        if not answer:
            raise RuntimeError("Gemini returned no text.")
        return answer


@st.cache_resource
def get_client() -> GeminiClient:
    return GeminiClient(API_KEY)


def build_system_instruction(
    user_name: str, context: Dict, knowledge: str = ""
) -> str:
    return f"""
ROLE:
You are a digital twin assistant for {user_name}.

PURPOSE:
Help the user understand their activity patterns and decide
what activity to consider next.

EVIDENCE (current data, JSON):
{json.dumps(context, indent=2)}

KNOWLEDGE BASE (retrieved for this question):
{knowledge or "None retrieved."}

RULES:
- Base activity-related recommendations on the supplied data.
- Distinguish recorded facts from estimates and suggestions.
- Probabilities describe historical patterns, not certainties.
- Do not claim access to data that was not supplied.
- If the evidence is insufficient, say so and ask a question.
- Do not invent activities, events or user preferences.

STYLE:
Be direct, practical and analytical.
""".strip()