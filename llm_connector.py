"""
LLM connector supporting:
  1. Groq  – free cloud API (recommended for low-spec laptops)
  2. Ollama – local models (Gemma 2B, Phi-3 Mini)
"""

import os
from typing import List, Tuple


# ── Prompt builder ────────────────────────────────────────────────────────────
def build_prompt(question: str, chunks: List[Tuple[str, str, float]]) -> str:
    context_parts = []
    for i, (text, source, _) in enumerate(chunks, 1):
        context_parts.append(f"[Source {i}: {source}]\n{text}")
    context = "\n\n".join(context_parts)

    return f"""You are an expert IAS (Indian Administrative Service) exam assistant. 
Answer the question below using ONLY the provided context from IAS study books.
If the answer is not in the context, say "I could not find this in the provided books."
Be concise, accurate, and helpful for UPSC preparation.

Context:
{context}

Question: {question}

Answer:"""


# ── Groq (free cloud API) ─────────────────────────────────────────────────────
def ask_groq(question: str, chunks: List[Tuple], api_key: str, model: str = "llama3-8b-8192") -> str:
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        prompt = build_prompt(question, chunks)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except ImportError:
        return "❌ groq package not installed. Run: pip install groq"
    except Exception as e:
        return f"❌ Groq API error: {e}"


# ── Ollama (local) ────────────────────────────────────────────────────────────
def ask_ollama(question: str, chunks: List[Tuple], model: str = "gemma:2b") -> str:
    try:
        import requests
        prompt = build_prompt(question, chunks)
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        if resp.status_code == 200:
            return resp.json().get("response", "No response").strip()
        else:
            return f"❌ Ollama error: {resp.status_code} – {resp.text}"
    except requests.exceptions.ConnectionError:
        return "❌ Ollama not running. Start it with: ollama serve"
    except Exception as e:
        return f"❌ Ollama error: {e}"
