"""
llm_handler.py
Generates answers from retrieved context using HuggingFace Inference API.
"""

from __future__ import annotations
import os
import re
import textwrap
from typing import List


def _clean_text(text: str) -> str:
    """Fix PDFs where every word is on its own line."""
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r" +", " ", text)
    return text.strip()


def _build_prompt(question: str, context_chunks: List[str]) -> str:
    cleaned = [_clean_text(c) for c in context_chunks]
    context = "\n\n---\n\n".join(
        f"[Excerpt {i+1}]:\n{chunk}" for i, chunk in enumerate(cleaned)
    )
    return textwrap.dedent(f"""
        You are a helpful assistant. Answer the user's question based ONLY on the
        provided document excerpts. If the answer is not in the excerpts, say so.
        Be concise and write in full sentences.

        Document excerpts:
        {context}

        Question: {question}

        Answer:
    """).strip()


def _answer_hf(prompt: str) -> str:
    import requests

    hf_token = os.getenv("HF_TOKEN", "")
    if not hf_token:
        raise RuntimeError("HF_TOKEN is not set in Streamlit secrets.")

    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json",
    }

    # Try multiple models in order — first one that responds wins
    models = [
        "HuggingFaceH4/zephyr-7b-beta",
        "mistralai/Mistral-7B-Instruct-v0.2",
        "tiiuae/falcon-7b-instruct",
    ]

    last_error = None
    for model_id in models:
        try:
            url = f"https://api-inference.huggingface.co/models/{model_id}"
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 512,
                    "temperature": 0.3,
                    "return_full_text": False,
                    "do_sample": True,
                },
            }
            resp = requests.post(url, headers=headers, json=payload, timeout=30)

            # Model still loading — try next
            if resp.status_code == 503:
                last_error = f"{model_id} is loading (503)"
                continue

            # Auth error — no point trying other models
            if resp.status_code == 401:
                raise RuntimeError(
                    "Invalid HF_TOKEN. Go to Streamlit → Manage App → Settings → Secrets "
                    "and make sure HF_TOKEN = \"hf_your_token\" is set correctly."
                )

            resp.raise_for_status()
            data = resp.json()

            if isinstance(data, list) and data:
                text = data[0].get("generated_text", "").strip()
                return _clean_text(text)

            last_error = f"{model_id} returned unexpected format: {data}"

        except requests.exceptions.Timeout:
            last_error = f"{model_id} timed out"
            continue
        except RuntimeError:
            raise
        except Exception as e:
            last_error = f"{model_id} error: {e}"
            continue

    raise RuntimeError(f"All HuggingFace models failed. Last error: {last_error}")


def _answer_extractive(question: str, context_chunks: List[str]) -> str:
    """Keyword-overlap fallback — always works without any API."""
    q_words = set(re.findall(r"\w+", question.lower()))
    best_sentence = ""
    best_score = -1

    for chunk in context_chunks:
        cleaned_chunk = _clean_text(chunk)
        sentences = re.split(r"(?<=[.!?])\s+", cleaned_chunk)
        for sent in sentences:
            words = set(re.findall(r"\w+", sent.lower()))
            score = len(q_words & words)
            if score > best_score:
                best_score = score
                best_sentence = sent

    if best_sentence:
        return best_sentence
    return "I could not find a relevant answer in the provided documents."


def get_answer(question: str, context_chunks: List[str]) -> str:
    """Try HuggingFace API first, fall back to extractive if it fails."""
    prompt = _build_prompt(question, context_chunks)

    # Ollama (local only)
    if os.getenv("USE_OLLAMA", "").strip() == "1":
        try:
            import requests
            model = os.getenv("OLLAMA_MODEL", "mistral")
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except Exception as e:
            print(f"[llm_handler] Ollama failed: {e}")

    # HuggingFace API
    try:
        return _answer_hf(prompt)
    except Exception as e:
        # Show the real error clearly so it's easy to debug
        extractive = _answer_extractive(question, context_chunks)
        return (
            f"⚠️ **LLM Error:** {e}\n\n"
            f"---\n"
            f"**Extractive answer from document:**\n\n{extractive}"
        )
