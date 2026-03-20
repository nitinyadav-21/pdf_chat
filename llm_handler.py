"""
llm_handler.py
Uses HuggingFace Inference API (v2) - the current working free endpoint.
"""

from __future__ import annotations
import os
import re
import textwrap
from typing import List


def _clean_text(text: str) -> str:
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r" +", " ", text)
    return text.strip()


def _build_messages(question: str, context_chunks: List[str]) -> List[dict]:
    cleaned = [_clean_text(c) for c in context_chunks]
    context = "\n\n---\n\n".join(
        f"[Excerpt {i+1}]:\n{chunk}" for i, chunk in enumerate(cleaned)
    )
    return [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Answer the user's question based ONLY "
                "on the provided document excerpts. If the answer is not in the excerpts, "
                "say so clearly. Be concise and write in full sentences."
            ),
        },
        {
            "role": "user",
            "content": f"Document excerpts:\n{context}\n\nQuestion: {question}",
        },
    ]


def _answer_hf(question: str, context_chunks: List[str]) -> str:
    import requests

    hf_token = os.getenv("HF_TOKEN", "")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN is not set. Go to Streamlit → Manage App → Settings → Secrets "
            "and add: HF_TOKEN = \"hf_your_token_here\""
        )

    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json",
    }

    # These models support the new /v1/chat/completions endpoint (free)
    models = [
        "mistralai/Mistral-7B-Instruct-v0.3",
        "HuggingFaceH4/zephyr-7b-beta",
        "microsoft/Phi-3-mini-4k-instruct",
    ]

    messages = _build_messages(question, context_chunks)
    last_error = None

    for model_id in models:
        try:
            url = f"https://api-inference.huggingface.co/models/{model_id}/v1/chat/completions"
            payload = {
                "model": model_id,
                "messages": messages,
                "max_tokens": 512,
                "temperature": 0.3,
            }
            resp = requests.post(url, headers=headers, json=payload, timeout=30)

            if resp.status_code == 401:
                raise RuntimeError(
                    "Invalid HF_TOKEN (401). Please check your token in Streamlit Secrets."
                )
            if resp.status_code in (503, 504):
                last_error = f"{model_id} unavailable ({resp.status_code})"
                continue
            if resp.status_code == 410:
                last_error = f"{model_id} discontinued (410)"
                continue

            resp.raise_for_status()
            data = resp.json()

            text = data["choices"][0]["message"]["content"].strip()
            return _clean_text(text)

        except RuntimeError:
            raise
        except requests.exceptions.Timeout:
            last_error = f"{model_id} timed out"
            continue
        except Exception as e:
            last_error = f"{model_id}: {e}"
            continue

    raise RuntimeError(f"All models failed. Last error: {last_error}")


def _answer_extractive(question: str, context_chunks: List[str]) -> str:
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

    return best_sentence or "I could not find a relevant answer in the provided documents."


def get_answer(question: str, context_chunks: List[str]) -> str:
    # Ollama (local only)
    if os.getenv("USE_OLLAMA", "").strip() == "1":
        try:
            import requests
            model = os.getenv("OLLAMA_MODEL", "mistral")
            prompt = question
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except Exception as e:
            print(f"[llm_handler] Ollama failed: {e}")

    # HuggingFace new chat completions API
    try:
        return _answer_hf(question, context_chunks)
    except Exception as e:
        extractive = _answer_extractive(question, context_chunks)
        return (
            f"⚠️ **LLM Error:** {e}\n\n"
            f"---\n"
            f"**Extractive answer from document:**\n\n{extractive}"
        )
