from __future__ import annotations
import os
import textwrap
from typing import List

def _build_prompt(question: str, context_chunks: List[str]) -> str:
    context = "\n\n---\n\n".join(
        f"[Excerpt {i+1}]:\n{chunk}" for i, chunk in enumerate(context_chunks)
    )
    return textwrap.dedent(f"""
        You are a helpful assistant. Answer the user's question based ONLY on the
        provided document excerpts. If the answer is not in the excerpts, say so clearly.
        Be concise and accurate.

        Document excerpts:
        {context}

        Question: {question}

        Answer:
    """).strip()

def _answer_hf(prompt: str) -> str:
    import requests

    model_id = os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
    hf_token = os.getenv("HF_TOKEN", "")

    headers = {"Content-Type": "application/json"}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 512,
            "temperature": 0.3,
            "return_full_text": False,
        },
    }

    url = f"https://api-inference.huggingface.co/models/{model_id}"
    resp = requests.post(url, headers=headers, json=payload, timeout=60)

    if resp.status_code == 503:
        raise RuntimeError("HuggingFace model is loading, please retry in a few seconds.")

    resp.raise_for_status()
    data = resp.json()

    if isinstance(data, list) and data:
        return data[0].get("generated_text", "").strip()
    raise RuntimeError(f"Unexpected HF response: {data}")

def _answer_extractive(question: str, context_chunks: List[str]) -> str:
    import re

    q_words = set(re.findall(r"\w+", question.lower()))
    best_sentence = ""
    best_score = -1

    for chunk in context_chunks:
        sentences = re.split(r"(?<=[.!?])\s+", chunk)
        for sent in sentences:
            words = set(re.findall(r"\w+", sent.lower()))
            score = len(q_words & words)
            if score > best_score:
                best_score = score
                best_sentence = sent

    if best_sentence:
        return (
            f"{best_sentence}\n\n"
            "*(Note: There is some problem :()*"
        )
    return "I could not find a relevant answer in the provided documents."

def get_answer(question: str, context_chunks: List[str]) -> str:
    prompt = _build_prompt(question, context_chunks)

    try:
        return _answer_hf(prompt)
    except Exception as e:
        print(f"[llm_handler] HuggingFace failed: {e}")

    return _answer_extractive(question, context_chunks)
