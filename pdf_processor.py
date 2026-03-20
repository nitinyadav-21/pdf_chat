from __future__ import annotations
import re
from pathlib import Path
from typing import List, Tuple

from pypdf import PdfReader


def _clean_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text_from_pdfs(
    file_paths: List[str],
    chunk_size: int = 500,
    overlap: int = 100,
) -> Tuple[List[str], List[dict]]:
    chunks: List[str] = []
    metadata: List[dict] = []

    for path in file_paths:
        file_name = Path(path).name
        try:
            reader = PdfReader(path)
        except Exception as e:
            print(f"[pdf_processor] Cannot read {file_name}: {e}")
            continue

        full_text_by_page: List[Tuple[int, str]] = []

        for page_num, page in enumerate(reader.pages, start=1):
            try:
                raw = page.extract_text() or ""
                cleaned = _clean_text(raw)
                if cleaned:
                    full_text_by_page.append((page_num, cleaned))
            except Exception:
                continue

        for page_num, page_text in full_text_by_page:
            page_chunks = _split_into_chunks(page_text, chunk_size, overlap)
            for idx, chunk in enumerate(page_chunks):
                if chunk.strip():
                    chunks.append(chunk)
                    metadata.append(
                        {
                            "source": file_name,
                            "page": page_num,
                            "chunk_index": idx,
                        }
                    )

    return chunks, metadata


def _split_into_chunks(
    text: str,
    chunk_size: int,
    overlap: int,
) -> List[str]:
    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end < len(text):
            boundary = _find_sentence_boundary(text, end)
            if boundary and boundary > start + overlap:
                end = boundary

        chunks.append(text[start:end].strip())
        next_start = end - overlap
        if next_start <= start:
            next_start = start + 1  
        start = next_start

    return chunks


def _find_sentence_boundary(text: str, near: int) -> int | None:
    search_from = max(0, near - 100)
    snippet = text[search_from:near]
    for punct in (". ", "! ", "? ", ".\n", "!\n", "?\n"):
        pos = snippet.rfind(punct)
        if pos != -1:
            return search_from + pos + len(punct)
    return None
