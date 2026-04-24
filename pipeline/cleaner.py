"""
cleaner.py — Step 1b: normalise a raw ASR transcript before JSON extraction.

Fixes player name mishearings, strips filler words, and segments the text
by event type so the downstream analyzer receives clean, structured input.
"""

import os
from pathlib import Path

from pipeline.llm import call_llm

_CHARS_PER_TOKEN = 4
_PROMPT_FILE = Path(__file__).parent.parent / "prompts" / "cleaning_prompt.txt"

MAX_CHUNK_TOKENS = int(os.getenv("MAX_CHUNK_TOKENS", "1500"))


def _load_prompt() -> str:
    if not _PROMPT_FILE.is_file():
        raise FileNotFoundError(f"Cleaning prompt not found: {_PROMPT_FILE}")
    return _PROMPT_FILE.read_text(encoding="utf-8")


def _chunk_text(text: str, max_tokens: int) -> list[str]:
    max_chars = max_tokens * _CHARS_PER_TOKEN
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    paragraphs = text.split("\n\n")
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para)
        if current_len + para_len > max_chars and current:
            chunks.append("\n\n".join(current))
            current = [para]
            current_len = para_len
        else:
            current.append(para)
            current_len += para_len + 2

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def clean(
    commentary: str,
    match_name: str,
    sport: str,
    base_url: str,
    model: str,
    max_chunk_tokens: int,
    output_dir: str,
    match_slug: str,
    provider: str = "ollama",
    api_key: str = "",
    groq_model: str = "llama-3.3-70b-versatile",
    temperature: float = 0.1,
) -> str:
    """Clean and normalise a raw ASR transcript.

    Returns clean plain text with event markers and corrected names,
    ready to pass to the analyzer for JSON extraction.
    """
    system_prompt = _load_prompt()
    context_header = f"Sport: {sport} | Match: {match_name}\n\n"

    chunks = _chunk_text(commentary, max_chunk_tokens)
    total = len(chunks)
    display_model = groq_model if provider == "groq" else model
    print(f"[Cleaner] Normalising {total} chunk(s) via {provider} / '{display_model}'...")

    os.makedirs(output_dir, exist_ok=True)
    raw_log_path = os.path.join(output_dir, f"{match_slug}_cleaner_raw.txt")

    cleaned_chunks: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        print(f"[Cleaner] Cleaning chunk {idx}/{total}...")
        user_content = context_header + chunk

        raw = call_llm(
            provider,
            system_prompt,
            user_content,
            base_url=base_url,
            ollama_model=model,
            api_key=api_key,
            groq_model=groq_model,
            temperature=temperature,
        )

        with open(raw_log_path, "a", encoding="utf-8") as fh:
            fh.write(f"=== chunk {idx}/{total} ===\n{raw}\n\n")

        cleaned_chunks.append(raw.strip())

    return "\n\n".join(cleaned_chunks)
