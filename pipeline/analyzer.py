"""
analyzer.py — LLM-based sports commentary analysis via Qwen/Ollama.

Sends commentary text (optionally split into chunks) to a local Ollama
instance running Qwen and extracts structured match data as JSON.
"""

import json
import os
import time
from pathlib import Path
from typing import Any

import requests

# Maximum characters per LLM chunk (approx 3 000 tokens ≈ 12 000 chars for latin text)
_CHARS_PER_TOKEN = 4
_PROMPT_FILE = Path(__file__).parent.parent / "prompts" / "analysis_prompt.txt"


def _load_prompt() -> str:
    """Load the analysis system prompt from disk.

    Returns:
        The prompt text as a string.

    Raises:
        FileNotFoundError: If the prompt file is missing.
    """
    if not _PROMPT_FILE.is_file():
        raise FileNotFoundError(f"Analysis prompt not found: {_PROMPT_FILE}")
    return _PROMPT_FILE.read_text(encoding="utf-8")


def _call_ollama(
    base_url: str,
    model: str,
    system_prompt: str,
    user_content: str,
    retries: int = 2,
) -> str:
    """Send a chat request to the Ollama local API and return the response text.

    Retries with exponential backoff on transient failures.

    Args:
        base_url: Ollama base URL, e.g. "http://localhost:11434".
        model: Model identifier, e.g. "qwen2.5:3b".
        system_prompt: The system-role message sent to the model.
        user_content: The user-role message (commentary text or JSON).
        retries: Number of retry attempts after the first failure.

    Returns:
        The raw text content of the model's response.

    Raises:
        RuntimeError: If all attempts fail.
    """
    url = f"{base_url.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    }

    last_error = None
    for attempt in range(retries + 1):
        try:
            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()
            data = response.json()
            return data["message"]["content"]
        except requests.exceptions.ConnectionError as exc:
            last_error = exc
            raise RuntimeError(
                f"Cannot connect to Ollama at {base_url}. "
                "Make sure Ollama is running: `ollama serve` and the model is pulled: "
                f"`ollama pull {model}`"
            ) from exc
        except requests.exceptions.HTTPError as exc:
            last_error = exc
            if attempt < retries:
                wait = 2 ** attempt
                print(f"[Analyzer] HTTP error ({exc}), retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"Ollama API returned HTTP error after {retries + 1} attempts: {exc}"
                ) from exc
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                wait = 2 ** attempt
                print(f"[Analyzer] Unexpected error ({exc}), retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"Ollama request failed after {retries + 1} attempts: {exc}"
                ) from last_error

    raise RuntimeError("Ollama request failed (unknown reason)")


def _extract_json(raw: str) -> dict[str, Any]:
    """Parse a JSON object from a raw LLM response string.

    Strips markdown code fences if present, then attempts to locate and
    parse the outermost JSON object.

    Args:
        raw: Raw text returned by the LLM.

    Returns:
        A parsed Python dictionary.

    Raises:
        ValueError: If no valid JSON object can be extracted.
    """
    # Strip markdown fences
    text = raw.strip()
    for fence in ("```json", "```"):
        if text.startswith(fence):
            text = text[len(fence):]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    # Find first { … } block
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in LLM response:\n{raw[:500]}")

    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"JSON decode error in LLM response: {exc}\nRaw snippet:\n{text[start:start+300]}"
        ) from exc


def _chunk_text(text: str, max_tokens: int) -> list[str]:
    """Split a long text into chunks of approximately max_tokens tokens.

    Splits on paragraph/sentence boundaries when possible.

    Args:
        text: The full commentary text.
        max_tokens: Approximate maximum tokens per chunk.

    Returns:
        A list of text chunks.
    """
    max_chars = max_tokens * _CHARS_PER_TOKEN
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    # Split on double newlines (paragraph breaks) first
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
            current_len += para_len + 2  # +2 for the separator

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def _merge_chunk_analyses(analyses: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge multiple chunk analysis JSONs into a single coherent analysis.

    Combines key_events, positive_elements, negative_elements, and
    notable_players. Takes the last non-empty raw_summary. Aggregates
    sentiment averages.

    Args:
        analyses: List of per-chunk analysis dictionaries.

    Returns:
        A single merged analysis dictionary.
    """
    if len(analyses) == 1:
        return analyses[0]

    merged: dict[str, Any] = {
        "match_info": {},
        "key_events": [],
        "positive_elements": [],
        "negative_elements": [],
        "performance_by_team": {
            "team_a": {"strengths": [], "weaknesses": [], "notable_players": []},
            "team_b": {"strengths": [], "weaknesses": [], "notable_players": []},
        },
        "commentary_sentiment": {
            "overall_tone": "mixed",
            "excitement_level": 5,
            "controversy_level": 5,
        },
        "raw_summary": "",
    }

    excitement_levels: list[int] = []
    controversy_levels: list[int] = []

    for analysis in analyses:
        # match_info: use first non-empty
        if not merged["match_info"] and analysis.get("match_info"):
            merged["match_info"] = analysis["match_info"]

        # Accumulate lists
        merged["key_events"].extend(analysis.get("key_events") or [])
        merged["positive_elements"].extend(analysis.get("positive_elements") or [])
        merged["negative_elements"].extend(analysis.get("negative_elements") or [])

        # Performance by team
        for side in ("team_a", "team_b"):
            src = (analysis.get("performance_by_team") or {}).get(side, {})
            dst = merged["performance_by_team"][side]
            dst["strengths"].extend(src.get("strengths") or [])
            dst["weaknesses"].extend(src.get("weaknesses") or [])
            dst["notable_players"].extend(src.get("notable_players") or [])

        # Sentiment aggregation
        sentiment = analysis.get("commentary_sentiment") or {}
        try:
            excitement_levels.append(int(sentiment.get("excitement_level", 5)))
        except (TypeError, ValueError):
            excitement_levels.append(5)
        try:
            controversy_levels.append(int(sentiment.get("controversy_level", 5)))
        except (TypeError, ValueError):
            controversy_levels.append(5)

        # Last non-empty summary wins
        if analysis.get("raw_summary"):
            merged["raw_summary"] = analysis["raw_summary"]

    if excitement_levels:
        merged["commentary_sentiment"]["excitement_level"] = round(
            sum(excitement_levels) / len(excitement_levels)
        )
    if controversy_levels:
        merged["commentary_sentiment"]["controversy_level"] = round(
            sum(controversy_levels) / len(controversy_levels)
        )

    # De-duplicate notable_players by name
    for side in ("team_a", "team_b"):
        seen: set[str] = set()
        unique: list[str] = []
        for player in merged["performance_by_team"][side]["notable_players"]:
            if isinstance(player, str) and player not in seen:
                seen.add(player)
                unique.append(player)
        merged["performance_by_team"][side]["notable_players"] = unique

    return merged


def analyze(
    commentary: str,
    base_url: str,
    model: str,
    max_chunk_tokens: int,
    output_dir: str,
    match_slug: str,
    sport: str = "football",
) -> dict[str, Any]:
    """Analyze sports commentary text with the Qwen model via Ollama.

    If the commentary exceeds max_chunk_tokens, it is split and each chunk
    is analyzed separately before the results are merged.

    Args:
        commentary: Full commentary text.
        base_url: Ollama API base URL.
        model: Qwen model identifier.
        max_chunk_tokens: Maximum tokens per LLM request chunk.
        output_dir: Directory to save the analysis JSON.
        match_slug: Safe filename slug derived from the match name.
        sport: Sport type hint included in the prompt context.

    Returns:
        The merged analysis as a Python dictionary.

    Raises:
        RuntimeError: On API failure or invalid JSON response (after retry).
    """
    system_prompt = _load_prompt()
    # Append sport context to user message
    sport_hint = f"[Sport: {sport}]\n\n"

    chunks = _chunk_text(commentary, max_chunk_tokens)
    total = len(chunks)
    print(f"[Analyzer] Processing {total} chunk(s) with model '{model}'...")

    chunk_analyses: list[dict[str, Any]] = []
    for idx, chunk in enumerate(chunks, start=1):
        print(f"[Analyzer] Analyzing chunk {idx}/{total}...")
        user_content = sport_hint + chunk

        for attempt in range(2):  # one retry on malformed JSON
            raw = _call_ollama(base_url, model, system_prompt, user_content)
            try:
                parsed = _extract_json(raw)
                chunk_analyses.append(parsed)
                break
            except ValueError as exc:
                if attempt == 0:
                    print(f"[Analyzer] Malformed JSON on chunk {idx}, retrying... ({exc})")
                else:
                    raise RuntimeError(
                        f"LLM returned invalid JSON for chunk {idx} after retry: {exc}"
                    ) from exc

    analysis = _merge_chunk_analyses(chunk_analyses)

    # Persist to disk
    os.makedirs(output_dir, exist_ok=True)
    analysis_path = os.path.join(output_dir, f"{match_slug}_analysis.json")
    with open(analysis_path, "w", encoding="utf-8") as fh:
        json.dump(analysis, fh, ensure_ascii=False, indent=2)

    print(f"[Analyzer] Analysis saved → {analysis_path}")
    return analysis
