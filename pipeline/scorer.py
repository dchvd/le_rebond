"""
scorer.py — Match and player rating generation via Qwen/Ollama.

Takes the structured analysis JSON produced by analyzer.py and asks the
Qwen model to generate coherent numerical ratings for the match, teams,
and individual players.
"""

import json
import os
import time
from pathlib import Path
from typing import Any

import requests

_PROMPT_FILE = Path(__file__).parent.parent / "prompts" / "scoring_prompt.txt"


def _load_prompt() -> str:
    """Load the scoring system prompt from disk.

    Returns:
        The prompt text as a string.

    Raises:
        FileNotFoundError: If the prompt file is missing.
    """
    if not _PROMPT_FILE.is_file():
        raise FileNotFoundError(f"Scoring prompt not found: {_PROMPT_FILE}")
    return _PROMPT_FILE.read_text(encoding="utf-8")


def _call_ollama(
    base_url: str,
    model: str,
    system_prompt: str,
    user_content: str,
    retries: int = 2,
) -> str:
    """Send a chat request to the Ollama local API.

    Retries with exponential backoff on transient failures.

    Args:
        base_url: Ollama base URL, e.g. "http://localhost:11434".
        model: Model identifier, e.g. "qwen2.5:3b".
        system_prompt: The system-role message.
        user_content: The user-role message.
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

    for attempt in range(retries + 1):
        try:
            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()
            data = response.json()
            return data["message"]["content"]
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(
                f"Cannot connect to Ollama at {base_url}. "
                "Make sure Ollama is running: `ollama serve`"
            ) from exc
        except requests.exceptions.HTTPError as exc:
            if attempt < retries:
                wait = 2 ** attempt
                print(f"[Scorer] HTTP error ({exc}), retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"Ollama API returned HTTP error after {retries + 1} attempts: {exc}"
                ) from exc
        except Exception as exc:
            if attempt < retries:
                wait = 2 ** attempt
                print(f"[Scorer] Unexpected error ({exc}), retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"Ollama request failed after {retries + 1} attempts: {exc}"
                ) from exc

    raise RuntimeError("Ollama request failed (unknown reason)")


def _extract_json(raw: str) -> dict[str, Any]:
    """Parse a JSON object from a raw LLM response string.

    Args:
        raw: Raw text returned by the LLM.

    Returns:
        A parsed Python dictionary.

    Raises:
        ValueError: If no valid JSON object can be extracted.
    """
    text = raw.strip()
    for fence in ("```json", "```"):
        if text.startswith(fence):
            text = text[len(fence):]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in LLM response:\n{raw[:500]}")

    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"JSON decode error: {exc}\nRaw snippet:\n{text[start:start+300]}"
        ) from exc


def _label_for_score(value: float) -> str:
    """Return the French quality label for a numeric score 0–10.

    Args:
        value: Numeric score between 0 and 10.

    Returns:
        One of: Mediocre, Correct, Bon, Tres bon, Exceptionnel.
    """
    if value <= 3:
        return "Mediocre"
    if value <= 5:
        return "Correct"
    if value <= 7:
        return "Bon"
    if value < 10:
        return "Tres bon"
    return "Exceptionnel"


def _validate_and_fix_scores(scores: dict[str, Any]) -> dict[str, Any]:
    """Validate the scorer output and repair common issues.

    Ensures all score fields are numeric and that the final_match_score
    label is consistent with the numeric value.

    Args:
        scores: Raw scores dictionary from the LLM.

    Returns:
        Cleaned scores dictionary.
    """
    def _to_float(val: Any, default: float = 5.0) -> float:
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    # Normalise match_rating scores
    for key in ("overall_quality", "entertainment_value", "competitiveness"):
        bucket = (scores.get("match_rating") or {}).get(key)
        if isinstance(bucket, dict):
            bucket["score"] = _to_float(bucket.get("score"), 5.0)

    # Normalise team_ratings
    for team_key in list((scores.get("team_ratings") or {}).keys()):
        team = scores["team_ratings"][team_key]
        for field in ("overall", "attack", "defense", "tactics"):
            team[field] = _to_float(team.get(field), 5.0)

    # Normalise player_ratings
    for player in scores.get("player_ratings") or []:
        player["score"] = _to_float(player.get("score"), 5.0)

    # Ensure final_match_score label is consistent
    fms = scores.get("final_match_score") or {}
    value = _to_float(fms.get("value"), 5.0)
    fms["value"] = value
    fms["label"] = _label_for_score(value)
    scores["final_match_score"] = fms

    return scores


def score(
    analysis: dict[str, Any],
    base_url: str,
    model: str,
    output_dir: str,
    match_slug: str,
) -> dict[str, Any]:
    """Generate coherent match ratings from the structured analysis.

    Args:
        analysis: The merged analysis dictionary from analyzer.py.
        base_url: Ollama API base URL.
        model: Qwen model identifier.
        output_dir: Directory to save the scores JSON.
        match_slug: Safe filename slug derived from the match name.

    Returns:
        The validated scores as a Python dictionary.

    Raises:
        RuntimeError: On API failure or invalid JSON response (after retry).
    """
    system_prompt = _load_prompt()
    user_content = json.dumps(analysis, ensure_ascii=False, indent=2)

    print(f"[Scorer] Generating ratings with model '{model}'...")

    for attempt in range(2):
        raw = _call_ollama(base_url, model, system_prompt, user_content)
        try:
            parsed = _extract_json(raw)
            break
        except ValueError as exc:
            if attempt == 0:
                print(f"[Scorer] Malformed JSON, retrying... ({exc})")
            else:
                raise RuntimeError(
                    f"LLM returned invalid JSON for scoring after retry: {exc}"
                ) from exc

    validated = _validate_and_fix_scores(parsed)

    # Persist to disk
    os.makedirs(output_dir, exist_ok=True)
    scores_path = os.path.join(output_dir, f"{match_slug}_scores.json")
    with open(scores_path, "w", encoding="utf-8") as fh:
        json.dump(validated, fh, ensure_ascii=False, indent=2)

    print(f"[Scorer] Scores saved → {scores_path}")
    return validated
