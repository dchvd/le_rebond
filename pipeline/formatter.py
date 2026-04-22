"""
formatter.py — Final JSON report assembly for Le Rebond.

Merges all pipeline stage outputs (analysis + scores) into a single
master report file saved to the outputs directory.
"""

import json
import os
from datetime import datetime
from typing import Any


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two dictionaries, preferring override values.

    Args:
        base: Base dictionary.
        override: Dictionary whose values take precedence.

    Returns:
        Merged dictionary (new object, inputs are not mutated).
    """
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def build_report(
    match_name: str,
    match_date: str,
    sport: str,
    analysis: dict[str, Any],
    scores: dict[str, Any],
    output_dir: str,
    match_slug: str,
    input_file: str,
) -> str:
    """Assemble and persist the full pipeline report as a JSON file.

    Combines match metadata, key events, positive/negative elements,
    team performance, commentary sentiment, and all rating scores into
    a single master JSON document.

    Args:
        match_name: Human-readable match name (e.g. "PSG vs Barcelona").
        match_date: Match date string provided by the user.
        sport: Sport type (e.g. "football").
        analysis: Structured analysis dictionary from analyzer.py.
        scores: Validated scores dictionary from scorer.py.
        output_dir: Directory where the report will be written.
        match_slug: Safe filename slug derived from the match name.
        input_file: Path to the original input file (for traceability).

    Returns:
        Absolute path to the written report JSON file.
    """
    # ── Pipeline metadata ──────────────────────────────────────────────────────
    meta = {
        "pipeline": "Le Rebond",
        "version": "1.0.0",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "input_file": input_file,
    }

    # ── Match context (CLI args + LLM-extracted info) ─────────────────────────
    llm_match_info = analysis.get("match_info") or {}
    match_context = {
        "name": match_name,
        "date": match_date,
        "sport": sport,
        "competition": llm_match_info.get("competition", ""),
        "teams": llm_match_info.get("teams") or [],
        "final_score": llm_match_info.get("final_score", ""),
    }

    # ── Full report assembly ───────────────────────────────────────────────────
    report: dict[str, Any] = {
        "meta": meta,
        "match": match_context,
        "key_events": analysis.get("key_events") or [],
        "positive_elements": analysis.get("positive_elements") or [],
        "negative_elements": analysis.get("negative_elements") or [],
        "performance_by_team": analysis.get("performance_by_team") or {},
        "commentary_sentiment": analysis.get("commentary_sentiment") or {},
        "raw_summary": analysis.get("raw_summary", ""),
        "ratings": {
            "match_rating": scores.get("match_rating") or {},
            "team_ratings": scores.get("team_ratings") or {},
            "player_ratings": scores.get("player_ratings") or [],
            "final_match_score": scores.get("final_match_score") or {},
        },
    }

    # ── Write to disk ──────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, f"{match_slug}_full_report.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    return os.path.abspath(report_path)
