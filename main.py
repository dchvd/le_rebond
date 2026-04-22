"""
main.py — Le Rebond CLI entrypoint.

Usage examples:
    # Audio input
    python main.py --input match.mp3 --match "PSG vs Barcelona" --date "2025-04-15" --sport football

    # Text input
    python main.py --input commentary.txt --match "PSG vs Barcelona" --date "2025-04-15"

    # Custom output directory
    python main.py --input match.mp3 --match "PSG vs Barcelona" --output-dir ./my_reports
"""

import os
import re
import sys
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv

# ── Load environment variables from .env (if present) ─────────────────────────
load_dotenv()

app = typer.Typer(
    name="le-rebond",
    help="Le Rebond — Sports commentary analysis and rating pipeline.",
    add_completion=False,
)

# ── Supported file extensions ──────────────────────────────────────────────────
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"}
TEXT_EXTENSIONS = {".txt", ".srt", ".vtt"}


def _slugify(text: str) -> str:
    """Convert a match name into a safe filesystem slug.

    Replaces spaces with underscores and removes characters that are
    not alphanumeric, underscores, or hyphens.

    Args:
        text: Raw match name string.

    Returns:
        A lowercase, filesystem-safe slug string.
    """
    slug = text.lower().replace(" ", "_")
    slug = re.sub(r"[^\w\-]", "", slug)
    return slug


def _load_text_file(path: str) -> str:
    """Read a plain-text or subtitle file and return its contents.

    Args:
        path: Path to the .txt, .srt, or .vtt file.

    Returns:
        File contents as a single string.

    Raises:
        typer.Exit: If the file cannot be read.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()
    except OSError as exc:
        typer.echo(f"[Error] Cannot read input file '{path}': {exc}", err=True)
        raise typer.Exit(code=1)


def _chunk_text(text: str, max_tokens: int) -> list[str]:
    """Split text into chunks of approximately max_tokens tokens.

    Used to feed long text directly into the analyzer when no audio
    transcription step is needed. Mirrors the chunking logic in analyzer.py
    so that the max_chunk_tokens setting is respected end-to-end.

    Args:
        text: Full commentary text.
        max_tokens: Approximate maximum tokens per chunk.

    Returns:
        List of text chunks.
    """
    chars_per_token = 4
    max_chars = max_tokens * chars_per_token
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


@app.command()
def run(
    input_file: str = typer.Option(
        ..., "--input", "-i", help="Path to the audio (.mp3/.wav) or text (.txt/.srt) file."
    ),
    match: str = typer.Option(
        ..., "--match", "-m", help='Match name, e.g. "PSG vs Barcelona".'
    ),
    date: str = typer.Option(
        "", "--date", "-d", help="Match date (ISO format recommended, e.g. 2025-04-15)."
    ),
    sport: str = typer.Option(
        "football", "--sport", "-s", help="Sport type: football, basketball, rugby."
    ),
    output_dir: Optional[str] = typer.Option(
        None, "--output-dir", "-o", help="Directory for output files (overrides .env OUTPUT_DIR)."
    ),
) -> None:
    """Run the full Le Rebond analysis pipeline on a commentary file."""
    from pipeline.analyzer import analyze
    from pipeline.formatter import build_report
    from pipeline.scorer import score
    from pipeline.transcriber import transcribe

    # ── Resolve configuration ──────────────────────────────────────────────────
    whisper_model = os.getenv("WHISPER_MODEL", "medium")
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    qwen_model = os.getenv("QWEN_MODEL", "qwen2.5:3b")
    max_chunk_tokens = int(os.getenv("MAX_CHUNK_TOKENS", "3000"))

    if output_dir is None:
        output_dir = os.getenv("OUTPUT_DIR", "./outputs")

    # Resolve relative to cwd
    output_dir = str(Path(output_dir).resolve())
    input_path = str(Path(input_file).resolve())
    match_slug = _slugify(match)

    # ── Validate input file ────────────────────────────────────────────────────
    if not Path(input_path).is_file():
        typer.echo(f"[Error] Input file not found: '{input_path}'", err=True)
        raise typer.Exit(code=1)

    suffix = Path(input_path).suffix.lower()
    is_audio = suffix in AUDIO_EXTENSIONS
    is_text = suffix in TEXT_EXTENSIONS

    if not is_audio and not is_text:
        typer.echo(
            f"[Error] Unsupported file extension '{suffix}'. "
            f"Audio: {', '.join(sorted(AUDIO_EXTENSIONS))}  "
            f"Text: {', '.join(sorted(TEXT_EXTENSIONS))}",
            err=True,
        )
        raise typer.Exit(code=1)

    typer.echo("=" * 60)
    typer.echo("  Le Rebond — Sports Commentary Analysis Pipeline")
    typer.echo("=" * 60)
    typer.echo(f"  Match  : {match}")
    typer.echo(f"  Date   : {date or 'not specified'}")
    typer.echo(f"  Sport  : {sport}")
    typer.echo(f"  Input  : {input_path}")
    typer.echo(f"  Output : {output_dir}")
    typer.echo("=" * 60)

    # ── STEP 1 — Transcription or text load ───────────────────────────────────
    if is_audio:
        typer.echo("\n[Step 1/4] Transcribing audio...")
        try:
            commentary = transcribe(
                audio_path=input_path,
                model_name=whisper_model,
                output_dir=output_dir,
                match_slug=match_slug,
            )
        except RuntimeError as exc:
            typer.echo(f"\n[Error] Transcription failed: {exc}", err=True)
            typer.echo(
                "Tip: Make sure ffmpeg is installed (`brew install ffmpeg` on macOS) "
                "and the audio file is a valid .mp3 or .wav.",
                err=True,
            )
            raise typer.Exit(code=1)
    else:
        typer.echo("\n[Step 1/4] Loading text commentary...")
        commentary = _load_text_file(input_path)
        typer.echo(f"           {len(commentary):,} characters loaded.")

    if not commentary.strip():
        typer.echo("[Error] Commentary is empty after loading/transcription.", err=True)
        raise typer.Exit(code=1)

    # ── STEP 2 — LLM Analysis ─────────────────────────────────────────────────
    typer.echo("\n[Step 2/4] Analyzing commentary with LLM...")
    try:
        analysis = analyze(
            commentary=commentary,
            base_url=ollama_url,
            model=qwen_model,
            max_chunk_tokens=max_chunk_tokens,
            output_dir=output_dir,
            match_slug=match_slug,
            sport=sport,
        )
    except RuntimeError as exc:
        typer.echo(f"\n[Error] Analysis failed: {exc}", err=True)
        typer.echo(
            "Tip: Make sure Ollama is running (`ollama serve`) and the model is available "
            f"(`ollama pull {qwen_model}`).",
            err=True,
        )
        raise typer.Exit(code=1)

    # ── STEP 3 — Score Generation ─────────────────────────────────────────────
    typer.echo("\n[Step 3/4] Generating ratings...")
    try:
        scores = score(
            analysis=analysis,
            base_url=ollama_url,
            model=qwen_model,
            output_dir=output_dir,
            match_slug=match_slug,
        )
    except RuntimeError as exc:
        typer.echo(f"\n[Error] Scoring failed: {exc}", err=True)
        raise typer.Exit(code=1)

    # ── STEP 4 — Final Report Assembly ────────────────────────────────────────
    typer.echo("\n[Step 4/4] Assembling final report...")
    report_path = build_report(
        match_name=match,
        match_date=date,
        sport=sport,
        analysis=analysis,
        scores=scores,
        output_dir=output_dir,
        match_slug=match_slug,
        input_file=input_path,
    )

    # ── Success summary ────────────────────────────────────────────────────────
    fms = scores.get("final_match_score", {})
    final_value = fms.get("value", "N/A")
    final_label = fms.get("label", "N/A")

    typer.echo("\n" + "=" * 60)
    typer.echo("  PIPELINE COMPLETE")
    typer.echo("=" * 60)
    typer.echo(f"  Final score : {final_value}/10 — {final_label}")
    typer.echo(f"  Summary     : {fms.get('summary', '')}")
    typer.echo(f"\n  Full report : {report_path}")
    typer.echo("=" * 60)


if __name__ == "__main__":
    app()
