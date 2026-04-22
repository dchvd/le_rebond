"""
transcriber.py — Audio to text transcription using OpenAI Whisper.

Handles long audio files (> 30 min) by splitting them into chunks with pydub,
transcribing each chunk, and concatenating the results.
"""

import os
import math
import tempfile
from pathlib import Path

from tqdm import tqdm


# Chunk duration in milliseconds (30 minutes)
CHUNK_DURATION_MS = 30 * 60 * 1000


def _load_whisper_model(model_name: str):
    """Load and return a Whisper model by name.

    Args:
        model_name: Whisper model size, e.g. "medium" or "large-v3".

    Returns:
        A loaded whisper model instance.

    Raises:
        RuntimeError: If the model cannot be loaded.
    """
    try:
        import whisper  # type: ignore
        print(f"[Transcriber] Loading Whisper model '{model_name}'...")
        return whisper.load_model(model_name)
    except ImportError as exc:
        raise RuntimeError(
            "openai-whisper is not installed. Run: pip install openai-whisper"
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to load Whisper model '{model_name}': {exc}") from exc


def _split_audio(audio_path: str) -> list[str]:
    """Split an audio file into chunks of CHUNK_DURATION_MS milliseconds.

    Args:
        audio_path: Path to the source audio file (.mp3 or .wav).

    Returns:
        List of paths to temporary chunk files (WAV format).

    Raises:
        RuntimeError: If pydub or ffmpeg is not available.
    """
    try:
        from pydub import AudioSegment  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "pydub is not installed. Run: pip install pydub"
        ) from exc

    try:
        audio = AudioSegment.from_file(audio_path)
    except Exception as exc:
        raise RuntimeError(
            f"Could not read audio file '{audio_path}'. "
            "Make sure ffmpeg is installed and the file format is supported (.mp3, .wav)."
        ) from exc

    duration_ms = len(audio)
    n_chunks = math.ceil(duration_ms / CHUNK_DURATION_MS)

    tmp_dir = tempfile.mkdtemp(prefix="lerebond_chunks_")
    chunk_paths: list[str] = []

    for i in range(n_chunks):
        start = i * CHUNK_DURATION_MS
        end = min(start + CHUNK_DURATION_MS, duration_ms)
        chunk = audio[start:end]
        chunk_path = os.path.join(tmp_dir, f"chunk_{i:03d}.wav")
        chunk.export(chunk_path, format="wav")
        chunk_paths.append(chunk_path)

    return chunk_paths


def _transcribe_single(model, audio_path: str) -> str:
    """Transcribe a single audio file with Whisper.

    Args:
        model: A loaded Whisper model.
        audio_path: Path to the audio file to transcribe.

    Returns:
        The transcribed text string.
    """
    result = model.transcribe(audio_path, verbose=False)
    return result.get("text", "").strip()


def transcribe(audio_path: str, model_name: str, output_dir: str, match_slug: str) -> str:
    """Transcribe an audio file and save the transcript to disk.

    For files longer than 30 minutes the audio is split into chunks that are
    transcribed individually and then concatenated.

    Args:
        audio_path: Path to the input audio file (.mp3 or .wav).
        model_name: Whisper model size ("medium", "large-v3", etc.).
        output_dir: Directory where the transcript text file will be saved.
        match_slug: Safe filename slug derived from the match name.

    Returns:
        The full transcript as a single string.

    Raises:
        RuntimeError: On transcription or I/O failure.
    """
    audio_path = str(Path(audio_path).resolve())
    if not os.path.isfile(audio_path):
        raise RuntimeError(
            f"Audio file not found: '{audio_path}'. "
            "Please verify the path and file extension (.mp3 / .wav)."
        )

    model = _load_whisper_model(model_name)

    try:
        from pydub import AudioSegment  # type: ignore
        audio = AudioSegment.from_file(audio_path)
        duration_ms = len(audio)
    except Exception:
        duration_ms = 0  # unknown duration — attempt single-pass transcription

    if duration_ms > CHUNK_DURATION_MS:
        print(
            f"[Transcriber] Audio is {duration_ms / 60000:.1f} min — splitting into chunks..."
        )
        chunk_paths = _split_audio(audio_path)
        transcripts: list[str] = []

        for chunk_path in tqdm(chunk_paths, desc="Transcribing chunks", unit="chunk"):
            text = _transcribe_single(model, chunk_path)
            transcripts.append(text)
            try:
                os.remove(chunk_path)
            except OSError:
                pass

        full_transcript = "\n\n".join(transcripts)
    else:
        print("[Transcriber] Transcribing audio (single pass)...")
        with tqdm(total=1, desc="Transcribing", unit="file") as pbar:
            full_transcript = _transcribe_single(model, audio_path)
            pbar.update(1)

    # Persist transcript
    os.makedirs(output_dir, exist_ok=True)
    transcript_path = os.path.join(output_dir, f"{match_slug}_transcript.txt")
    with open(transcript_path, "w", encoding="utf-8") as fh:
        fh.write(full_transcript)

    print(f"[Transcriber] Transcript saved → {transcript_path}")
    return full_transcript
