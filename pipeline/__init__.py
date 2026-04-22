"""
Le Rebond — Sports Commentary Analysis Pipeline.

Modules:
    transcriber : Audio → text via OpenAI Whisper
    analyzer    : LLM-based fact and sentiment extraction via Qwen/Ollama
    scorer      : Coherent match and player rating generation via Qwen/Ollama
    formatter   : Final JSON report assembly
"""
