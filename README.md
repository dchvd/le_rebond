# Le Rebond — Sports Commentary Analysis Pipeline

Analyse des commentaires sportifs audio ou texte pour extraire les éléments positifs/négatifs et générer une note cohérente.

## Stack

| Composant | Outil |
|---|---|
| Transcription audio | OpenAI Whisper (local) |
| Analyse LLM | Qwen 2.5 via Ollama (local) |
| CLI | Typer |
| Sortie | JSON |

## Installation

### 1. Dépendances système

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg
```

### 2. Dépendances Python

```bash
pip install -r requirements.txt
```

### 3. Ollama + modèle Qwen

```bash
# Installer Ollama : https://ollama.com
ollama pull qwen2.5:3b
# ou pour un meilleur modèle :
ollama pull qwen2.5:7b
```

### 4. Configuration

```bash
cp .env.example .env
# Éditer .env si nécessaire
```

## Utilisation

```bash
# Entrée texte
python main.py --input commentary.txt --match "PSG vs Barcelona" --date "2025-04-15"

# Entrée audio
python main.py --input match.mp3 --match "PSG vs Barcelona" --date "2025-04-15" --sport football

# Dossier de sortie personnalisé
python main.py --input match.mp3 --match "PSG vs Barcelona" --output-dir ./my_reports

# Aide
python main.py --help
```

## Options CLI

| Option | Description | Défaut |
|---|---|---|
| `--input` | Fichier audio (.mp3/.wav) ou texte (.txt/.srt) | requis |
| `--match` | Nom du match | requis |
| `--date` | Date du match | "" |
| `--sport` | Sport (football/basketball/rugby) | football |
| `--output-dir` | Dossier de sortie | `./outputs` |

## Structure des sorties

```
outputs/
├── {match}_transcript.txt       # Transcription audio (si audio en entrée)
├── {match}_analysis.json        # Analyse LLM brute
├── {match}_scores.json          # Notes générées
└── {match}_full_report.json     # Rapport complet (master)
```

## Structure du rapport final

```json
{
  "meta": { "pipeline": "Le Rebond", "generated_at": "...", "input_file": "..." },
  "match": { "name": "...", "date": "...", "sport": "...", "teams": [], "final_score": "" },
  "key_events": [...],
  "positive_elements": [...],
  "negative_elements": [...],
  "performance_by_team": { "team_a": {...}, "team_b": {...} },
  "commentary_sentiment": { "overall_tone": "...", "excitement_level": 0, "controversy_level": 0 },
  "raw_summary": "...",
  "ratings": {
    "match_rating": { "overall_quality": {...}, "entertainment_value": {...}, "competitiveness": {...} },
    "team_ratings": { "team_a": {...}, "team_b": {...} },
    "player_ratings": [...],
    "final_match_score": { "value": 7.5, "label": "Tres bon", "summary": "..." }
  }
}
```

## Pipeline

```
Input (audio/texte)
    │
    ▼
[Étape 1] Transcription (Whisper)  ← audio uniquement
    │
    ▼
[Étape 2] Analyse LLM (Qwen/Ollama)
    │  → extraction faits, événements, éléments +/-
    ▼
[Étape 3] Génération des notes (Qwen/Ollama)
    │  → match, équipes, joueurs
    ▼
[Étape 4] Assemblage rapport final (JSON)
```

## Variables d'environnement (.env)

```env
WHISPER_MODEL=medium          # medium | large-v3
OLLAMA_BASE_URL=http://localhost:11434
QWEN_MODEL=qwen2.5:3b         # qwen2.5:3b | qwen2.5:7b
MAX_CHUNK_TOKENS=3000
OUTPUT_DIR=./outputs
```
