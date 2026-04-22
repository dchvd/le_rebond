---
name: Use conda for Python package installs
description: User prefers conda install over pip install for Python dependencies
type: feedback
---

Always use `conda install` instead of `pip install` when suggesting or running Python package installation commands.

**Why:** User's environment is managed via conda.

**How to apply:** Replace `pip install <packages>` with `conda install <packages>` (or `conda install -c conda-forge <packages>` for packages not in the default channel). For packages only available via pip (like openai-whisper), use `pip install` inside the conda environment as a fallback and note this.
