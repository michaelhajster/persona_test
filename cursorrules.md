# .cursor-rules

###################################################################
# 1) PROJEKT- & CODE-ORGANISATION
###################################################################

- Programmiersprache: **Python** (Version >= 3.8).
- Hauptskripte: `pipeline.py`, `prompt_templates.py`, etc.
- Datenspeicher: CSV, JSON, JSONL (z. B. in `data/` Ordner).
- Dependencies: in `requirements.txt`, Installation via `pip`.

###################################################################
# 2) CODING-CONVENTIONS
###################################################################

- Struktur:
  - `data/` für Datensätze, Wahlprogramme, News, Output
  - `pipeline.py` → Hauptablauf: 4 API-Calls pro Persona
  - `prompt_templates.py` → Enthält Prompt-Strings
  - `requirements.txt` → Packages (openai, pandas, tqdm, dotenv, pydantic, …)
- Bei Nutzung von **Structured Outputs**:
  - Definiere Pydantic-Modelle in einer separaten Datei (z.B. `schemas.py`) oder in `pipeline.py`.
  - Achte darauf, dass alle Felder in den Pydantic-Klassen **required** sind, oder optional mit `Union[str, None]`.
- Kein Hardcoding sensibler Daten (API-Keys) – stattdessen `.env`.

###################################################################
# 3) WORKFLOW-REGELN
###################################################################

- **Nach jeder größeren Task**:
  1. Erzeuge/ändere die relevanten Python-Files, 
  2. teste mind. einmal via `python pipeline.py --sample_size 2` (Kurztest),
  3. Fixe auftretende Fehler,
  4. aktualisiere `.cursor-updates` mit einem Eintrag („Was wurde geändert?“),
  5. Mache einen Git-Commit.

- `.cursor-tasks` wird beim Abarbeiten nur abgehakt (kein Löschen), 
- `.cursor-updates` nach jedem Task-Etappenziel erweitern.

###################################################################
# 4) TOOLS & SCRIPTS
###################################################################

- **`python pipeline.py`**: Führt die 4 Schritte durch.
  - Argumente: `--csv_path`, `--sample_size`, `--output_path`, `--do_analyze`, etc.
- **Test**: Momentan kein PyTest. Ggf. manuelles Testen.

###################################################################
# 5) UMGANG MIT PROMPTS UND MODELLEN
###################################################################

- Standard: GPT-4, GPT-4-32k oder GPT-4o (ggf. via `MODEL_NAME` in `.env`).
- **Structured Outputs** via `client.beta.chat.completions.parse(...)` + Pydantic-Klasse = garantiert valides JSON.
- Fallback: Bei älteren Modellen kann `response_format={"type":"json_object"}` genutzt werden.

###################################################################
# 6) WICHTIGE "DON'Ts"
###################################################################

- Keine Private Keys ins Repo.
- Keine unendlich langen Ausgaben (Token-Limits prüfen).
- Nicht ohne Rückfrage große zusätzliche Libraries einführen.

###################################################################
# 7) ABSCHLUSS
###################################################################
(#) Ruleset für dein Python-Projekt mit Cursor + Structured Outputs.
