# .cursor-tasks

---
title: "Tasks für das Wahl-O-Mat-KI-Projekt"
description: |
  Diese Liste enthält alle wesentlichen Schritte, um die 
  4-stufige KI-Pipeline end-to-end aufzusetzen und zu testen.
  Bitte jeweils abhaken und anschließend in .cursor-updates eintragen.
---

## 1. PROJEKTSETUP

### 1.1 [x] Initiales Repository & Grundstruktur
- `requirements.txt` mit openai, pandas, tqdm, python-dotenv
- `data/` Ordner anlegen, z. B. für:
  - `gles_clean.csv`
  - `wahlomat_questions.json`
  - `wahlprogramme/` (Partei-Texte)
  - `news/` (Datei mit aktuellen News)
  - `output/` (Abschluss-Ergebnisse .jsonl)
- `.env.example` anlegen für `OPENAI_API_KEY`.

### 1.2 [x] Skriptgerüst: pipeline.py
- Lege Funktionsrümpfe an: 
  - load_and_sample_data()
  - load_wahlomat_questions()
  - load_party_programs()
  - load_news()
  - create_persona_from_row()
  - ask_wahlomat_questions()
  - judge_wahlomat()
  - ask_final_choice()
  - run_pipeline() (mit Loop über n Personas)
- CLI-Parser (argparse) mit Parametern (sample_size, etc.).

### 1.3 [x] prompt_templates.py
- Enthält 4 Prompts:
  1. Persona-Generierung
  2. Wahl-O-Mat-Antworten
  3. Judge (Verteilungs-Logik)
  4. Finale Wahlentscheidung

### 1.4 [x] Testdaten anlegen
- Dummy CSV (2–5 Zeilen) => `gles_clean.csv`
- `wahlomat_questions.json` => Mind. 3–5 Thesen
- `data/wahlprogramme/` => cdu_csu_kurz.txt, spd_kurz.txt, etc.
- `data/news/daily_news_test.txt` => Beliebige Schlagzeilen

### 1.5 [x] Kurzer Testlauf
- `python pipeline.py --sample_size 2 --do_analyze`
- Ggf. Output in `data/output/pipeline_results.jsonl`.
- Prüfen, ob JSON valide.


## 2. FEINTUNING / CODE-VERBESSERUNGEN

### 2.1 [x] Exception Handling bei JSONDecodeError
- Fallback, ggf. Retrys (1–2 Versuche).
- Logik in `call_openai_api()` o.ä.

#### 2.1a [x] **Integration „Structured Outputs" via Pydantic**
- Erstelle `schemas.py` (o.ä.) mit Pydantic-Klassen (z.B. `PersonaSchema`, `WahlomatSchema` etc.).
- Ändere `call_openai_api()` → nutze `client.beta.chat.completions.parse(..., response_format=PersonaSchema)`.
- Stelle sicher, dass alle Felder required oder optional sind (mit Union).
- Testen mit GPT-4o oder GPT-4-2024-08-06 (o.ä.).

### 2.2 [ ] Token-Limit-Prüfung
- Optional: Kürze zu lange Partei-Programme.

### 2.3 [ ] Code-Refactoring
- Strukturen vereinfachen, Variablennamen, docstrings.

## 3. ANALYSE & DOKU

### 3.1 [x] Analyse-Funktion 
- Auswertung: Wie oft welche Partei gewählt?
- Detaillierte Statistiken (Sicherheit, Übereinstimmung)
- Visualisierungen via matplotlib/seaborn:
  - Stimmenverteilung (Balkendiagramm)
  - Übereinstimmungswerte (Boxplot)
  - Altersgruppen-Analyse (Gestapeltes Balkendiagramm)

### 3.2 [x] README aktualisieren
- Ausführliche Anleitung (Installation, Ausführen).
- Erkläre die 4 API-Calls (Persona -> Wahlomat -> Judge -> Finale Wahl).
- Erwähne `.cursor-rules`, `.cursor-tasks`, `.cursor-updates`.

### 3.3 [ ] Deployment / Showcase
- Optional: HPC-Cluster / GPU-Server / Notebook
- Ggf. Vorführung / Demo.

## 4. (Optional) WEITERENTWICKLUNG

### 4.1 [ ] Real-Datensatz einbinden
- Größere Menge (z. B. 300–1000 Personas).
- Achte auf OpenAI-Kosten.

### 4.2 [ ] Zusätzliche Features / Edge Cases
- Mehr News-Injektionen
- Party-Splitting (CDU vs CSU einzeln?)
- Persona-Bias-Tests

### 4.3 [ ] Automatisierte Reports
- Täglicher Run? 
- Summaries in Slack/Webhook?
