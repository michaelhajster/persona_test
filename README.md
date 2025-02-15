# Wahl-O-Mat KI-Projekt

Ein KI-gestütztes System zur Simulation von Wahlentscheidungen basierend auf Persona-Daten. Das System nutzt GPT-4 für eine 4-stufige Analyse-Pipeline.

## Features

- 🤖 KI-gesteuerte Persona-Generierung aus GLES-Daten
- 📊 Automatische Wahl-O-Mat Antworten basierend auf Persona-Profilen
- 🎯 Präzise Partei-Matching-Algorithmen
- 📈 Detaillierte Visualisierungen und Analysen
- ✅ Validierte JSON-Outputs mit Pydantic
- 🔄 Robustes Error-Handling und Retry-Logik

## Setup

### Voraussetzungen

- Python 3.8+
- OpenAI API Key
- Virtuelle Umgebung (empfohlen)

### Installation

1. Repository klonen:
   ```bash
   git clone [repository-url]
   cd wahl-o-mat-ki
   ```

2. Virtuelle Umgebung erstellen und aktivieren:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Unix
   # oder
   .\venv\Scripts\activate  # Windows
   ```

3. Dependencies installieren:
   ```bash
   pip install -r requirements.txt
   ```

4. Umgebungsvariablen konfigurieren:
   ```bash
   cp .env.example .env
   # Füge deinen OpenAI API-Key in .env ein
   ```

## Projektstruktur

```
.
├── data/
│   ├── gles_clean.csv        # Persona-Basisdaten
│   ├── wahlomat_questions.json
│   ├── wahlprogramme/        # Partei-Programme
│   ├── news/                 # Aktuelle News
│   ├── output/              # Pipeline-Ergebnisse
│   └── analysis/            # Generierte Grafiken
├── pipeline.py              # Hauptskript
├── prompt_templates.py      # GPT-Prompts
├── schemas.py              # Pydantic Modelle
├── requirements.txt
├── .env.example
├── .cursor-tasks.md        # Aufgabenliste
├── .cursor-updates.md      # Changelog
└── README.md
```

## Die 4-Stufige Pipeline

1. **Persona-Generierung**
   - Input: GLES-Datensatz
   - Output: Detailliertes Wählerprofil (Name, Alter, Beruf, etc.)
   - Validierung: PersonaSchema

2. **Wahl-O-Mat Simulation**
   - Input: Persona + Wahl-O-Mat Fragen
   - Output: Position (-1/0/1) zu jeder These
   - Validierung: WahlomatSchema

3. **Partei-Matching (Judge)**
   - Input: Wahl-O-Mat Antworten
   - Output: Übereinstimmungswerte für alle Parteien
   - Validierung: JudgeSchema

4. **Finale Wahlentscheidung**
   - Input: Persona + Partei-Matches + News
   - Output: Wahlentscheidung mit Begründung
   - Validierung: FinalChoiceSchema

## Verwendung

### Basis-Ausführung
```bash
python pipeline.py --sample_size 5 --do_analyze
```

### Parameter
- `--sample_size`: Anzahl zu generierender Personas (default: alle)
- `--csv_path`: Pfad zur GLES-Datei (default: data/gles_clean.csv)
- `--output_path`: Pfad für Ergebnisse (default: data/output/pipeline_results.jsonl)
- `--do_analyze`: Aktiviert detaillierte Analyse mit Visualisierungen

### Analyse-Outputs

Die Pipeline generiert drei Visualisierungen in `data/analysis/`:
1. `party_distribution.png`: Stimmenverteilung nach Parteien
2. `match_scores.png`: Übereinstimmungswerte aller Parteien
3. `age_distribution.png`: Wahlverhalten nach Altersgruppen

## Entwicklungs-Dokumentation

- `.cursor-tasks.md`: Detaillierte Aufgabenliste mit Fortschritt
- `.cursor-updates.md`: Chronologischer Changelog aller Änderungen

## Technische Details

- **Structured Outputs**: Verwendung von Pydantic für Type-Safety
- **Error-Handling**: Automatische Retries bei API-Fehlern
- **Logging**: Detailliertes Logging aller Pipeline-Schritte
- **Visualisierung**: Matplotlib/Seaborn für Datenanalyse

## Limitierungen

- Token-Limits bei sehr langen Parteiprogrammen
- API-Kosten bei großen Datensätzen
- Mögliche Bias in den generierten Personas

## Nächste Schritte

- [ ] Token-Limit-Optimierung
- [ ] Code-Refactoring
- [ ] Integration größerer Datensätze
- [ ] Automatisierte Reports 