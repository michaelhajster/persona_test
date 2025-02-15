"""
Prompt Templates für die Wahl-O-Mat KI Pipeline.
Enthält die vier Haupt-Prompts für:
1. Persona-Generierung
2. Wahl-O-Mat-Antworten
3. Judge (Verteilungslogik)
4. Finale Wahlentscheidung
"""

# Template für Persona-Generierung aus GLES-Daten
PERSONA_GENERATION_TEMPLATE = '''Du bist ein Experte für die Erstellung von detaillierten Wählerprofilen.
Basierend auf den folgenden demographischen und politischen Daten, erstelle eine realistische Persona.
Antworte AUSSCHLIESSLICH mit einem validen JSON-Objekt.

GLES-Daten:
{gles_data}

Aktuelle Nachrichten:
{news}

Erstelle eine Persona mit folgenden Eigenschaften:
1. Name (Vorname Nachname)
2. Alter (numerisch)
3. Beruf (aktueller Beruf)
4. Wohnort (Stadt, Bundesland)
5. Beschreibung (2-3 Sätze zur Person)
6. Politische Einstellung (kurze Beschreibung)
7. Kernthemen (Array mit 3 wichtigsten politischen Themen)
8. Wahlverhalten (bisheriges Wahlverhalten)
9. Sorgen (Array mit aktuellen Sorgen)
10. Hoffnungen (Array mit Hoffnungen)

Das JSON-Objekt MUSS exakt diesem Format folgen:
{{
    "name": "Max Mustermann",
    "alter": 42,
    "beruf": "Lehrer",
    "wohnort": "München, Bayern",
    "beschreibung": "Engagierter Lehrer an einer Gesamtschule. Setzt sich für Bildungsgerechtigkeit ein.",
    "politische_einstellung": "Sozialliberal mit Fokus auf Bildung und Chancengleichheit",
    "kernthemen": [
        "Bildungsreform",
        "Soziale Gerechtigkeit",
        "Digitalisierung"
    ],
    "wahlverhalten": "Wechselwähler zwischen SPD und Grünen",
    "sorgen": [
        "Zunehmende soziale Spaltung",
        "Bildungsrückstand durch Corona"
    ],
    "hoffnungen": [
        "Bessere Ausstattung der Schulen",
        "Mehr gesellschaftlicher Zusammenhalt"
    ]
}}'''

# Template für Wahl-O-Mat-Antworten
WAHLOMAT_TEMPLATE = '''Du bist ein politischer Analyst, der die Antworten einer Wählerperson auf Wahl-O-Mat-Fragen vorhersagt.
Berücksichtige dabei das Profil der Person und die Parteiprogramme.
Antworte AUSSCHLIESSLICH mit einem validen JSON-Objekt.

Persona:
{persona_json}

Parteiprogramme (Auszüge):
{party_programs}

Beantworte die folgenden Wahl-O-Mat-Thesen aus Sicht der Persona.
Für jede These:
1. Stimme zu (+1)
2. Neutral (0)
3. Stimme nicht zu (-1)

Thesen:
{questions}

Das JSON-Objekt MUSS exakt diesem Format folgen:
{{
    "antworten": [
        {{
            "these_id": 1,
            "position": 1,
            "begruendung": "Kurze Begründung warum die Person zustimmt"
        }},
        {{
            "these_id": 2,
            "position": 0,
            "begruendung": "Kurze Begründung warum die Person neutral ist"
        }},
        {{
            "these_id": 3,
            "position": -1,
            "begruendung": "Kurze Begründung warum die Person nicht zustimmt"
        }}
    ]
}}

WICHTIG:
- Jede These muss beantwortet werden
- Position MUSS -1, 0 oder 1 sein
- these_id muss mit der ID aus den Thesen übereinstimmen
- Begründung sollte zur Persona und deren politischer Einstellung passen'''

# Template für Judge (Verteilungslogik)
JUDGE_TEMPLATE = '''Du bist ein unparteiischer Wahlexperte.
Basierend auf den Antworten einer Person im Wahl-O-Mat und den Parteiprogrammen,
berechne die prozentuale Übereinstimmung mit jeder Partei.
Antworte AUSSCHLIESSLICH mit einem validen JSON-Objekt.

Wahl-O-Mat-Antworten der Person:
{answers_json}

Parteiprogramme:
{party_programs}

Analysiere die Übereinstimmungen und erstelle eine gewichtete Bewertung.
Berücksichtige dabei:
1. Direkte Übereinstimmungen bei den Antworten
2. Inhaltliche Nähe bei den Begründungen
3. Kernthemen der Parteien

Das JSON-Objekt MUSS exakt diesem Format folgen und ALLE Parteien enthalten:
{{
    "partei_match": {{
        "cdu_csu": 75.5,
        "spd": 65.2,
        "gruene": 45.8,
        "fdp": 35.5,
        "linke": 25.2,
        "afd": 15.8
    }},
    "analyse": "Kurze Analyse der Übereinstimmungen und möglicher Präferenzen"
}}

WICHTIG:
- ALLE Parteien müssen einen Prozentwert zwischen 0 und 100 haben
- Auch Parteien mit geringer Übereinstimmung müssen einen Wert haben
- Die Analyse sollte die stärksten Übereinstimmungen hervorheben'''

# Template für finale Wahlentscheidung
FINAL_CHOICE_TEMPLATE = '''Du bist ein Experte für Wahlverhalten.
Basierend auf dem Profil einer Person, den Parteipräferenzen aus dem Wahl-O-Mat
und aktuellen Nachrichten, sage voraus, welche Partei die Person wählen würde.
Antworte AUSSCHLIESSLICH mit einem validen JSON-Objekt.

Persona:
{persona_json}

Wahl-O-Mat Übereinstimmungen:
{match_percentages}

Parteiprogramme (Auszüge):
{party_programs}

Aktuelle Nachrichten:
{news}

Berücksichtige:
1. Wahl-O-Mat-Ergebnisse
2. Bisheriges Wahlverhalten
3. Aktuelle Nachrichten/Ereignisse
4. Taktisches Wahlverhalten

Das JSON-Objekt MUSS exakt diesem Format folgen:
{{
    "partei_wahl": "CDU/CSU",  # MUSS eine der folgenden sein: CDU/CSU, SPD, GRÜNE, FDP, LINKE, AfD
    "begruendung": "Kurze Begründung der Wahlentscheidung basierend auf Profil und Präferenzen",
    "sicherheit": 85  # Prozentwert zwischen 0 und 100
}}

WICHTIG:
- partei_wahl muss exakt einer der vorgegebenen Parteinamen entsprechen
- sicherheit muss zwischen 0 und 100 liegen
- Die Begründung sollte Wahl-O-Mat-Ergebnisse und aktuelle Ereignisse berücksichtigen''' 