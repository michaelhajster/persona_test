"""
Pydantic Schemas für die Wahl-O-Mat KI Pipeline.
Definiert die Datenstrukturen für:
1. Persona-Generierung
2. Wahl-O-Mat-Antworten
3. Judge (Verteilungslogik)
4. Finale Wahlentscheidung
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class PersonaSchema(BaseModel):
    """Schema für generierte Wähler-Persona."""
    name: str
    alter: int
    beruf: str
    wohnort: str
    beschreibung: str
    politische_einstellung: str
    kernthemen: List[str]
    wahlverhalten: str
    sorgen: List[str]
    hoffnungen: List[str]

    @field_validator("alter")
    def validate_alter(cls, v):
        if not 16 <= v <= 100:
            raise ValueError("Alter muss zwischen 16 und 100 Jahren liegen")
        return v

    @field_validator("kernthemen")
    def validate_kernthemen(cls, v):
        if not 1 <= len(v) <= 5:
            raise ValueError("Kernthemen müssen 1-5 Einträge haben")
        return v


class WahlomatAnswer(BaseModel):
    """Schema für einzelne Wahl-O-Mat Antwort."""
    these_id: int
    position: int  # 1 (Zustimmung), 0 (neutral), -1 (Ablehnung)
    begruendung: str

    @field_validator("position")
    def validate_position(cls, v):
        if v not in [-1, 0, 1]:
            raise ValueError("Position muss -1, 0 oder 1 sein")
        return v


class WahlomatSchema(BaseModel):
    """Schema für alle Wahl-O-Mat Antworten."""
    antworten: List[WahlomatAnswer]


class PartyMatch(BaseModel):
    """Schema für Partei-Übereinstimmungen."""
    cdu_csu: float
    spd: float
    gruene: float
    fdp: float
    linke: float
    afd: float

    @field_validator("*")
    def validate_percentage(cls, v):
        if not 0 <= v <= 100:
            raise ValueError("Prozentwert muss zwischen 0 und 100 liegen")
        return v


class JudgeSchema(BaseModel):
    """Schema für Bewertung der Übereinstimmungen."""
    partei_match: PartyMatch
    analyse: str


class FinalChoiceSchema(BaseModel):
    """Schema für finale Wahlentscheidung."""
    partei_wahl: str
    begruendung: str
    sicherheit: int

    @field_validator("partei_wahl")
    def validate_partei(cls, v):
        valid_parties = {"CDU/CSU", "SPD", "GRÜNE", "FDP", "LINKE", "AfD"}
        if v not in valid_parties:
            raise ValueError(f"Partei muss eine von {valid_parties} sein")
        return v

    @field_validator("sicherheit")
    def validate_sicherheit(cls, v):
        if not 0 <= v <= 100:
            raise ValueError("Sicherheit muss zwischen 0 und 100 liegen")
        return v 