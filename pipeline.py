#!/usr/bin/env python3
"""
Wahl-O-Mat KI Pipeline

Führt die 4-stufige KI-Pipeline aus:
1. Persona-Generierung aus GLES-Daten
2. Wahl-O-Mat Fragen beantworten
3. Verteilungslogik (Judge)
4. Finale Wahlentscheidung
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict

from prompt_templates import (FINAL_CHOICE_TEMPLATE, JUDGE_TEMPLATE,
                            PERSONA_GENERATION_TEMPLATE, WAHLOMAT_TEMPLATE)
from schemas import (FinalChoiceSchema, JudgeSchema, PersonaSchema,
                    WahlomatSchema)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    max_retries=2
)

def call_openai_api_structured(prompt: str, schema: type, max_retries: int = 2) -> Union[PersonaSchema, WahlomatSchema, JudgeSchema, FinalChoiceSchema]:
    """Make an OpenAI API call with structured output."""
    for attempt in range(max_retries):
        try:
            completion = client.beta.chat.completions.parse(
                model=os.getenv("MODEL_NAME", "gpt-4"),
                messages=[
                    {"role": "system", "content": "Du bist ein politischer Analyst. Antworte ausschließlich im spezifizierten Format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                response_format=schema
            )
            
            if completion.choices[0].message.refusal:
                logger.error(f"Model refused: {completion.choices[0].message.refusal}")
                if attempt == max_retries - 1:
                    raise ValueError("Model refused to comply")
                continue
                
            return completion.choices[0].message.parsed
                
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"Final attempt failed: {e}")
                raise
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
    raise ValueError("All attempts failed")

def load_and_sample_data(csv_path: str, sample_size: int) -> pd.DataFrame:
    """Load and sample rows from GLES dataset."""
    try:
        df = pd.read_csv(csv_path)
        if sample_size > 0:
            return df.sample(n=min(sample_size, len(df)))
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def load_wahlomat_questions(path: str = "data/wahlomat_questions.json") -> List[Dict]:
    """Load Wahl-O-Mat questions from JSON file."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading Wahl-O-Mat questions: {e}")
        raise

def load_party_programs(directory: str = "data/wahlprogramme") -> Dict[str, str]:
    """Load party programs from text files."""
    programs = {}
    try:
        # Define the specific files we want to load
        party_files = {
            "cdu_csu": "cdu_csu.txt",
            "spd": "spd.txt",
            "gruene": "grüne.txt",
            "fdp": "fdp.txt",
            "linke": "linke.txt",
            "afd": "afd.txt"
        }
        
        for party, filename in party_files.items():
            file_path = Path(directory) / filename
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    programs[party] = f.read()
            else:
                logger.warning(f"Party program file not found: {filename}")
        
        return programs
    except Exception as e:
        logger.error(f"Error loading party programs: {e}")
        raise

def load_news(path: str = "data/news/news_kuratiert.txt") -> str:
    """Load and process current news data."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            news_content = f.read()
            
        # Process the news content to create a more focused summary
        # Split into sections and take the most recent/relevant parts
        sections = news_content.split("##")
        
        # Get the most recent month's news (last complete section)
        recent_news = sections[-2] if len(sections) > 2 else sections[-1]
        
        # Extract key points
        key_points = []
        for line in recent_news.split("\n"):
            if line.strip().startswith("- "):
                # Clean up markdown formatting
                clean_line = line.replace("**", "").replace("_", "").strip("- ")
                key_points.append(clean_line)
        
        # Create a focused summary
        summary = "Aktuelle Nachrichten:\n\n"
        summary += "\n".join(f"- {point}" for point in key_points)
        
        return summary
            
    except Exception as e:
        logger.error(f"Error loading news: {e}")
        return "Keine aktuellen Nachrichten verfügbar."

def create_persona_from_row(row: pd.Series, news: str) -> PersonaSchema:
    """Generate persona description from GLES data row."""
    # Convert row to a more readable format
    data = {
        "alter": row["Date of Birth: Year"],
        "geschlecht": row["Gender"],
        "bildung": row["Education: School"],
        "beruf": row["Gainful Employment: current"],
        "bundesland": row["Federal State"],
        "wohnort_groesse": "Not available",  # This column seems to be missing in the new data
        "einkommen": row["Household Net Income"],
        "migration": "Not available",  # This column seems to be missing in the new data
        "religion": row["Religious Denomination"],
        "politisches_interesse": row["Political Interest"],
        "links_rechts": row["Left-Right Assessment: Ego"],
        "demokratie_zufriedenheit": row["Democracy: Satisfaction (5-point Scale)"],
        "letzte_wahl": row["Party Identification (Version A)"]
    }
    
    prompt = PERSONA_GENERATION_TEMPLATE.format(
        gles_data=json.dumps(data, ensure_ascii=False, indent=2),
        news=news
    )
    return call_openai_api_structured(prompt, PersonaSchema)

def ask_wahlomat_questions(
    persona: PersonaSchema,
    questions: List[Dict],
    programs: Dict[str, str]
) -> WahlomatSchema:
    """Get persona's answers to Wahl-O-Mat questions."""
    prompt = WAHLOMAT_TEMPLATE.format(
        persona_json=persona.model_dump_json(indent=2),
        party_programs="\n\n".join(f"{party}:\n{program}" for party, program in programs.items()),
        questions=json.dumps(questions, ensure_ascii=False, indent=2)
    )
    return call_openai_api_structured(prompt, WahlomatSchema)

def judge_wahlomat(
    answers: WahlomatSchema,
    programs: Dict[str, str]
) -> JudgeSchema:
    """Calculate party match percentages based on answers."""
    prompt = JUDGE_TEMPLATE.format(
        answers_json=answers.model_dump_json(indent=2),
        party_programs="\n\n".join(f"{party}:\n{program}" for party, program in programs.items())
    )
    return call_openai_api_structured(prompt, JudgeSchema)

def ask_final_choice(
    persona: PersonaSchema,
    match_percentages: JudgeSchema,
    programs: Dict[str, str],
    news: str
) -> FinalChoiceSchema:
    """Get final voting decision from persona."""
    prompt = FINAL_CHOICE_TEMPLATE.format(
        persona_json=persona.model_dump_json(indent=2),
        match_percentages=match_percentages.model_dump_json(indent=2),
        party_programs="\n\n".join(f"{party}:\n{program}" for party, program in programs.items()),
        news=news
    )
    result = call_openai_api_structured(prompt, FinalChoiceSchema)
    return result

def save_results(results: List[Dict], output_path: str):
    """Save pipeline results as JSONL."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            # Convert Pydantic models to dict for JSON serialization
            result_dict = {
                "persona": result["persona"].model_dump(),
                "answers": result["answers"].model_dump(),
                "match_percentages": result["match_percentages"].model_dump(),
                "final_choice": result["final_choice"].model_dump()
            }
            f.write(json.dumps(result_dict, ensure_ascii=False) + '\n')

def run_pipeline(
    csv_path: str,
    sample_size: int,
    output_path: str,
    do_analyze: bool = False
) -> List[Dict]:
    """Run the complete pipeline."""
    # Load all required data
    df = load_and_sample_data(csv_path, sample_size)
    questions = load_wahlomat_questions()
    programs = load_party_programs()
    news = load_news()
    
    results = []
    
    # Process each persona
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing personas"):
        try:
            # Step 1: Create persona
            persona = create_persona_from_row(row, news)
            logger.info(f"Created persona: {persona.name}")
            
            # Step 2: Get Wahl-O-Mat answers
            answers = ask_wahlomat_questions(persona, questions, programs)
            logger.info(f"Got {len(answers.antworten)} Wahl-O-Mat answers")
            
            # Step 3: Judge answers
            match_percentages = judge_wahlomat(answers, programs)
            logger.info(f"Calculated party matches: {match_percentages.partei_match}")
            
            # Step 4: Get final choice
            final_choice = ask_final_choice(persona, match_percentages, programs, news)
            logger.info(f"Final choice: {final_choice.partei_wahl} (Sicherheit: {final_choice.sicherheit}%)")
            
            # Store results
            result = {
                "persona": persona,
                "answers": answers,
                "match_percentages": match_percentages,
                "final_choice": final_choice
            }
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error processing row: {e}")
            continue
    
    # Save results
    save_results(results, output_path)
    logger.info(f"Saved results to {output_path}")
    
    if do_analyze:
        analyze_results(results)
    
    return results

def analyze_results(results: List[Dict]):
    """Analyze pipeline results with detailed statistics and visualizations."""
    # Setup
    plt.style.use('seaborn-v0_8')
    
    # Party name mapping
    party_map = {
        'CDU/CSU': 'cdu_csu',
        'SPD': 'spd',
        'GRÜNE': 'gruene',
        'FDP': 'fdp',
        'LINKE': 'linke',
        'AfD': 'afd'
    }
    
    # 1. Party Choice Analysis
    choices = defaultdict(int)
    certainties = defaultdict(list)
    match_scores = defaultdict(list)
    age_groups = defaultdict(lambda: defaultdict(int))
    
    for result in results:
        # Final choices
        choice = result["final_choice"].partei_wahl
        choices[choice] += 1
        certainties[choice].append(result["final_choice"].sicherheit)
        
        # Match scores from model dump
        matches = result["match_percentages"].partei_match.model_dump()
        for party, score in matches.items():
            match_scores[party].append(float(score))
            
        # Age demographics
        age = result["persona"].alter
        age_group = f"{(age // 10) * 10}-{(age // 10) * 10 + 9}"
        age_groups[age_group][choice] += 1
    
    # Create output directory
    os.makedirs("data/analysis", exist_ok=True)
    
    # 2. Visualization: Party Distribution
    plt.figure(figsize=(12, 6))
    parties = list(choices.keys())
    votes = list(choices.values())
    percentages = [v/len(results)*100 for v in votes]
    
    bars = plt.bar(parties, percentages)
    plt.title("Stimmenverteilung nach Parteien")
    plt.ylabel("Prozent der Stimmen")
    
    # Add percentage labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom')
    
    plt.savefig("data/analysis/party_distribution.png")
    plt.close()
    
    # 3. Visualization: Match Score Distribution
    plt.figure(figsize=(12, 6))
    box_data = [scores for scores in match_scores.values()]
    plt.boxplot(box_data, tick_labels=match_scores.keys())
    plt.title("Verteilung der Übereinstimmungswerte")
    plt.ylabel("Übereinstimmung (%)")
    plt.xticks(rotation=45)
    plt.savefig("data/analysis/match_scores.png")
    plt.close()
    
    # 4. Visualization: Age Distribution
    age_data = []
    for age_group in sorted(age_groups.keys()):
        for party in parties:
            age_data.append({
                'Altersgruppe': age_group,
                'Partei': party,
                'Anzahl': age_groups[age_group][party]
            })
    
    plt.figure(figsize=(12, 6))
    age_df = pd.DataFrame(age_data)
    age_pivot = age_df.pivot(index='Altersgruppe', columns='Partei', values='Anzahl')
    age_pivot.plot(kind='bar', stacked=True)
    plt.title("Wahlverhalten nach Altersgruppen")
    plt.ylabel("Anzahl Wähler")
    plt.legend(title="Partei", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("data/analysis/age_distribution.png")
    plt.close()
    
    # 5. Print Summary Statistics
    print("\nWahl-Analyse:")
    print("-" * 40)
    print("\nStimmenverteilung:")
    for party, count in sorted(choices.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(results)) * 100
        avg_certainty = np.mean(certainties[party])
        party_key = party_map.get(party, party.lower())
        avg_match = np.mean([score for score in match_scores[party_key] if score is not None])
        print(f"{party:8}: {count:2} Stimmen ({percentage:4.1f}%)")
        print(f"        Durchschn. Sicherheit: {avg_certainty:4.1f}%")
        print(f"        Durchschn. Übereinstimmung: {avg_match:4.1f}%")
    
    print("\nAnalyse-Dateien erstellt in data/analysis/:")
    print("- party_distribution.png: Stimmenverteilung nach Parteien")
    print("- match_scores.png: Verteilung der Übereinstimmungswerte")
    print("- age_distribution.png: Wahlverhalten nach Altersgruppen")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Wahl-O-Mat KI Pipeline")
    parser.add_argument(
        "--csv_path",
        default="data/gles_clean.csv",
        help="Path to GLES dataset"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=0,
        help="Number of personas to process (0 = all)"
    )
    parser.add_argument(
        "--output_path",
        default="data/output/pipeline_results.jsonl",
        help="Path for output JSONL file"
    )
    parser.add_argument(
        "--do_analyze",
        action="store_true",
        help="Analyze results after pipeline completion"
    )
    
    args = parser.parse_args()
    
    # Run pipeline
    run_pipeline(
        args.csv_path,
        args.sample_size,
        args.output_path,
        args.do_analyze
    )

if __name__ == "__main__":
    main() 