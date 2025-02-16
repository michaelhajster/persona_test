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
from typing import Dict, List, Optional, Union, Set
from datetime import datetime

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
from party_utils import (PARTY_KEY_TO_LABEL, PARTY_LABEL_TO_KEY,
                        party_key_to_label, party_label_to_key)

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
            "gruene": "gruene.txt",
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
        return f"Aktuelle Nachrichten:\n\n{news_content}"
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

def get_completed_batches(output_path: str) -> Set[int]:
    """Get set of already completed batch numbers."""
    completed = set()
    base_dir = os.path.dirname(output_path)
    if not os.path.exists(base_dir):
        return completed
        
    for filename in os.listdir(base_dir):
        if filename.startswith(os.path.basename(output_path) + ".batch_") and not filename.endswith(".progress"):
            try:
                batch_num = int(filename.split("_")[-1])
                completed.add(batch_num)
            except ValueError:
                continue
    return completed

def load_existing_results(output_path: str, completed_batches: Set[int]) -> List[Dict]:
    """Load results from completed batches."""
    results = []
    for batch_num in sorted(completed_batches):
        batch_path = f"{output_path}.batch_{batch_num}"
        try:
            with open(batch_path, 'r', encoding='utf-8') as f:
                batch_results = [json.loads(line) for line in f]
                results.extend(batch_results)
                logger.info(f"Loaded {len(batch_results)} results from batch {batch_num}")
        except Exception as e:
            logger.error(f"Error loading batch {batch_num}: {e}")
    return results

def run_pipeline(
    csv_path: str,
    sample_size: int,
    output_path: str,
    do_analyze: bool = False,
    batch_size: int = 50,
    resume: bool = True  # New parameter to control resuming
) -> List[Dict]:
    """Run the complete pipeline with batch processing and resume capability."""
    # Load all required data
    df = load_and_sample_data(csv_path, sample_size)
    questions = load_wahlomat_questions()
    programs = load_party_programs()
    news = load_news()
    
    # Calculate total number of batches
    total_batches = len(df) // batch_size + (1 if len(df) % batch_size else 0)
    
    # Get completed batches and load existing results if resuming
    completed_batches = get_completed_batches(output_path) if resume else set()
    results = load_existing_results(output_path, completed_batches) if resume else []
    
    if completed_batches:
        logger.info(f"Found {len(completed_batches)} completed batches: {sorted(completed_batches)}")
        logger.info(f"Loaded {len(results)} existing results")
    
    try:
        # Process remaining batches
        for batch_idx in range(total_batches):
            batch_num = batch_idx + 1
            
            # Skip completed batches
            if batch_num in completed_batches:
                logger.info(f"Skipping completed batch {batch_num}")
                continue
                
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(df))
            batch_df = df.iloc[start_idx:end_idx]
            
            # Create progress file for this batch
            progress_path = f"{output_path}.batch_{batch_num}.progress"
            with open(progress_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "batch": batch_num,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "total_rows": len(batch_df),
                    "timestamp": datetime.now().isoformat()
                }, f, ensure_ascii=False)
            
            logger.info(f"Processing batch {batch_num}/{total_batches} (rows {start_idx}-{end_idx})")
            
            # Process each persona in the batch
            batch_results = []
            for _, row in tqdm(batch_df.iterrows(), total=len(batch_df), desc=f"Batch {batch_num}"):
                try:
                    # Generate persona
                    persona = create_persona_from_row(row, news)
                    
                    # Get Wahl-O-Mat answers
                    answers = ask_wahlomat_questions(persona, questions, programs)
                    
                    # Calculate match percentages
                    match_percentages = judge_wahlomat(answers, programs)
                    
                    # Get final choice
                    final_choice = ask_final_choice(persona, match_percentages, programs, news)
                    
                    # Store results
                    batch_results.append({
                        "persona": persona,
                        "answers": answers,
                        "match_percentages": match_percentages,
                        "final_choice": final_choice
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing row: {e}")
                    continue
            
            # Add batch results to main results
            results.extend(batch_results)
            
            # Save batch results
            batch_path = f"{output_path}.batch_{batch_num}"
            save_results(batch_results, batch_path)
            logger.info(f"Saved batch {batch_num} with {len(batch_results)} results")
            
            # Remove progress file after successful completion
            if os.path.exists(progress_path):
                os.remove(progress_path)
    
    except KeyboardInterrupt:
        logger.info("\nPipeline interrupted by user. Progress has been saved.")
        logger.info(f"To resume, run the pipeline again with the same output path.")
        return results
    
    # Save final combined results
    save_results(results, output_path)
    logger.info(f"Pipeline completed. Total results: {len(results)}")
    
    if do_analyze:
        analyze_results(results)
    
    return results

def analyze_results(results: List[Dict]):
    """Analyze pipeline results with detailed statistics and visualizations."""
    # Setup
    plt.style.use('seaborn-v0_8')
    
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
        for party_key, score in matches.items():
            match_scores[party_key].append(float(score))
            
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
    for party_label, count in sorted(choices.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(results)) * 100
        avg_certainty = np.mean(certainties[party_label])
        party_key = party_label_to_key(party_label)
        avg_match = np.mean([score for score in match_scores[party_key] if score is not None])
        print(f"{party_label:8}: {count:2} Stimmen ({percentage:4.1f}%)")
        print(f"        Durchschn. Sicherheit: {avg_certainty:4.1f}%")
        print(f"        Durchschn. Übereinstimmung: {avg_match:4.1f}%")
    
    print("\nAnalyse-Dateien erstellt in data/analysis/:")
    print("- party_distribution.png: Stimmenverteilung nach Parteien")
    print("- match_scores.png: Verteilung der Übereinstimmungswerte")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Wahl-O-Mat KI Pipeline")
    parser.add_argument(
        "--csv_path",
        default="data/handbereinigt_michi.csv",
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
    parser.add_argument(
        "--batch_size",
        type=int,
        default=50,
        help="Number of personas to process in each batch"
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Don't resume from previous progress"
    )
    
    args = parser.parse_args()
    
    # Run pipeline
    run_pipeline(
        args.csv_path,
        args.sample_size,
        args.output_path,
        args.do_analyze,
        args.batch_size,
        not args.no_resume
    )

if __name__ == "__main__":
    main() 