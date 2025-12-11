

import os
import json
import time
from pathlib import Path
from typing import Dict, Optional, List, Any
import pandas as pd
import yaml
from google import genai
from google.genai import types
from tqdm import tqdm
import logging
from dotenv import load_dotenv
import concurrent.futures
import threading

# Load environment variables from .env file
load_dotenv()


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GeminiLabeler:
    """Handles Gemini API calls for transcript chunk labeling."""
    

    BATCH_SCORING_PROMPT = """
    You are a strict data annotator for an Information Retrieval system.
    Your goal is to evaluate educational video segments with extreme critical rigor.
    
    **CONTEXT:**
    Skill: '{skill_name}'
    Description: {skill_description}
    *Note: The video was found via search, so it is likely ON TOPIC. Your job is to determine if this specific CHUNK contains the teaching value, or if it is just fluff/context.*

    **INPUT CHUNKS:**
    {chunks_text}

    **TASK:**
    Rate EACH chunk individually on 5 dimensions (1-5 scale).
    
    **SCORING RUBRIC (STRICT):**
    
    **1. Relevance to Skill (Search Intent Satisfaction)**
    - 1 (Off-Topic): Unrelated content (intros, ads, other topics).
    - 2 (Tangential): Related concepts (e.g., comparing to other tools) or prerequisites, but not the target skill itself.
    - 3 (On-Topic/Surface): About the skill, but low information (setup, imports, future promises, or vague summaries).
    - 4 (Relevant): Directly explains or demonstrates the skill with concrete info.
    - 5 (Highly Relevant): The core answer. Dense with specific syntax, logic, or active application of the skill.

    **2. Depth (Technical Detail)**
    - 1: Zero technical content (just chat/opinion).
    - 2: Surface level: Defines terms or lists features without explaining them.
    - 3: Standard Tutorial: Shows the "happy path" usage (basic API calls with default settings).
    - 4: Detailed: Explains *parameters*, *options*, or common *pitfalls*. Tells you "how to configure it."
    - 5: Expert: Explains the **underlying mechanics** (math/logic/architecture), **optimization** (speed/efficiency), or **advanced customization** beyond standard docs.

    **3. Clarity (Presentation Quality)**
    - 1: Confusing, rambling, or audio is unintelligible.
    - 2: Hard to follow; speaker backtracks or is disorganized.
    - 3: Standard: Understandable, but conversational/unscripted. (Most YouTube videos are here).
    - 4: Polished: Clear structure, no "umms/ahhs", very direct.
    - 5: Professional: Scripted perfection. Every sentence adds value. No wasted time.

    **4. Practical Examples (Code/Application)**
    - 1: No examples.
    - 2: Abstract/Vague: Describes code verbally ("You would usually loop here...") without showing it.
    - 3: Toy Example: Uses clean, synthetic data (`x = [1,2,3]`, `iris dataset`) or minimal boilerplate code.
    - 4: Applied: Solves a specific, self-contained problem with realistic-looking data.
    - 5: Real-World Scenario: A complex, end-to-end workflow. Handles **messy/raw data**, integrates with other tools (e.g., plotting the model results), or addresses specific edge cases/errors.

    **5. Instructional Language (Pedagogy)**
    - 1: Casual/Vlog style ("So yeah, we do this...").
    - 2: Show-and-Tell ("Watch me type this").
    - 3: Explainer: Explains what they are doing as they do it.
    - 4: Teacher: Anticipates student questions, explains common pitfalls.
    - 5: Professor: Uses analogies, visualizes concepts mentally, and connects current topic to broader ML concepts.

    **CRITICAL CONSTRAINT:**
    - **Grade on a Curve:** A score of 3 means "Average YouTube Tutorial."
    - **Reserve 5s:** Only give a 5 if the chunk is truly exceptional. If you give 5s to everything, you fail the task.

    **RESPONSE FORMAT:**
    Return ONLY a valid JSON LIST of objects. 
    **IMPORTANT:** You must write the "reasoning" field BEFORE the scores to justify your decision.
    [
        {{"chunk_id": int, "reasoning": "string", "traditional_relevance": int, "depth": int, "clarity": int, "practical_examples": int, "instructional_language": int}},
        ...
    ]
    """

    def __init__(
        self, 
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash-lite",
        max_retries: int = 3,
        retry_delay: int = 2
    ):
        """
        Initialize the Gemini labeler.
        
        Args:
            api_key: Google API key (if None, reads from GEMINI_API_KEY or GOOGLE_API_KEY env var)
            model: Model to use (e.g., "gemini-2.5-flash-lite")
            max_retries: Maximum number of retries for failed API calls
            retry_delay: Delay between retries in seconds
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key must be provided or set in GEMINI_API_KEY or GOOGLE_API_KEY environment variable")
        
        # Set the API key in environment for the client
        os.environ["GEMINI_API_KEY"] = self.api_key
        
        # Initialize the Gemini client
        self.client = genai.Client()
        
        self.model_name = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        logger.info(f"Initialized Gemini Labeler with model: {model}")
    
    def score_batch(self, batch_rows: List[Dict[str, Any]], skill_name: str, skill_description: str) -> List[Dict[str, Any]]:
        """
        Score a batch of chunks using Gemini API.
        
        Args:
            batch_rows: List of dictionaries containing chunk data (must have 'chunk_id' and 'text')
            skill_name: Name of the skill
            skill_description: Description of the skill
            
        Returns:
            List of dictionaries with scores for each chunk
        """
        # Format chunks for prompt
        chunks_text_parts = []
        for row in batch_rows:
            chunks_text_parts.append(f"[ID: {row['chunk_id']}] \"{row['text']}\"")
        chunks_text = "\n".join(chunks_text_parts)

        prompt = self.BATCH_SCORING_PROMPT.format(
            skill_name=skill_name,
            skill_description=skill_description,
            chunks_text=chunks_text
        )
        
        for attempt in range(self.max_retries):
            try:
                # Use the new Gemini SDK syntax with configuration
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction="You are an expert educational content analyst. You must respond only with valid JSON list in the exact format specified.",
                        temperature=0.3,
                        response_mime_type="application/json"
                    )
                )
                
                # Check if response has text
                if not response.text:
                    raise ValueError("Empty response from API")
                
              
                results = json.loads(response.text)
                
                if not isinstance(results, list):
                    raise ValueError(f"Expected list response, got {type(results)}")

                # Validate and process results
                processed_results = []
                required_fields = ['chunk_id', 'traditional_relevance', 'depth', 'clarity', 'practical_examples', 'instructional_language', 'reasoning']
                
                # Create a map of chunk_id to original row for easy lookup/validation
                chunk_ids_in_batch = {row['chunk_id'] for row in batch_rows}
                
                for result in results:
                    # Validate fields
                    if not all(field in result for field in required_fields):
                        logger.warning(f"Missing fields in result: {result}")
                        continue
                        
                    # Validate chunk_id exists in batch
                    if result['chunk_id'] not in chunk_ids_in_batch:
                        logger.warning(f"Received score for unknown chunk_id: {result['chunk_id']}")
                        continue

                    # Validate scores are in 1-5 range
                    valid_scores = True
                    for field in required_fields[1:6]: # Skip chunk_id and reasoning
                        if not (1 <= result[field] <= 5):
                            logger.warning(f"Score out of range for {field}: {result[field]}")
                            valid_scores = False
                            break
                    
                    if valid_scores:
                        processed_results.append(result)
                
                return processed_results
                
            except json.JSONDecodeError as e:
                logger.warning(f"Attempt {attempt + 1}/{self.max_retries}: JSON decode error - {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{self.max_retries}: API error - {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
        
        logger.error(f"Failed to score batch after {self.max_retries} attempts")
        return []
    
    def load_skills(self, config_path: Path) -> Dict[str, Dict[str, str]]:
        """
        Load skills from YAML config file.
        
        Args:
            config_path: Path to skills.yml file
            
        Returns:
            Dictionary mapping skill_id to {name, description}
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        skills = {}
        for skill in data.get('skills', []):
            skills[skill['skill_id']] = {
                'name': skill['name'],
                'description': skill['description']
            }
        
        logger.info(f"Loaded {len(skills)} skills from config")
        return skills
    
    def score_chunks_batched(
        self, 
        chunks_df: pd.DataFrame,
        save_interval: int = 500,
        output_path: Optional[Path] = None,
        max_workers: int = 10
    ) -> pd.DataFrame:
        """
        Score chunks in batches using parallel execution.
        
        Args:
            chunks_df: DataFrame with 'chunk_id', 'text', 'video_id', 'skill_name', 'skill_description' columns
            save_interval: Save progress every N chunks
            output_path: Path to save intermediate results
            max_workers: Number of parallel threads to use
            
        Returns:
            DataFrame with chunk scores
        """
        results = []
        total_chunks = len(chunks_df)
        
        logger.info(f"Starting to score {total_chunks} chunks using {self.model_name} with {max_workers} workers")
        
        # Prepare all batches first
        all_batches = []
        
        # Check for existing results to resume
        processed_keys = set()
        if output_path and output_path.exists():
            try:
                existing_df = pd.read_csv(output_path)
                if 'chunk_id' in existing_df.columns and 'video_id' in existing_df.columns and 'traditional_relevance' in existing_df.columns:
                    # FIX: Force conversion to numeric, turning errors into NaN
                    existing_df['traditional_relevance'] = pd.to_numeric(existing_df['traditional_relevance'], errors='coerce')
                    
                    # Only consider chunks with valid scores as processed
                    valid_rows = existing_df[
                        (existing_df['traditional_relevance'].notna()) & 
                        (existing_df['traditional_relevance'] > 0)
                    ]
                    # Create a set of (video_id, chunk_id) tuples
                    processed_keys = set(zip(valid_rows['video_id'], valid_rows['chunk_id']))
                    logger.info(f"Resuming: Found {len(processed_keys)} successfully labeled chunks in {output_path}")
            except Exception as e:
                logger.warning(f"Could not read existing results file: {e}. Starting from scratch.")

        # Group by video_id to keep context together
        grouped = chunks_df.groupby('video_id')
        
        for video_id, group in grouped:
            # Get skill info from the first row of the group
            first_row = group.iloc[0]
            skill_name = first_row['skill_name']
            skill_description = first_row['skill_description']
            skill_id = first_row['skill_id']
            
            # Convert group to list of dicts
            group_rows = group.to_dict('records')
            
            # Create batches of 10
            batch_size = 10
            for i in range(0, len(group_rows), batch_size):
                batch = group_rows[i:i+batch_size]
                
                # Check if ANY chunk in this batch needs processing
                # If all chunks in batch are done, skip the batch
                batch_keys = {(video_id, row['chunk_id']) for row in batch}
                if batch_keys.issubset(processed_keys):
                    continue
                    
                all_batches.append({
                    'batch': batch,
                    'skill_name': skill_name,
                    'skill_description': skill_description,
                    'skill_id': skill_id,
                    'video_id': video_id
                })
        
        total_batches = len(all_batches)
        if total_batches == 0:
            logger.info("All chunks already processed! Nothing to do.")
            if output_path and output_path.exists():
                return pd.read_csv(output_path)
            return pd.DataFrame()

        results_lock = threading.Lock()
        
        def process_batch(batch_info):
            batch = batch_info['batch']
            skill_name = batch_info['skill_name']
            skill_description = batch_info['skill_description']
            skill_id = batch_info['skill_id']
            video_id = batch_info['video_id']
            
            batch_results = []
            
            # Score batch
            batch_scores = self.score_batch(batch, skill_name, skill_description)
            
            # Add metadata to results
            for score in batch_scores:
                score['video_id'] = video_id
                score['skill_id'] = skill_id
                batch_results.append(score)
            
            # Handle missing scores
            scored_chunk_ids = {s['chunk_id'] for s in batch_scores}
            for row in batch:
                if row['chunk_id'] not in scored_chunk_ids:
                    batch_results.append({
                        'chunk_id': row['chunk_id'],
                        'video_id': video_id,
                        'skill_id': skill_id,
                        'traditional_relevance': None,
                        'depth': None,
                        'clarity': None,
                        'practical_examples': None,
                        'instructional_language': None,
                        'reasoning': "Failed to score"
                    })
            
            return batch_results

        # Execute in parallel
        with tqdm(total=total_batches, desc="Scoring batches") as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_batch = {executor.submit(process_batch, batch_info): batch_info for batch_info in all_batches}
                
                for future in concurrent.futures.as_completed(future_to_batch):
                    try:
                        batch_results = future.result()
                        
                        with results_lock:
                            results.extend(batch_results)
                            
                            # Save checkpoint safely (append mode)
                            if output_path:
                                temp_df = pd.DataFrame(batch_results)
                                # Check if file exists to determine if we need header
                                header = not output_path.exists()
                                temp_df.to_csv(output_path, mode='a', header=header, index=False)
                                
                                if len(results) % save_interval < 20:
                                    logger.info(f"Checkpoint: {len(results)} new chunks processed")
                                
                    except Exception as e:
                        logger.error(f"Batch processing failed: {e}")
                    
                    pbar.update(1)
        

        if output_path and output_path.exists():
            results_df = pd.read_csv(output_path)
            logger.info(f"Final results saved to {output_path}")
            
            # CLEANUP: Deduplicate and remove failed attempts
            logger.info("Cleaning up output file (removing duplicates and failed attempts)...")
            
            # Create a helper column for validity
            # Valid if traditional_relevance is not NaN and > 0
            results_df['is_valid'] = results_df['traditional_relevance'].notna() & (pd.to_numeric(results_df['traditional_relevance'], errors='coerce') > 0)
            
            # Sort: Put valid rows LAST so drop_duplicates(keep='last') keeps them
            # If multiple valid rows, keep the last one (most recent)
            results_df = results_df.sort_values(['video_id', 'chunk_id', 'is_valid'], ascending=[True, True, True])
            
            # Deduplicate by ID
            results_df = results_df.drop_duplicates(subset=['video_id', 'chunk_id'], keep='last')
            
            # Drop the helper column
            results_df = results_df.drop(columns=['is_valid'])
            
            # Save the cleaned file
            results_df.to_csv(output_path, index=False)
            logger.info(f"Cleaned file saved. Final count: {len(results_df)}")
            
        else:
            results_df = pd.DataFrame(results)
        
        logger.info(f"Scoring complete. Total chunks in file: {len(results_df)}")
        
        return results_df


def main():
    """Example usage of the Gemini labeler."""
    # Paths
    DATA_DIR = Path(__file__).parent.parent.parent / 'data' / 'processed'
    CONFIG_DIR = Path(__file__).parent.parent.parent / 'config'
    RESULTS_DIR = Path(__file__).parent.parent.parent / 'results' / 'labeling'
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info("Please use run_labeling.py for the full pipeline.")

if __name__ == "__main__":
    main()
