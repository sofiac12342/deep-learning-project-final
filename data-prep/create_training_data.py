"""


This script transforms raw Gemini labels into a training dataset with:
- Normalized scores (0-1 scale)
- Weighted composite label
- Hard Negative mining (based on SBERT scores)
- 5-Fold Cross-Validation splits (grouped by video_id)

Usage:
    python src/data_prep/create_training_data.py
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from sklearn.model_selection import GroupKFold
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_skills(skills_path: Path) -> dict:
    """Load skill definitions from YAML config."""
    with open(skills_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    skill_map = {}
    for skill in config['skills']:
        skill_id = skill['skill_id']
        query_text = f"{skill['name']}: {skill['description']}"
        skill_map[skill_id] = query_text
    
    return skill_map


def normalize_score(score: float, min_val: float = 1.0, max_val: float = 5.0) -> float:
    """Normalize score from original scale to 0-1 range."""
    return (score - min_val) / (max_val - min_val)


def calculate_label(row: pd.Series, weights: dict = None) -> float:
    if weights is None:
        weights = {
        'traditional_relevance': 0.40,
        'depth': 0.25,
        'practical_examples': 0.20,
        'clarity': 0.10,
        'instructional_language': 0.05
    }
    
    label = sum(
        weights[col] * row[f'{col}_norm'] 
        for col in weights.keys()
    )
    
    return label


def main():
    BASE_DIR = Path(__file__).parent.parent.parent
    
    # Input paths
    LABELED_CHUNKS_PATH = BASE_DIR / 'data' / 'processed' / 'labeled_chunks.csv'
    TRANSCRIPT_CHUNKS_PATH = BASE_DIR / 'data' / 'processed' / 'transcript_chunks.csv'
    SBERT_SCORES_PATH = BASE_DIR / 'results' / 'baselines' / 'sbert_chunk_scores.csv'
    SKILLS_PATH = BASE_DIR / 'config' / 'skills.yml'
    
    # Output path
    OUTPUT_DIR = BASE_DIR / 'data' / 'training'
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH = OUTPUT_DIR / 'train_data_5folds.csv'
    
    logger.info("Loading data...")
    
    # Load labeled chunks (contains Gemini scores)
    labeled_df = pd.read_csv(LABELED_CHUNKS_PATH)
    logger.info(f"Loaded {len(labeled_df)} labeled chunks")
    
    # Load transcript chunks (contains the actual text)
    chunks_df = pd.read_csv(TRANSCRIPT_CHUNKS_PATH)
    logger.info(f"Loaded {len(chunks_df)} transcript chunks")
    
    # Load skill descriptions
    skill_map = load_skills(SKILLS_PATH)
    logger.info(f"Loaded {len(skill_map)} skill definitions")
    
    #merge
    logger.info("Merging datasets...")
    
  
    df = labeled_df.merge(
        chunks_df[['chunk_id', 'video_id', 'text']],
        on=['chunk_id', 'video_id'],
        how='inner'
    )
    
    logger.info(f"After merge: {len(df)} rows")
    

    df['query_text'] = df['skill_id'].map(skill_map)
    
    # Drop rows with missing skill descriptions
    missing_skills = df['query_text'].isna().sum()
    if missing_skills > 0:
        logger.warning(f"Dropping {missing_skills} rows with unknown skill_id")
        df = df.dropna(subset=['query_text'])
    
  
    logger.info("Cleaning data...")
    
    # Drop rows where scoring failed (missing traditional_relevance)
    initial_count = len(df)
    df = df.dropna(subset=['traditional_relevance'])
    df = df[df['traditional_relevance'] > 0]
    logger.info(f"Dropped {initial_count - len(df)} rows with missing/invalid scores")

    logger.info("Normalizing scores to 0-1 scale...")
    
    score_columns = ['traditional_relevance', 'depth', 'clarity', 'practical_examples', 'instructional_language']
    
    for col in score_columns:
        df[f'{col}_norm'] = df[col].apply(normalize_score)
    

    logger.info("Calculating composite label...")
    
    df['label'] = df.apply(calculate_label, axis=1)
    

    logger.info("Mining hard negatives...")
    
    df['is_hard_negative'] = False
    
    if SBERT_SCORES_PATH.exists():
        try:
            # Load SBERT scores (might be large, so we only load needed columns)
            # Note: SBERT file uses 'sbert_similarity' as the score column name
            sbert_df = pd.read_csv(SBERT_SCORES_PATH, usecols=['video_id', 'chunk_id', 'skill_id', 'sbert_similarity'])
            logger.info(f"Loaded {len(sbert_df)} SBERT chunk scores")
            
            # Calculate top 25% threshold for SBERT scores
            sbert_threshold = sbert_df['sbert_similarity'].quantile(0.75)
            logger.info(f"SBERT top 25% threshold: {sbert_threshold:.4f}")
            
            # Merge SBERT scores
            df = df.merge(
                sbert_df[['video_id', 'chunk_id', 'skill_id', 'sbert_similarity']].rename(columns={'sbert_similarity': 'sbert_score'}),
                on=['video_id', 'chunk_id', 'skill_id'],
                how='left'
            )
            
            # Hard negative: High SBERT score BUT low Gemini label
            # These are chunks that look relevant to SBERT but aren't actually good quality
            df['is_hard_negative'] = (
                (df['sbert_score'] >= sbert_threshold) & 
                (df['label'] < 0.4)
            )
            
            hard_neg_count = df['is_hard_negative'].sum()
            logger.info(f"Found {hard_neg_count} hard negatives ({100*hard_neg_count/len(df):.1f}%)")
            
            # Drop the sbert_score column (no longer needed)
            df = df.drop(columns=['sbert_score'], errors='ignore')
            
        except Exception as e:
            logger.warning(f"Could not load SBERT scores: {e}")
            logger.warning("Proceeding without hard negative mining")
    else:
        logger.warning(f"SBERT scores file not found at {SBERT_SCORES_PATH}")
        logger.warning("Proceeding without hard negative mining")
    
    # ===== 5-FOLD CROSS-VALIDATION SPLIT =====
    logger.info("Creating 5-fold cross-validation splits...")
    
 
    # This prevents data leakage!
    gkf = GroupKFold(n_splits=5)
    
    # Initialize fold column
    df['fold'] = -1
    
    # Assign folds based on video_id groups
    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(df, groups=df['video_id'])):
        df.iloc[val_idx, df.columns.get_loc('fold')] = fold_idx
    
    # Verify fold distribution
    fold_counts = df['fold'].value_counts().sort_index()
    logger.info(f"Fold distribution:\n{fold_counts}")
    
    # ===== SELECT OUTPUT COLUMNS =====
    output_columns = [
        'video_id',
        'chunk_id',
        'skill_id',
        'query_text',
        'text',
        'label',
        'fold',
        'is_hard_negative'
    ]
    
    # Keep only the columns we need
    df_output = df[output_columns].copy()
    
    # ===== SAVE OUTPUT =====
    logger.info(f"Saving training data to {OUTPUT_PATH}...")
    df_output.to_csv(OUTPUT_PATH, index=False)
    
    # ===== LOG STATISTICS =====
    print("\n" + "="*60)
    print("TRAINING DATA CREATION COMPLETE")
    print("="*60)
    print(f"\nDataset Statistics:")
    print(f"   Total rows: {len(df_output):,}")
    print(f"   Unique videos: {df_output['video_id'].nunique():,}")
    print(f"   Unique skills: {df_output['skill_id'].nunique():,}")
    
    print(f"\n Label Distribution:")
    print(f"   Min:    {df_output['label'].min():.3f}")
    print(f"   Max:    {df_output['label'].max():.3f}")
    print(f"   Mean:   {df_output['label'].mean():.3f}")
    print(f"   Median: {df_output['label'].median():.3f}")
    print(f"   Std:    {df_output['label'].std():.3f}")
    
    print(f"\n Label Bins:")
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']
    df_output['label_bin'] = pd.cut(df_output['label'], bins=bins, labels=labels, include_lowest=True)
    print(df_output['label_bin'].value_counts().sort_index())
    
    print(f"\n Hard Negatives: {df_output['is_hard_negative'].sum():,} ({100*df_output['is_hard_negative'].mean():.1f}%)")
    
    print(f"\nFold Distribution:")
    for fold in range(5):
        fold_size = (df_output['fold'] == fold).sum()
        print(f"   Fold {fold}: {fold_size:,} samples")
    
    print(f"\n Saved to: {OUTPUT_PATH}")
    print("="*60)


if __name__ == "__main__":
    main()
