import pandas as pd
from pathlib import Path
import logging
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def aggregate_scores():
    parser = argparse.ArgumentParser(description='Aggregate chunk scores to video level')
    parser.add_argument('--input-file', type=str, 
                        default=str(Path(__file__).parent.parent.parent / 'data' / 'processed' / 'labeled_chunks.csv'),
                        help='Path to input labeled chunks CSV')
    parser.add_argument('--output-file', type=str,
                        default=str(Path(__file__).parent.parent.parent / 'data' / 'processed' / 'labeled_videos.csv'),
                        help='Path to output labeled videos CSV')
    
    args = parser.parse_args()
    
    INPUT_FILE = Path(args.input_file)
    OUTPUT_FILE = Path(args.output_file)
    
    if not INPUT_FILE.exists():
        logger.error(f"Input file not found: {INPUT_FILE}")
        logger.error("Please run clean_results.py first to generate the clean chunk data.")
        return

    logger.info(f"Loading chunk scores from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    
    # Filter out rows with missing scores (just in case)
    # Ensure we have the necessary columns
    required_cols = ['traditional_relevance', 'video_id', 'skill_id']
    if not all(col in df.columns for col in required_cols):
        logger.error(f"Missing required columns in input file. Found: {df.columns}")
        return

    df = df[df['traditional_relevance'] > 0].copy()
    
    logger.info(f"Processing {len(df)} labeled chunks...")

    # Define the columns we want to aggregate
    score_cols = [
        'traditional_relevance', 
        'depth', 
        'clarity', 
        'practical_examples', 
        'instructional_language'
    ]
    
    # Ensure all score columns exist
    score_cols = [col for col in score_cols if col in df.columns]

    # Function to calculate Top-K Mean for a group
    def get_top_k_mean(group, k=5):
        # Sort by traditional_relevance descending
        top_k = group.sort_values('traditional_relevance', ascending=False).head(k)
        

        return top_k[score_cols].mean()

    
    logger.info("Aggregating Top-5 chunks per video...")
    
    # We group by video_id and skill_id to ensure unique video-skill pairs
    # If skill_id is not in the chunk file, we might need to merge it back, 
    # but based on previous context, it should be there or we group by video_id only if 1 skill per video.
    # Assuming video_id is unique enough or we want per-skill scores.
    
    if 'skill_id' in df.columns:
        group_cols = ['video_id', 'skill_id']
    else:
        group_cols = ['video_id']
        
    video_scores = df.groupby(group_cols).apply(get_top_k_mean).reset_index()
    
    # Add a count column to see how many chunks contributed (useful for debugging)
    chunk_counts = df.groupby(group_cols).size().reset_index(name='chunk_count')
    video_scores = video_scores.merge(chunk_counts, on=group_cols)

    # Save results
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    video_scores.to_csv(OUTPUT_FILE, index=False)
    
    logger.info(f" Aggregation complete.")
    logger.info(f"   Saved video-level scores to: {OUTPUT_FILE}")
    logger.info(f"   Total videos scored: {len(video_scores)}")
    
    # Show sample
    print("\nSample Results (Top 5 rows):")
    print(video_scores.head().to_string())

if __name__ == "__main__":
    aggregate_scores()
