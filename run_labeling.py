

import argparse
from pathlib import Path
import pandas as pd
from gemini_labeler import GeminiLabeler
import logging
from dotenv import load_dotenv
import os

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Gemini chunk labeling')
    parser.add_argument('--limit', type=int, help='Limit number of videos to process (for testing)')
    parser.add_argument('--data-dir', type=str, default=str(Path(__file__).parent.parent.parent / 'data' / 'processed'), help='Directory containing input data')
    parser.add_argument('--config', type=str, default=str(Path(__file__).parent.parent.parent / 'config' / 'skills.yml'), help='Path to skills configuration file')
    parser.add_argument('--output-dir', type=str, default=str(Path(__file__).parent.parent.parent / 'results' / 'labeling'), help='Directory to save results')
    
    args = parser.parse_args()
    
    # paths
    DATA_DIR = Path(args.data_dir)
    CONFIG_PATH = Path(args.config)
    RESULTS_DIR = Path(args.output_dir)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # check for API key
    if not os.getenv('GEMINI_API_KEY') and not os.getenv('GOOGLE_API_KEY'):
        logger.error("GEMINI_API_KEY or GOOGLE_API_KEY environment variable not set!")
        logger.error("Set it with: $env:GEMINI_API_KEY='your-key-here'")
        return

    # Initialize labeler
    labeler = GeminiLabeler(model="gemini-3-pro-preview")

    # Load data
    logger.info(f"Loading transcript chunks and video-skill mappings from {DATA_DIR}...")
    chunks_df = pd.read_csv(DATA_DIR / 'transcript_chunks.csv')
    video_skills_df = pd.read_csv(DATA_DIR / 'video_skill_candidates.csv')
    
    # Load skills 
    logger.info(f"Loading skills configuration from {CONFIG_PATH}...")
    skills = labeler.load_skills(CONFIG_PATH)
    
    # Add skill info to video_skills_df
    video_skills_df['skill_name'] = video_skills_df['skill_id'].apply(lambda x: skills[x]['name'] if x in skills else None)
    video_skills_df['skill_description'] = video_skills_df['skill_id'].apply(lambda x: skills[x]['description'] if x in skills else None)
    
    # Filter out videos with unknown skills
    video_skills_df = video_skills_df.dropna(subset=['skill_name'])

    if args.limit:
        logger.info(f"Limiting to first {args.limit} videos for testing")
        limited_videos = video_skills_df['video_id'].unique()[:args.limit]
        video_skills_df = video_skills_df[video_skills_df['video_id'].isin(limited_videos)]
        chunks_df = chunks_df[chunks_df['video_id'].isin(limited_videos)]

    # Merge chunks with video skills
    logger.info("Merging chunks with skill info...")
    # Ensure we only keep chunks for videos we have skills for
    chunks_df = chunks_df.merge(
        video_skills_df[['video_id', 'skill_id', 'skill_name', 'skill_description']], 
        on='video_id', 
        how='inner'
    )
    
    # Sort by video_id and chunk_id
    chunks_df = chunks_df.sort_values(['video_id', 'chunk_id'])
    
    logger.info(f"Prepared {len(chunks_df)} chunks for labeling")

    # Run labeling
    print(f"\n Starting batch chunk labeling with Gemini...")
    
    # Score chunks
    chunk_scores_path = RESULTS_DIR / 'chunk_scores_gemini.csv'
    logger.info(f"Results will be saved to: {chunk_scores_path}\n")
    
    chunk_scores = labeler.score_chunks_batched(
        chunks_df,
        save_interval=500,
        output_path=chunk_scores_path,
        max_workers=15 
    )
    
    # Print summary
    print("\n" + "="*60)
    print("LABELING COMPLETE")
    print("="*60)
    print(f"\nTotal chunks processed: {len(chunk_scores):,}")
    
    failed_chunks = chunk_scores['traditional_relevance'].isnull().sum()
    print(f"Failed chunks: {failed_chunks:,}")
    
    print("\n Chunk-level score statistics:")
    print(chunk_scores[['traditional_relevance', 'depth', 'clarity', 
                       'practical_examples', 'instructional_language']].describe())
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
