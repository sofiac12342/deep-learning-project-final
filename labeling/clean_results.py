import pandas as pd
from pathlib import Path
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clean_results():
    # Paths
    BASE_DIR = Path(__file__).parent.parent.parent
    RAW_FILE = BASE_DIR / 'results' / 'labeling' / 'chunk_scores_gemini.csv'
    OUTPUT_FILE = BASE_DIR / 'data' / 'processed' / 'labeled_chunks.csv'
    
    if not RAW_FILE.exists():
        logger.error(f"File not found: {RAW_FILE}")
        return

    logger.info(f"Loading raw results from {RAW_FILE}...")
    df = pd.read_csv(RAW_FILE)
    
    initial_count = len(df)
    logger.info(f"Initial row count: {initial_count}")
    

    df_clean = df.dropna(subset=['traditional_relevance'])
    

    df_clean = df_clean[df_clean['traditional_relevance'] > 0]
    
    cleaned_count_1 = len(df_clean)
    removed_failed = initial_count - cleaned_count_1
    logger.info(f"Removed {removed_failed} failed/empty rows.")
    
    #  Remove duplicates

    df_clean = df_clean.drop_duplicates(subset=['video_id', 'chunk_id'], keep='last')
    
    final_count = len(df_clean)
    removed_duplicates = cleaned_count_1 - final_count
    logger.info(f"Removed {removed_duplicates} duplicate rows.")
    
    logger.info(f"Final row count: {final_count}")
    
    # Save 
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(OUTPUT_FILE, index=False)
    logger.info(f"Cleaned file saved to {OUTPUT_FILE}")
    
    # Verify stats
    print("\nFinal Statistics:")
    print(df_clean[['traditional_relevance', 'depth', 'clarity']].describe())

if __name__ == "__main__":
    clean_results()
