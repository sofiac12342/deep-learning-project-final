"""
Data Acquisition Pipeline 
Main entry point: collects YouTube videos, metadata, and transcripts.
"""
import os
import sys
import argparse
import random
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime
import time
import pandas as pd
import yaml
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable
)
import isodate
import requests

from src.acquisition.youtube_client import YouTubeClient


def load_skills_config(config_path: str = "config/skills.yml") -> List[Dict]:
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    skills = config.get('skills', [])
    print(f"Loaded {len(skills)} skills from {config_path}")
    return skills


def collect_candidate_videos(
    youtube_client: YouTubeClient,
    skills: List[Dict],
    batch_id: str,
    max_results_per_query: int = 20
) -> pd.DataFrame:
    print("\n=== Collecting Candidate Videos ===")
    
    candidates = []
    
    for skill in skills:
        skill_id = skill['skill_id']
        skill_name = skill['name']
        queries = skill['queries']
        
        print(f"\nSkill: {skill_name} ({skill_id})")
        print(f"  Running {len(queries)} queries...")
        
        for query in queries:
            video_ids = youtube_client.search_videos(query, max_results=max_results_per_query)
            
            for video_id in video_ids:
                candidates.append({
                    'video_id': video_id,
                    'skill_id': skill_id,
                    'batch_id': batch_id
                })
    
    # Create DataFrame and deduplicate
    df = pd.DataFrame(candidates)
    
    if df.empty:
        print("\nWARNING: No candidate videos collected!")
        return df
    
    print(f"\nTotal candidate pairs collected: {len(df)}")
    
    # Deduplicate by (video_id, skill_id)
    df_dedup = df.drop_duplicates(subset=['video_id', 'skill_id'], keep='first')
    print(f"After deduplication: {len(df_dedup)} unique (video_id, skill_id) pairs")
    
    return df_dedup


def fetch_videos_metadata_df(
    youtube_client: YouTubeClient,
    video_ids: List[str],
    batch_id: str
) -> pd.DataFrame:
    print("\n=== Fetching Video Metadata ===")
    print(f"Fetching metadata for {len(video_ids)} unique videos...")
    
    metadata_list = youtube_client.fetch_videos_metadata(video_ids)
    
    if not metadata_list:
        print("WARNING: No metadata retrieved!")
        return pd.DataFrame()
    
    # Convert to DataFrame and add batch_id
    df = pd.DataFrame(metadata_list)
    df['batch_id'] = batch_id
    
    # Reorder columns for clarity
    columns_order = [
        'video_id', 'title', 'description', 'channel_title',
        'published_at', 'duration', 'view_count', 'like_count',
        'comment_count', 'default_language', 'default_audio_language',
        'batch_id'
    ]
    df = df[[col for col in columns_order if col in df.columns]]
    
    return df


def fetch_transcripts(video_ids: List[str], batch_id: str) -> pd.DataFrame:
 
    print("\n=== Fetching Transcripts ===")
    print(f"Attempting to fetch transcripts for {len(video_ids)} videos...")
    
    # Calculate estimated completion time with randomized delays
    min_delay = 45  # Min 45 seconds
    max_delay = 65  # Maximum 65 seconds
    avg_delay = (min_delay + max_delay) / 2
    estimated_minutes = (len(video_ids) * avg_delay) / 60
    estimated_hours = estimated_minutes / 60
    print(f"Estimated runtime: {estimated_hours:.1f} hours ({estimated_minutes:.0f} minutes)")
    print(f"Using randomized delays between {min_delay}-{max_delay} seconds (very conservative)")
 
    proxy_host = os.getenv('PROXY_HOST')
    proxy_port = os.getenv('PROXY_PORT')
    proxy_username = os.getenv('PROXY_USERNAME')
    proxy_password = os.getenv('PROXY_PASSWORD')
    
    if all([proxy_host, proxy_port, proxy_username, proxy_password]):
        proxy_url = f"http://{proxy_username}:{proxy_password}@{proxy_host}:{proxy_port}"
        os.environ['HTTP_PROXY'] = proxy_url
        os.environ['HTTPS_PROXY'] = proxy_url
        print(f"Using Webshare rotating proxy: {proxy_host}:{proxy_port}")
        print("Combining proxy rotation with long delays for maximum protection")
    else:
        # Clear any proxy environment variables 
        os.environ.pop('HTTP_PROXY', None)
        os.environ.pop('HTTPS_PROXY', None)
        print("Proxy credentials not found - using direct connection")
    
    transcripts_data = []
    success_count = 0
    
    api = YouTubeTranscriptApi()
    
    start_time = time.time()
    
    for i, video_id in enumerate(video_ids):
        if (i + 1) % 10 == 0:  # Update every 10 videos
            elapsed_minutes = (time.time() - start_time) / 60
            remaining_videos = len(video_ids) - (i + 1)
            estimated_remaining_minutes = (remaining_videos * avg_delay) / 60
            eta_timestamp = datetime.now().timestamp() + (estimated_remaining_minutes * 60)
            eta_str = datetime.fromtimestamp(eta_timestamp).strftime('%I:%M %p')
            print(f"  Progress: {i + 1}/{len(video_ids)} videos | Success: {success_count} | Elapsed: {elapsed_minutes:.1f} min | ETA: {eta_str}")
        
        transcript_text = ""
        has_transcript = False
        
        try:
            # Use the new fetch() method
            fetched = api.fetch(
                video_id,
                languages=['en', 'en-US', 'en-GB']
            )
            
            # Join all transcript snippets into one text
            transcript_text = " ".join(snippet.text for snippet in fetched)
            has_transcript = True
            success_count += 1
            print(f"  ✓ Fetched transcript for {video_id}")
        
        except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable) as e:
            # Transcript not available - this is expected for many videos
            print(f"  ✗ No transcript: {video_id} ({type(e).__name__})")
        except Exception as e:
            # Unexpected error - log but continue
            error_msg = str(e)[:100]
            print(f"  ✗ Unexpected error for video {video_id}: {error_msg}")
        
        transcripts_data.append({
            'video_id': video_id,
            'transcript_text': transcript_text,
            'has_transcript': has_transcript,
            'batch_id': batch_id
        })
        
        # Randomized delay 
        delay = min_delay + (max_delay - min_delay) * random.random()
        time.sleep(delay)
    
    total_time_minutes = (time.time() - start_time) / 60
    print(f"\nCompleted in {total_time_minutes:.1f} minutes ({total_time_minutes/60:.1f} hours)")
    print(f"Successfully fetched {success_count}/{len(video_ids)} transcripts ({100*success_count/len(video_ids):.1f}% success rate)")
    
    return pd.DataFrame(transcripts_data)


def parse_duration_to_seconds(duration_str: str) -> int:
    try:
        duration = isodate.parse_duration(duration_str)
        return int(duration.total_seconds())
    except Exception:
        return 0


def apply_quality_filters(
    videos_df: pd.DataFrame,
    transcripts_df: pd.DataFrame,
    min_duration_seconds: int = 90
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply quality filters to videos and transcripts.
    
    Filters applied:
    - Duration must be at least min_duration_seconds (default: 90 seconds)
    
    Args:
        videos_df: DataFrame with video metadata
        transcripts_df: DataFrame with transcripts
        min_duration_seconds: Minimum video duration in seconds
    
    Returns:
        Tuple of (filtered_videos_df, filtered_transcripts_df)
    """
    print("\n=== Applying Quality Filters ===")
    print(f"Videos before filtering: {len(videos_df)}")
    
    # Parse duration to seconds
    videos_df['duration_seconds'] = videos_df['duration'].apply(parse_duration_to_seconds)
    
    # Filter by duration
    videos_filtered = videos_df[videos_df['duration_seconds'] >= min_duration_seconds].copy()
    
    print(f"Videos after duration filter (>= {min_duration_seconds}s): {len(videos_filtered)}")
    

    filtered_video_ids = set(videos_filtered['video_id'])
    transcripts_filtered = transcripts_df[
        transcripts_df['video_id'].isin(filtered_video_ids)
    ].copy()
    
    print(f"Transcripts after filtering: {len(transcripts_filtered)}")
    
    return videos_filtered, transcripts_filtered


def save_data(
    candidates_df: pd.DataFrame,
    videos_raw_df: pd.DataFrame,
    transcripts_raw_df: pd.DataFrame,
    videos_filtered_df: pd.DataFrame,
    transcripts_filtered_df: pd.DataFrame,
    output_dir: str = "data/processed"
):

    print("\n=== Saving Data ===")
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save all datasets
    files = {
        'video_skill_candidates.csv': candidates_df,
        'videos_raw.csv': videos_raw_df,
        'transcripts_raw.csv': transcripts_raw_df,
        'videos_filtered.csv': videos_filtered_df,
        'transcripts_filtered.csv': transcripts_filtered_df
    }
    
    for filename, df in files.items():
        filepath = Path(output_dir) / filename
        df.to_csv(filepath, index=False, encoding='utf-8')
        print(f"  Saved: {filepath} ({len(df)} rows)")


def main():
    parser = argparse.ArgumentParser(description="ML Skill Video Ranking - Data Acquisition Pipeline")
    parser.add_argument("--config", default="config/skills.yml", help="Path to skills configuration file")
    parser.add_argument("--output-dir", default="data/processed", help="Directory to save processed data")
    args = parser.parse_args()

    print("=" * 60)
    print("ML Skill Video Ranking - Module 1: Data Acquisition")
    print("=" * 60)
    print(f"Configuration: {args.config}")
    print(f"Output Directory: {args.output_dir}")
    
    # Load environment variables
    load_dotenv()
    
    # Get configuration from environment
    api_key = os.getenv('YOUTUBE_API_KEY')
    if not api_key:
        print("\nERROR: YOUTUBE_API_KEY environment variable not set!")
        print("Please create a .env file based on .env.example and add your API key.")
        sys.exit(1)
    
    batch_id = os.getenv('BATCH_ID', f"{datetime.now().strftime('%Y-%m-%d')}-static-v1")
    print(f"\nBatch ID: {batch_id}")
    
    # Initialize YouTube client
    print("\nInitializing YouTube API client...")
    youtube_client = YouTubeClient(api_key)
    

    skills = load_skills_config(args.config)
    
    # Collect candidate videos
    candidates_df = collect_candidate_videos(youtube_client, skills, batch_id)
    
    if candidates_df.empty:
        print("\nERROR: No candidate videos collected. Exiting.")
        sys.exit(1)
    
    # Fetch metadata for unique videos
    unique_video_ids = candidates_df['video_id'].unique().tolist()
    videos_raw_df = fetch_videos_metadata_df(youtube_client, unique_video_ids, batch_id)
    
    if videos_raw_df.empty:
        print("\nERROR: No video metadata retrieved. Exiting.")
        sys.exit(1)
    
    #  Fetch transcripts
    transcripts_raw_df = fetch_transcripts(unique_video_ids, batch_id)
    
    #  Apply quality filters
    videos_filtered_df, transcripts_filtered_df = apply_quality_filters(
        videos_raw_df,
        transcripts_raw_df,
        min_duration_seconds=90
    )
    
    # Save
    save_data(
        candidates_df,
        videos_raw_df,
        transcripts_raw_df,
        videos_filtered_df,
        transcripts_filtered_df,
        output_dir=args.output_dir
    )
    
    print("\n" + "=" * 30)
    print("Pipeline completed successfully!")
    print("=" * 30)
    print("\nSummary:")
    print(f"  - Candidate pairs: {len(candidates_df)}")
    print(f"  - Unique videos (raw): {len(videos_raw_df)}")
    print(f"  - Videos with transcripts: {transcripts_raw_df['has_transcript'].sum()}")
    print(f"  - Videos after filtering: {len(videos_filtered_df)}")
    print(f"\nOutput files saved in: {args.output_dir}")


if __name__ == "__main__":
    main()
