"""
YouTube Data API v3 Client
Provides wrapper functions for searching videos and fetching metadata.
"""

import os
from typing import List, Dict, Optional
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


class YouTubeClient:
  
    def __init__(self, api_key: Optional[str] = None):
    
        self.api_key = api_key or os.getenv('YOUTUBE_API_KEY')
        if not self.api_key:
            raise ValueError(
                "YouTube API key not found. set YOUTUBE_API_KEY environment variable "
                "or pass it to the constructor."
            )
        
        # Build YouTube service
        self.youtube = build('youtube', 'v3', developerKey=self.api_key)
    
    def search_videos(self, query: str, max_results: int = 20) -> List[str]:
    
        try:
            # Call the search.list method to retrieve results
            search_response = self.youtube.search().list(
                q=query,
                type='video',
                part='id',
                maxResults=min(max_results, 50),  # API max is 50
                relevanceLanguage='en',  # Prefer English content
                videoEmbeddable='true',  # Only embeddable videos
                videoSyndicated='true'   # only syndicated videos
            ).execute()
            
            # Extract video IDs from search results
            video_ids = [
                item['id']['videoId']
                for item in search_response.get('items', [])
                if item['id']['kind'] == 'youtube#video'
            ]
            
            print(f"  Found {len(video_ids)} videos for query: '{query}'")
            return video_ids
        
        except HttpError as e:
            print(f"  HTTP error occurred while searching for '{query}': {e}")
            return []
        except Exception as e:
            print(f"  Unexpected error during search for '{query}': {e}")
            return []
    
    def fetch_videos_metadata(self, video_ids: List[str]) -> List[Dict]:

        all_metadata = []
        
        # Process in batches of 50 (API limit)
        batch_size = 50
        for i in range(0, len(video_ids), batch_size):
            batch = video_ids[i:i + batch_size]
            
            try:
                # Call the videos.list method
                videos_response = self.youtube.videos().list(
                    part='snippet,contentDetails,statistics',
                    id=','.join(batch)
                ).execute()
                
                # Extract metadata for each video
                for item in videos_response.get('items', []):
                    snippet = item.get('snippet', {})
                    content_details = item.get('contentDetails', {})
                    statistics = item.get('statistics', {})
                    
                    metadata = {
                        'video_id': item['id'],
                        'title': snippet.get('title', ''),
                        'description': snippet.get('description', ''),
                        'channel_title': snippet.get('channelTitle', ''),
                        'published_at': snippet.get('publishedAt', ''),
                        'duration': content_details.get('duration', ''),
                        'view_count': int(statistics.get('viewCount', 0)),
                        'like_count': int(statistics.get('likeCount', 0)),
                        'comment_count': int(statistics.get('commentCount', 0)),
                        'default_language': snippet.get('defaultLanguage', ''),
                        'default_audio_language': snippet.get('defaultAudioLanguage', '')
                    }
                    all_metadata.append(metadata)
                
                print(f"  Fetched metadata for {len(batch)} videos (batch {i//batch_size + 1})")
            
            except HttpError as e:
                print(f"  HTTP error occurred while fetching metadata: {e}")
            except Exception as e:
                print(f"  Unexpected error during metadata fetch: {e}")
        
        print(f"Total metadata fetched: {len(all_metadata)} videos")
        return all_metadata

def get_youtube_client(api_key: Optional[str] = None) -> YouTubeClient:

    return YouTubeClient(api_key)
