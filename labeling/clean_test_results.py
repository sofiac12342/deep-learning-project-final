import pandas as pd
from pathlib import Path

# Path to your file
file_path = Path("data/test_dataset/labeling/chunk_scores_gemini.csv")

print(f"Reading {file_path}...")
df = pd.read_csv(file_path)
print(f"Original row count: {len(df)}")


df['traditional_relevance'] = pd.to_numeric(df['traditional_relevance'], errors='coerce')


df_clean = df.dropna(subset=['traditional_relevance'])
df_clean = df_clean[df_clean['traditional_relevance'] > 0]


df_clean = df_clean.sort_values(['video_id', 'chunk_id'], ascending=[True, True])


df_clean = df_clean.drop_duplicates(subset=['video_id', 'chunk_id'], keep='last')


df_clean.to_csv(file_path, index=False)


