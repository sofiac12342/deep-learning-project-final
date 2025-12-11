"""
Validate Gemini Labels vs Human Labels
======================================
Calculates correlation between Gemini's automated labels and human labels
to measure how accurate the AI labeling was.

Usage:
    python validate_gemini_labels.py path/to/human_labels.csv
"""

import pandas as pd
from scipy.stats import pearsonr, spearmanr
import sys

def validate_labels(csv_path):
    # Load labeled data
    df = pd.read_csv(csv_path)
    
    # Keep only rows with actual labels
    df = df[df['chunk_id'].notna()].copy()
    print(f"Analyzing {len(df)} labeled chunks\n")
    
    # The 4 quality dimensions
    dimensions = ['relevance', 'depth', 'clarity', 'examples']
    
    print("=" * 50)
    print("CORRELATION: Human vs Gemini Labels")
    print("=" * 50)
    
    for dim in dimensions:
        gemini_col = f'gemini_{dim}'
        human_col = f'human_{dim}'
        
        # Get paired values
        mask = df[gemini_col].notna() & df[human_col].notna()
        gemini = df.loc[mask, gemini_col]
        human = df.loc[mask, human_col]
        
        # Calculate correlations
        pearson_r, p_value = pearsonr(gemini, human)
        spearman_r, _ = spearmanr(gemini, human)
        
        print(f"\n{dim.upper()}:")
        print(f"  Pearson r  = {pearson_r:.3f}")
        print(f"  Spearman ρ = {spearman_r:.3f}")
        print(f"  p-value    = {p_value:.4f} {'✓ significant' if p_value < 0.05 else ''}")
    
    # Overall correlation
    print("\n" + "=" * 50)
    all_gemini = df[['gemini_relevance', 'gemini_depth', 'gemini_clarity', 'gemini_examples']].values.flatten()
    all_human = df[['human_relevance', 'human_depth', 'human_clarity', 'human_examples']].values.flatten()
    
    # Remove NaN pairs
    mask = ~(pd.isna(all_gemini) | pd.isna(all_human))
    all_gemini = all_gemini[mask]
    all_human = all_human[mask]
    
    overall_r, _ = pearsonr(all_gemini, all_human)
    mae = abs(all_gemini - all_human).mean()
    
    print(f"\nOVERALL CORRELATION: r = {overall_r:.3f}")
    print(f"Mean Absolute Error: {mae:.2f} points (on 1-5 scale)")
    print(f"Based on {len(all_gemini)} total ratings")
    
    # Interpretation
    print("\n" + "=" * 50)
    if overall_r >= 0.7:
        print("STRONG agreement - Gemini labels are reliable!")
    elif overall_r >= 0.5:
        print("MODERATE agreement - Gemini labels are reasonably accurate")
    elif overall_r >= 0.3:
        print("WEAK agreement - some discrepancy")
    else:
        print("LOW agreement - significant discrepancy")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        # Default path
        csv_path = '../data/processed/human_labels.csv'
    
    validate_labels(csv_path)

