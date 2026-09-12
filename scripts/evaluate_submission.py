"""Script to evaluate submission predictions against test ground truth (solution.csv).

This script is completely separate from the training pipeline to comply with
the Satria Data 2026 guidelines (test set labels must not be used in training,
validation, model selection, or tuning).
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import argparse
import sys
from pathlib import Path
import pandas as pd
from sklearn.metrics import classification_report, f1_score

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate macro-F1 score for test submission.")
    parser.add_argument(
        "--submission",
        type=str,
        default="outputs/submission.csv",
        help="Path to the generated submission file (default: outputs/submission.csv)"
    )
    parser.add_argument(
        "--solution",
        type=str,
        default="data/solution.csv",
        help="Path to the test solution ground truth file (default: data/solution.csv)"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    sub_path = Path(args.submission)
    sol_path = Path(args.solution)
    
    if not sub_path.exists():
        print(f"Error: Submission file '{sub_path}' not found.")
        sys.exit(1)
        
    if not sol_path.exists():
        print(f"Error: Solution file '{sol_path}' not found.")
        sys.exit(1)
        
    # Load files
    sub_df = pd.read_csv(sub_path)
    sol_df = pd.read_csv(sol_path)
    
    # Rename columns to avoid confusion
    sub_df = sub_df.rename(columns={"predicted": "predicted_sub"})
    sol_df = sol_df.rename(columns={"predicted": "predicted_sol"})
    
    # Merge on id to ensure alignment
    merged = pd.merge(sol_df, sub_df, on="id", how="inner")
    
    if len(merged) == 0:
        print("Error: No matching ids found between solution and submission.")
        sys.exit(1)
        
    if len(merged) < len(sol_df):
        print(f"Warning: Only {len(merged)} out of {len(sol_df)} rows matched.")
        
    # Calculate metrics
    y_true = merged["predicted_sol"]
    y_pred = merged["predicted_sub"]
    
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    
    print("=" * 60)
    print(" EVALUATION RESULTS ")
    print("=" * 60)
    print(f"Submission: {sub_path}")
    print(f"Solution:   {sol_path}")
    print(f"Matched samples: {len(merged)}")
    print("-" * 60)
    print(f"Macro-Averaged F1-Score: {macro_f1:.6f}")
    print("-" * 60)
    
    # Print class-wise report
    target_names = ["0_Recyclable", "1_Electronic", "2_Organic"]
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=target_names, digits=6))
    print("=" * 60)

if __name__ == "__main__":
    main()
