#!/usr/bin/env python3
"""
Extract time series features from multivariate CSV files using tsfresh.

This script reads CSV files containing multivariate time series data,
extracts features for specified columns using tsfresh, and saves the
extracted features to CSV format.
"""

import argparse
import logging
from pathlib import Path
import pandas as pd
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute


# Default subset of essential columns for feature extraction
DEFAULT_SUBSET_ESSENTIAL = [
    'x-pos_center_shoulder',
    'x-pos_center_hip',
    'x-pos_left_ankle',
    'x-pos_right_ankle',
    'x-pos_right_wrist',
    'x-pos_left_wrist',
    'dist_left_elbow-right_elbow',
    'dist_left_ankle-right_ankle',
    'dist_left_wrist-left_hip',
    'dist_right_wrist-right_hip',
    'dist_left_wrist-right_wrist',
    'dist_left_knee-right_knee',
    'angle_right_shoulder-(right_wrist-right_hip)',
    'angle_left_shoulder-(left_hip-left_wrist)',
    'angle_left_hip-(left_ankle-right_ankle)',
    'angle_right_hip-(right_ankle-left_ankle)',
    'area_shoulders-hips',
    'area_hips-ankles',
    'diff_shoulders',
    'diff_hips'
]


def extract_ts_features(csv_path, output_path, subset_columns=None):
    """
    Extract time series features from a CSV file using tsfresh.
    
    Parameters
    ----------
    csv_path : str or Path
        Path to input CSV file containing time series data
    output_path : str or Path
        Path where extracted features CSV will be saved
    subset_columns : list of str, optional
        List of column names to extract features from.
        If None, uses DEFAULT_SUBSET_ESSENTIAL.
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing extracted features
    """
    if subset_columns is None:
        subset_columns = DEFAULT_SUBSET_ESSENTIAL
    
    logging.info(f"Reading time series data from {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Filter to only the columns that exist in the dataframe
    available_columns = [col for col in subset_columns if col in df.columns]
    missing_columns = [col for col in subset_columns if col not in df.columns]
    
    if missing_columns:
        logging.warning(f"The following columns were not found in {csv_path}: {missing_columns}")
    
    if not available_columns:
        logging.error(f"None of the specified columns were found in {csv_path}")
        raise ValueError("No valid columns found for feature extraction")
    
    logging.info(f"Extracting features from {len(available_columns)} columns")
    
    # Prepare data for tsfresh: add an id column and reshape
    # tsfresh expects data in long format with 'id', 'time', and 'value' columns
    # But for multiple columns, we can pass the wide format directly
    
    # Add an ID column (single time series in this case)
    df_with_id = df[available_columns].copy()
    df_with_id.insert(0, 'id', 0)
    
    # Extract features using tsfresh
    logging.info("Extracting features with tsfresh...")
    try:
        extracted_features = extract_features(
            df_with_id,
            column_id='id',
            disable_progressbar=False,
            n_jobs=1  # Use single job to avoid multiprocessing issues
        )
        
        # Impute NaN/Inf values that tsfresh might produce
        logging.info("Imputing missing values in extracted features...")
        impute(extracted_features)
        
        # Save to CSV
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        extracted_features.to_csv(output_path)
        logging.info(f"Saved {extracted_features.shape[1]} extracted features to {output_path}")
        
        return extracted_features
        
    except Exception as e:
        logging.error(f"Feature extraction failed: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Extract time series features from CSV using tsfresh"
    )
    parser.add_argument(
        "input_csv",
        help="Path to input CSV file containing time series data"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to output CSV file for extracted features (default: <input>_features.csv)"
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        default=None,
        help="Space-separated list of column names to extract features from. "
             "If not provided, uses default essential subset."
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=getattr(logging, args.log_level)
    )
    
    # Determine output path
    input_path = Path(args.input_csv)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_features.csv"
    
    # Determine columns to use
    columns = args.columns if args.columns else None
    
    # Extract features
    try:
        features = extract_ts_features(input_path, output_path, columns)
        logging.info(f"Feature extraction complete. Shape: {features.shape}")
    except Exception as e:
        logging.error(f"Failed to extract features: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
