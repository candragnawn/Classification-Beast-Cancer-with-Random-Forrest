import pandas as pd
import numpy as np

def clean_data(df: pd.DataFrame,
               drop_duplicates: bool = True,
               handle_missing: str = "mean",  # options: mean, median, mode, drop
               lower_case_columns: bool = True,
               strip_whitespace: bool = True,
               remove_outliers: bool = False,
               z_thresh: float = 3.0):
    
    df = df.copy()
    
    if lower_case_columns:
        df.columns = df.columns.str.lower().str.replace(" ", "_")
    
    if strip_whitespace:
        df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    
    if drop_duplicates:
        df = df.drop_duplicates()
    
    if handle_missing == "mean":
        df = df.fillna(df.mean(numeric_only=True))
    elif handle_missing == "median":
        df = df.fillna(df.median(numeric_only=True))
    elif handle_missing == "mode":
        df = df.fillna(df.mode().iloc[0])
    elif handle_missing == "drop":
        df = df.dropna()
    
    if remove_outliers:
        numeric_cols = df.select_dtypes(include=np.number).columns
        z_scores = np.abs((df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std())
        df = df[(z_scores < z_thresh).all(axis=1)]
    
    return df