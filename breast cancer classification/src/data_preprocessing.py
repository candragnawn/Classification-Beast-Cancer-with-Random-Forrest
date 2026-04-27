import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(df: pd.DataFrame, 
                    target_column: str = None,
                    encode_categorical: bool = True,
                    scale_numeric: bool = False):
   
    df = df.copy()
    
    if encode_categorical:
        le = LabelEncoder()
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = le.fit_transform(df[col].astype(str))
            
    if scale_numeric:
        scaler = StandardScaler()
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 0:
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            
    return df