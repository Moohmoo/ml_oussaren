import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(file_path: str):
    """Load the data from a CSV file.

    Parameters:
        - file_path: str, the path to the CSV file

    Returns:
        - data: DataFrame containing the data
    """
    data = pd.read_csv(file_path)
    return data

def handle_missing_values(data: pd.DataFrame, missing_values: list = ['?']):
    """Identify missing character, replace them with the median of the column.
    Then, create a new column to indicate if the value was missing or not.
    
    Parameters:
        - data: DataFrame
        - missing_values: list, default=['?']
        
    Returns:
        - data: DataFrame with missing character handled
    """
    data = data.replace(missing_values, np.nan)
    col_na = data.columns[data.isna().any()]
    for col in col_na:
        data[col] = data[col].astype(float)
        data[col] = data[col].fillna(data[col].median())
        data[f'has_{col}'] = data[col].notna().astype(int)
    return data

def get_columns(data: pd.DataFrame, type: str):
    """Get the list of columns in the DataFrame based on the type.

    Parameters:
        - data: DataFrame
        - type: str, 'categorial' or 'numeric'

    Returns:
        - columns: list containing the names of columns
    """
    target_column = 'target'
    categorical_col = []
    numeric_col = []
    for col in data.columns:
        if col != target_column:
            unique_values = data[col].nunique()
            if unique_values <= 10:
                categorical_col.append(col)
            else :
                numeric_col.append(col)
    return categorical_col if type == 'categorical' else numeric_col

def normalize_data(data: pd.DataFrame):
    """Normalize the data using the min-max normalization method.

    Parameters:
        - data: DataFrame

    Returns:
        - data: DataFrame with normalized values
    """
    scaler = StandardScaler()
    numeric_columns = get_columns(data, 'numeric')
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
    return data