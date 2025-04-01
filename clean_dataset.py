import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from fuzzywuzzy import process, fuzz

# Constants
INPUT_FILE = "dataset.xlsx"
OUTPUT_FILE = "cleaned_dataset.csv"
SIMILARITY_THRESHOLD = 80
MULTI_VALUE_SEPARATORS = ['/', '&', ' and ', '-', ' x ']
STANDARD_SEPARATOR = ","

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(file_path):
    """
    Load data from an Excel file.
    
    Args:
        file_path (str): Path to the Excel file
        
    Returns:
        pandas.DataFrame: Loaded dataframe or None if loading fails
    """
    try:
        df = pd.read_excel(file_path, sheet_name=0)
        logger.info(f"Loaded {len(df)} records from Excel")
        return df
    except FileNotFoundError:
        logger.error(f"Error: File '{file_path}' not found. Please check the file path")
        return None
    except Exception as e:
        logger.error(f"Error loading file: {str(e)}")
        return None

def remove_duplicates(df):
    """Remove duplicate rows from the dataframe."""
    initial_rows = len(df)
    df.drop_duplicates(inplace=True)
    logger.info(f"Dropped {initial_rows - len(df)} duplicate rows. Now {len(df)} rows remain.")
    return df

def drop_unnecessary_columns(df, columns_to_drop):
    """Drop specified columns from the dataframe."""
    df.drop(columns=columns_to_drop, inplace=True)
    logger.info(f"Dropped columns: {', '.join(columns_to_drop)}")
    return df

def standardize_text(df):
    """Convert all string columns to lowercase."""
    df = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)
    logger.info("Standardized text to lowercase")
    return df

def standardize_dates(df, date_column, date_format="%d/%m/%Y"):
    """Standardize date columns to a consistent format."""
    df[date_column] = pd.to_datetime(df[date_column], format=date_format, errors="coerce").dt.strftime(date_format)
    logger.info(f"Standardized '{date_column}' column")
    return df

def normalize_multi_values(column, separator=STANDARD_SEPARATOR):
    """
    Normalize multiple values in a column.
    
    Args:
        column: String column value
        separator: Separator to use for normalized values
        
    Returns:
        Normalized string
    """
    if pd.isna(column):
        return np.nan
        
    # Replace various separators with the standard one
    for sep in MULTI_VALUE_SEPARATORS:
        column = column.replace(sep, separator)

    # Split, clean, and sort values
    values = [value.strip() for value in column.split(separator) if value.strip()]
    values.sort()

    return separator.join(values)

def normalize_columns_with_multiple_values(df, columns):
    """Apply multi-value normalization to specified columns."""
    for column in columns:
        df[column] = df[column].apply(normalize_multi_values)
    
    logger.info(f"Normalized columns to handle multiple values: {', '.join(columns)}")
    return df

def convert_numeric_columns(df, columns_dict):
    """
    Convert columns to appropriate numeric types.
    
    Args:
        df: DataFrame
        columns_dict: Dictionary mapping column names to conversion specs
                     (e.g., {'column': ('type', allow_na)})
    """
    for column, (dtype, nullable) in columns_dict.items():
        df[column] = pd.to_numeric(df[column], errors="coerce")
        if nullable:
            df[column] = df[column].astype(dtype)
        
    logger.info(f"Converted numeric columns: {', '.join(columns_dict.keys())}")
    return df

def set_data_types(df, type_dict):
    """Set appropriate data types for columns."""
    df = df.astype(type_dict)
    logger.info("Assigned appropriate data types")
    return df

def filter_rows(df, conditions):
    """
    Filter rows based on multiple conditions.
    
    Args:
        df: DataFrame
        conditions: List of dictionaries with keys:
                    - 'columns': list of columns to check
                    - 'condition': function to apply (optional)
                    - 'message': log message
    """
    for condition in conditions:
        initial_count = len(df)
        
        # Filter for NA values first
        if 'columns' in condition:
            df.dropna(subset=condition['columns'], inplace=True)
        
        # Apply custom condition if provided
        if 'condition' in condition:
            df = df[condition['condition'](df)]
            
        logger.info(f"{condition['message']} {initial_count - len(df)} rows removed. Now {len(df)} rows remain.")
    
    return df

def clean_text_column(df, column):
    """Clean and standardize a text column."""
    initial_unique = len(df[column].unique())
    
    def clean_value(value):
        if pd.isna(value):
            return value
        value = value.strip().lower()
        return value
    
    df[column] = df[column].apply(clean_value)
    logger.info(f"Cleaned '{column}' column. Unique values: {initial_unique} → {len(df[column].unique())}")
    return df

def find_similar_values(df, column, threshold=SIMILARITY_THRESHOLD):
    """
    Find and report similar values in a column using fuzzy matching.
    
    Args:
        df: DataFrame
        column: Column to check for similar values
        threshold: Similarity threshold (0-100)
    """
    unique_values = df[column].unique()
    logger.info(f"Checking similarities in {column} ({len(unique_values)} unique values)")
    
    # Create a dict to track which items have been compared
    compared = set()
    similar_items = {}
    
    for value in unique_values:
        if pd.isna(value) or value in compared:
            continue
            
        # Find matches with a similarity score above the threshold
        matches = process.extractBests(
            value, 
            [v for v in unique_values if v != value and v not in compared], 
            scorer=fuzz.token_sort_ratio, 
            score_cutoff=threshold
        )
        
        if matches:
            similar_items[value] = matches
            # Add matched items to compared set
            for match, _ in matches:
                compared.add(match)
            
    # Log similar items
    if similar_items:
        logger.info(f"Found similar values in '{column}':")
        for value, matches in similar_items.items():
            matches_str = ", ".join([f"{match} ({score}%)" for match, score in matches])
            logger.info(f"  - {value}: similar to {matches_str}")
    else:
        logger.info(f"No similar values found in '{column}' above threshold {threshold}%")
    
    return similar_items

def main():
    """Main function to orchestrate the data cleaning process."""
    # Load data
    df = load_data(INPUT_FILE)
    if df is None:
        return
    
    # Initial cleaning
    df = remove_duplicates(df)
    df = drop_unnecessary_columns(df, ["gps"])
    df = standardize_text(df)
    
    # Date standardization
    df = standardize_dates(df, "date")
    
    # Normalize columns with multiple values
    df = normalize_columns_with_multiple_values(df, ["colour", "shape", "organisation"])
    
    # Convert numeric columns
    numeric_columns = {
        "latitude": ("float", True),
        "longitude": ("float", True),
        "amount": ("Int64", True),
        "group": ("Int64", True)
    }
    df = convert_numeric_columns(df, numeric_columns)
    
    # Set data types
    type_dict = {
        "plasticType": "category",
        "colour": str,
        "size": "category",
        "shape": str,
        "comments": str,
        "organisation": str,
    }
    df = set_data_types(df, type_dict)
    
    # Filter out invalid rows
    filter_conditions = [
        {
            'columns': ["amount"],
            'condition': lambda df: df["amount"] != 0,
            'message': "Filtered rows with NA or zero 'amount'."
        },
        {
            'columns': ["location", "latitude", "longitude"],
            'message': "Filtered rows with NA location data."
        }
    ]
    df = filter_rows(df, filter_conditions)
    
    # Clean text columns
    for column in ["location", "organisation"]:
        df = clean_text_column(df, column)
    
    # Find similar values
    find_similar_values(df, "organisation")
    
    # Save cleaned dataset
    df.to_csv(OUTPUT_FILE, index=False)
    logger.info(f"Saved cleaned dataset to '{OUTPUT_FILE}'")
    
    # Display final info
    logger.info(f"Final dataset: {len(df)} rows")
    return df

if __name__ == "__main__":
    main()