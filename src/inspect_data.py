import pandas as pd
import numpy as np

def inspect_data(filepath):
    print(f"Loading data from {filepath}...")
    df = pd.read_excel(filepath)
    
    print("\n--- Basic Information ---")
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")
    
    print("\n--- Column Names and Data Types ---")
    print(df.dtypes)
    
    print("\n--- First 3 Rows ---")
    print(df.head(3))
    
    print("\n--- Last 3 Rows ---")
    print(df.tail(3))
    
    print("\n--- Missing Values ---")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    
    print("\n--- Duplicate Rows ---")
    print(f"Number of duplicate rows: {df.duplicated().sum()}")
    
    print("\n--- Target Variable Distribution ('Churn Label') ---")
    if 'Churn Label' in df.columns:
        print(df['Churn Label'].value_counts(normalize=True))
        print(df['Churn Label'].value_counts())
    
    print("\n--- Investigating 'Total Charges' ---")
    if 'Total Charges' in df.columns:
        # Check if there are spaces or empty strings
        empty_charges = df[df['Total Charges'].apply(lambda x: isinstance(x, str) and x.strip() == '')]
        print(f"Number of rows with empty string in 'Total Charges': {len(empty_charges)}")
        try:
            pd.to_numeric(df['Total Charges'])
            print("'Total Charges' can be converted to numeric directly.")
        except Exception as e:
            print(f"Error converting 'Total Charges' to numeric: {e}")

    # Output unique values for categorical columns to spot inconsistencies
    print("\n--- Unique Values in Categorical Columns ---")
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if col not in ['CustomerID', 'Customer ID', 'Count']:  # Skip obvious ID columns
            unique_vals = df[col].unique()
            if len(unique_vals) < 20:
                print(f"{col}: {unique_vals}")
            else:
                print(f"{col}: {len(unique_vals)} unique values (too many to print)")

if __name__ == "__main__":
    filepath = "data/raw/Telco_customer_churn.xlsx"
    inspect_data(filepath)
