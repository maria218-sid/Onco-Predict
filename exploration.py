import pandas as pd
import numpy as np


# --- 1. The Robust Data Loading ---
# Logic: Load the data directly from the standardized Hugging Face dataset repository. 
# This bypasses local file errors (encoding, delimiter, file not found).
try:
    # Use the full link format that Pandas can interpret and load data
    DATA_URL = "hf://datasets/scikit-learn/breast-cancer-wisconsin/breast_cancer.csv"
    df = pd.read_csv(DATA_URL)
    
    print("-" * 50)
    print(f"SUCCESS: Data loaded directly from Hugging Face Hub.")
    print(f"DataFrame has {len(df)} rows and {len(df.columns)} columns.")

except Exception as e:
    print("-" * 50)
    print(f"FATAL ERROR: Could not load data from Hugging Face. Check your network or permissions: {e}")
    exit()
"""
# --- 2. INSPECTION ---
print("-" * 50)
print("1. DATA HEAD (First 5 Rows): Quick Content Preview")
print(df.head()) # To confirm the data loaded correctly and to check the column names.

print("-" * 50)
print("2. DATA INFO (Structure and Missing Values): Crucial for Cleaning")
df.info() # This is the most crucial step. It tells you exactly where your problems are (missing values) and what type of data Python thinks it's working with.

print("-" * 50)
print("3. STATISTICAL SUMMARY: Quick Range and Distribution Check")
print(df.describe()) # quickly understand the spread and scale of your numerical data, helping you spot impossible values or major outliers.
print("-" * 50)
"""
# --- DATA CLEANING START ---

# 1. Handle Missing Values / Drop Useless Columns

# AI Suggests: Drop 'Unnamed: 32' and 'id'.
# Manually add comments explaining WHY:
# ----------------------------------------------------------------------------------
# WHY: 'Unnamed: 32' was found to have 0 non-null entries (entirely empty). It provides
# zero information for the model and must be dropped (Listwise Deletion: Column).
# WHY: The 'id' column is a unique patient identifier. It has no predictive power
# and, if left in, could cause the model to overfit (memorize IDs instead of features).
# ----------------------------------------------------------------------------------
df_cleaned = df.drop(columns=['Unnamed: 32', 'id'], axis=1)

print(" Step 1: Dropped 'Unnamed: 32' and 'id' columns.")
print(f"New DataFrame shape: {df_cleaned.shape}")

# 2. Convert Target Variable Type (Necessary Cleaning)

# Logic: Machine Learning models only work with numbers. We must convert the target
# column 'diagnosis' from object ('M', 'B') to a binary integer (1, 0).
df_cleaned['diagnosis'] = df_cleaned['diagnosis'].map({'M': 1, 'B': 0})

print(" Step 2: Converted 'diagnosis' (M/B) to binary (1/0).")
print("-" * 50)

# 3. Final Inspection Check
print("Final Check:")
print(df_cleaned['diagnosis'].value_counts())
print("\nFinal Data Info:")
df_cleaned.info()

# --- DATA CLEANING END ---

# --- THURSDAY: FEATURE SELECTION & SCALING ---
from sklearn.preprocessing import StandardScaler

# 1. Selection: Picking our specific runners for the race
X = df_cleaned[['radius_mean', 'texture_mean', 'smoothness_mean']]
y = df_cleaned['diagnosis']

# WHY: (Your turn! Why do we pick just these 3 and keep 'diagnosis' separate?)
# -------------------------------------------------------------------------
# 
# -------------------------------------------------------------------------

# 2. Scaling: Putting everyone on the same playing field
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# WHY: (Your turn! Why do we scale numbers like 14.0 and 0.09 to be similar?)
# -------------------------------------------------------------------------
# 
# -------------------------------------------------------------------------

# 3. Final Output: Show the "Equalized" data
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
print("-" * 50)
print("THURSDAY TASK: Selection and Scaling Complete.")
print("New Scaled Data (First 5 rows):")
print(X_scaled_df.head())

# --- FRIDAY: SAVE DATA ---

# 1. Combine the scaled features (X) and the diagnosis (y) back into one table
# Logic: We use 'concat' to glue the columns together. 
# reset_index(drop=True) makes sure the rows line up perfectly.
df_final = pd.concat([X_scaled_df, y.reset_index(drop=True)], axis=1)

# 2. Save to CSV
# Instruction: Ensure index=False to avoid creating an extra "Unnamed" column.
df_final.to_csv("cleaned_cancer_data.csv", index=False)

print("-" * 50)
print("✅ FRIDAY TASK COMPLETE: Cleaned data saved as 'cleaned_cancer_data.csv'")
print(f"Final file shape: {df_final.shape}")