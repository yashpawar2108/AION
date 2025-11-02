# All data preprocessing functions and pipeline creation.
# }
import config # Assuming config.py holds thresholds and settings
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import List, Tuple, Dict, Any

# --- [NEW] Imports needed for the custom flattener ---
from sklearn.base import BaseEstimator, TransformerMixin
# --- End new imports ---

# This is the new dependency you'll need for Target Encoding
# --- Make sure to install it: pip install category_encoders ---
try:
    from category_encoders import TargetEncoder
except ImportError:
    print("Warning: 'category_encoders' library not found. Please install with 'pip install category_encoders' for Target Encoding.")
    # Define a placeholder if not installed, though it will fail later
    TargetEncoder = None 

# --- [NEW] Custom Transformer to fix 2D array issue ---
class TextFlattener(BaseEstimator, TransformerMixin):
    """
    Converts a 2D array (n_samples, 1) from an imputer/ColumnTransformer
    into a 1D array (n_samples,) that TfidfVectorizer expects.
    """
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # X will be (n_samples, 1), .ravel() flattens it to (n_samples,)
        return X.ravel()
# --- End new class ---


# --- Feature Classification Helpers ---

def get_feature_metrics(s: pd.Series) -> Dict[str, Any]:
    """Calculates uniqueness ratio and average word count for a series."""
    if s.empty:
        return {'num_unique': 0, 'unique_ratio': 0.0, 'avg_words': 0.0}

    # Ensure series is treated as string for word count
    s_str = s.dropna().astype(str)
    
    num_rows = len(s)
    num_unique = s.nunique()
    unique_ratio = num_unique / num_rows if num_rows > 0 else 0.0
    
    # Calculate average words
    if s_str.empty:
        avg_words = 0.0
    else:
        avg_words = s_str.str.split().str.len().mean()
        avg_words = 0.0 if pd.isna(avg_words) else avg_words
        
    return {
        'num_unique': num_unique,
        'unique_ratio': unique_ratio,
        'avg_words': avg_words
    }


def classify_features(X: pd.DataFrame, id_cols: List[str] = None) -> Dict[str, List[str]]:
    """
    Automatically classifies columns into different feature types based on rules.
    
    Uses heuristics defined in config.py to sort columns into:
    - numeric
    - categorical
    - high_cardinality
    - text
    - id_to_drop
    """
    if id_cols is None:
        id_cols = []
        
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    object_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # Remove ID cols from processing lists
    numeric_features = [col for col in numeric_features if col not in id_cols]
    object_features = [col for col in object_features if col not in id_cols]

    # Initialize lists
    categorical_features = []
    high_card_features = []
    text_features = []

    # --- Feature Identification Logic ---
    for col in object_features:
        metrics = get_feature_metrics(X[col])
        
        # Rule 1: Textual Features
        if metrics['avg_words'] >= config.TEXT_AVG_WORD_THRESHOLD:
            text_features.append(col)
            
        # Rule 2: High-Cardinality Features
        elif metrics['num_unique'] > config.HIGH_CARD_THRESHOLD:
            high_card_features.append(col)
            
        # Rule 3: Standard Categorical Features
        else:
            categorical_features.append(col)

    return {
        "numeric": numeric_features,
        "categorical": categorical_features,
        "high_cardinality": high_card_features,
        "text": text_features,
        "id_to_drop": id_cols
    }


# --- Main Preprocessor Function ---

def create_preprocessor(X_train: pd.DataFrame, y_train: pd.Series, id_cols_to_drop: List[str] = None) -> ColumnTransformer:
    """
    Creates a full preprocessing pipeline using ColumnTransformer.
    """
    
    if TargetEncoder is None:
        raise ImportError("Please install 'category_encoders' with 'pip install category_encoders' to use this preprocessor.")

    # 1. Classify features
    feature_dict = classify_features(X_train, id_cols=id_cols_to_drop)
    
    numeric_features = feature_dict["numeric"]
    categorical_features = feature_dict["categorical"]
    high_card_features = feature_dict["high_cardinality"]
    text_features = feature_dict["text"]
    id_features = feature_dict["id_to_drop"]
    
    print("--- Initial Feature Classification ---")
    print(f"Numeric features: {numeric_features}")
    print(f"Categorical features: {categorical_features}")
    print(f"High-cardinality features: {high_card_features}")
    print(f"Text features: {text_features}")
    print(f"ID columns to drop: {id_features}")
    

    # --- [NEW] Step 1.5: Clean feature lists ---
    # Remove columns that are 100% missing *before* assigning to pipelines.
    # This fixes errors where imputers are fit on completely empty columns.
    
    print("--- Cleaning Feature Lists (Removing 100% NaN columns) ---")
    
    def clean_list(feature_list: List[str], list_name: str) -> List[str]:
        cols_to_drop = []
        for col in feature_list:
            if X_train[col].isna().all(): # Check if *all* values are NaN
                cols_to_drop.append(col)
        
        if cols_to_drop:
            print(f"Dropped from {list_name}: {cols_to_drop}")
            return [col for col in feature_list if col not in cols_to_drop]
        return feature_list

    numeric_features = clean_list(numeric_features, "Numeric")
    categorical_features = clean_list(categorical_features, "Categorical")
    high_card_features = clean_list(high_card_features, "High-Cardinality")
    text_features = clean_list(text_features, "Text")
    
    print("----------------------------------------------------------")

    
    # 2. --- Create pipelines for each feature type ---

    # --- Numeric Transformer Pipelines ---
    
    SKEW_THRESHOLD = getattr(config, 'SKEW_THRESHOLD', 1.0)
    HIGH_MISSING_THRESHOLD = getattr(config, 'HIGH_MISSING_THRESHOLD', 0.25)

    if config.NUMERIC_SCALER == 'standard':
        numeric_scaler = StandardScaler()
    elif config.NUMERIC_SCALER == 'minmax':
        numeric_scaler = MinMaxScaler()
    elif config.NUMERIC_SCALER == 'robust':
        numeric_scaler = RobustScaler()
    else:
        print(f"Warning: Unknown NUMERIC_SCALER '{config.NUMERIC_SCALER}'. Defaulting to StandardScaler.")
        numeric_scaler = StandardScaler()

    num_mean_impute = []
    num_median_impute = []
    num_knn_impute = []

    for col in numeric_features:
        missing_pct = X_train[col].isna().mean()
        skewness = X_train[col].skew()
        
        # Note: 100% NaN columns are already removed, so missing_pct < 1.0
        
        if missing_pct > HIGH_MISSING_THRESHOLD:
            num_knn_impute.append(col)
        elif abs(skewness) > SKEW_THRESHOLD:
            num_median_impute.append(col)
        else:
            num_mean_impute.append(col)

    print("--- Dynamic Numeric Strategy ---")
    print(f"Using KNNImputer for (high missing): {num_knn_impute}")
    print(f"Using MedianImputer for (high skew): {num_median_impute}")
    print(f"Using MeanImputer for (normal): {num_mean_impute}")
    print("--------------------------------")

    # Create the three potential numeric pipelines
    mean_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", numeric_scaler)
    ])
    
    median_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", numeric_scaler)
    ])
    
    knn_transformer = Pipeline(steps=[
        ("imputer", KNNImputer(n_neighbors=5)),
        ("scaler", numeric_scaler)
    ])


    # --- Categorical Transformer Pipeline ---
    categorical_steps = [
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ]
    categorical_transformer = Pipeline(steps=categorical_steps)

    
    # --- High-Cardinality Categorical Pipeline ---
    high_card_steps = [
        ("target_encoder", TargetEncoder(handle_missing='value', handle_unknown='value'))
    ]
    high_card_transformer = Pipeline(steps=high_card_steps)


    # --- Textual Feature Pipeline ---
    text_steps = [
        ("imputer", SimpleImputer(strategy="constant", fill_value="")),
        
        # --- [THIS IS THE FIX] ---
        # Add flattener to convert (n_samples, 1) to (n_samples,)
        ("flattener", TextFlattener()),
        # --- [END FIX] ---

        ("tfidf", TfidfVectorizer(
            max_features=config.TEXT_MAX_FEATURES, 
            stop_words="english",
            # This line was fixed in the previous step
            ngram_range=config.TEXT_NGRAM_RANGE
        ))
    ]
    text_transformer = Pipeline(steps=text_steps)


    # 3. --- Combine all transformers ---
    
    transformers_list = []
    
    # Add the three dynamic numeric pipelines
    # This logic is safe: if a list is empty, its 'if' block is skipped,
    # and no transformer is added. This prevents the 0-feature error.
    if num_mean_impute:
        transformers_list.append(("num_mean", mean_transformer, num_mean_impute))
    if num_median_impute:
        transformers_list.append(("num_median", median_transformer, num_median_impute))
    if num_knn_impute:
        transformers_list.append(("num_knn", knn_transformer, num_knn_impute))
        
    # Add the others
    if categorical_features:
        transformers_list.append(("cat", categorical_transformer, categorical_features))
    if high_card_features:
        transformers_list.append(("high_card", high_card_transformer, high_card_features))
    if text_features:
        transformers_list.append(("text", text_transformer, text_features))
    if id_features:
        transformers_list.append(("drop_ids", "drop", id_features))

    preprocessor = ColumnTransformer(
        transformers=transformers_list,
        remainder="passthrough" # Keep any columns not classified (e.g., dates)
    )
    
    return preprocessor

