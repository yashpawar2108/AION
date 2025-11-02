# feature_engineering.py
# Functions for creating and selecting features using a stateful pipeline.

import config
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.decomposition import PCA

# --- Custom Transformer: Drop High-Null Columns ---
# We can't use SimpleImputer(strategy='drop') as it's not a feature selector.

class HighNullDropper(BaseEstimator, TransformerMixin):
    """
    Drops columns with a missing value percentage higher than the threshold.
    """
    def __init__(self, threshold):
        self.threshold = threshold
        self.cols_to_drop_ = []

    def fit(self, X, y=None):
        # Ensure X is a DataFrame to use .isna()
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        missing_pct = X.isna().mean()
        self.cols_to_drop_ = missing_pct[missing_pct > self.threshold].index.tolist()
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return X.drop(columns=self.cols_to_drop_)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            raise ValueError("input_features must be provided to HighNullDropper.get_feature_names_out")
        return [col for col in input_features if col not in self.cols_to_drop_]


# --- Custom Transformer: Drop High-Correlation Columns ---

class CorrelationDropper(BaseEstimator, TransformerMixin):
    """
    Drops features that are highly correlated with another feature.
    It keeps the first feature in the pair.
    """
    def __init__(self, threshold):
        self.threshold = threshold
        self.cols_to_drop_ = []

    def fit(self, X, y=None):
        # Ensure X is a DataFrame to use .corr()
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        corr_matrix = X.corr(method='pearson').abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find features with correlation greater than threshold
        to_drop = [column for column in upper.columns if any(upper[column] > self.threshold)]
        self.cols_to_drop_ = to_drop
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return X.drop(columns=self.cols_to_drop_)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            raise ValueError("input_features must be provided to CorrelationDropper.get_feature_names_out")
        return [col for col in input_features if col not in self.cols_to_drop_]


# --- Main Pipeline Builder ---

def create_feature_pipeline() -> Pipeline:
    """
    Creates a full feature selection/engineering pipeline based on config.
    
    This pipeline assumes input is *after* preprocessing (i.e., numeric).
    """
    
    # Get settings from config, providing sensible defaults
    FE_HIGH_NULL_THRESHOLD = getattr(config, 'FE_HIGH_NULL_THRESHOLD', 0.95)
    FE_CORR_THRESHOLD = getattr(config, 'FE_CORR_THRESHOLD', 0.90)
    FE_USE_LASSO = getattr(config, 'FE_USE_LASSO', True)
    FE_USE_PCA = getattr(config, 'FE_USE_PCA', False)
    FE_PCA_EXPLAINED_VAR = getattr(config, 'FE_PCA_EXPLAINED_VAR', 0.95)
    
    # We can't use Lasso and PCA at the same time, as PCA needs all
    # features, while Lasso removes them.
    if FE_USE_LASSO and FE_USE_PCA:
        print("Warning: FE_USE_LASSO and FE_USE_PCA are both True. Disabling PCA.")
        FE_USE_PCA = False

    pipeline_steps = []
    
    # Step 1: Remove columns with > 95% missing values
    pipeline_steps.append(
        ("high_null_dropper", HighNullDropper(threshold=FE_HIGH_NULL_THRESHOLD))
    )
    
    # Step 2: Remove zero-variance columns (all the same value)
    pipeline_steps.append(
        ("variance_threshold", VarianceThreshold(threshold=0))
    )
    
    # Step 3: Remove highly correlated features
    pipeline_steps.append(
        ("correlation_dropper", CorrelationDropper(threshold=FE_CORR_THRESHOLD))
    )
    
    # Step 4 (Optional): Lasso Feature Selection
    if FE_USE_LASSO:
        print("Adding Lasso feature selection to pipeline.")
        # SelectFromModel will use LassoCV (Cross-Validation) to find
        # the best alpha (regularization) and drop features with
        # coefficients of 0.
        lasso_selector = SelectFromModel(
            LassoCV(cv=5, random_state=42), 
            threshold="median" # Keep features with importance above median
        )
        pipeline_steps.append(("lasso_selector", lasso_selector))

    # Step 5 (Optional): PCA Dimensionality Reduction
    if FE_USE_PCA:
        print(f"Adding PCA to pipeline, retaining {FE_PCA_EXPLAINED_VAR * 100}% variance.")
        # PCA will reduce dimensions to the number of components
        # that explain 95% of the variance.
        pca = PCA(n_components=FE_PCA_EXPLAINED_VAR, random_state=42)
        pipeline_steps.append(("pca", pca))

    if not pipeline_steps:
        print("No feature engineering steps enabled.")
        # Return a "passthrough" pipeline
        return Pipeline(steps=[("passthrough", "passthrough")])

    print("Feature engineering pipeline created.")
    return Pipeline(steps=pipeline_steps)
