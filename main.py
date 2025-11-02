# main.py
# This is the main orchestrator for the AutoML pipeline.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline # Import Pipeline

# Import our custom modules
import config
import data_loader
import preprocessing
import feature_engineering 
import model_selection
import tuning
import evaluation

def run_pipeline():
    """
    Executes the full AutoML pipeline from start to finish.
    """
    print("Starting AutoML pipeline...")

    # 1. Load Data
    print(f"Loading data from: {config.DATA_PATH}")
    try:
        df = data_loader.load_data(config.DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Data file not found at {config.DATA_PATH}")
        print("Please create a 'data' folder and add your dataset.")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. Define Features (X) and Target (y)
    if config.TARGET_COLUMN not in df.columns:
        print(f"Error: Target column '{config.TARGET_COLUMN}' not found in data.")
        print(f"Available columns: {df.columns.tolist()}")
        return
        
    y = df[config.TARGET_COLUMN]
    X = df.drop(columns=[config.TARGET_COLUMN])
    print(f"Target column set to: '{config.TARGET_COLUMN}'")

    # 3. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Data split: {len(X_train)} train samples, {len(X_test)} test samples.")

    # 4. Preprocessing --- NO LONGER FITTING ---
    print("Creating preprocessing pipeline...")
    
    # Get ID columns from config
    id_cols = getattr(config, 'ID_COLUMNS', [])
    
    # We *only* create the preprocessor. We DO NOT fit or transform it here.
    preprocessor = preprocessing.create_preprocessor(X_train, y_train, id_cols_to_drop=id_cols)
    

    # 5. Feature Engineering --- [THIS IS THE FIX] ---
    print("Checking for feature engineering step...")
    
    # Check if the preprocessor found any numeric features to process.
    # The FE pipeline is only for numeric data, so we check this first.
    has_numeric = False
    if hasattr(preprocessor, 'transformers'):
        # preprocessor.transformers is a list of (name, pipeline, columns)
        for name, _, cols in preprocessor.transformers:
            if name.startswith("num_") and cols: # e.g., "num_mean", "num_knn"
                has_numeric = True
                break

    if config.APPLY_FEATURE_ENGINEERING and has_numeric:
        print("Creating feature engineering pipeline (numeric features found).")
        fe_pipeline = feature_engineering.create_feature_pipeline()
    elif config.APPLY_FEATURE_ENGINEERING and not has_numeric:
        print("Skipping feature engineering: No numeric features were found by the preprocessor.")
        fe_pipeline = Pipeline(steps=[("passthrough", "passthrough")])
    else:
        if not config.APPLY_FEATURE_ENGINEERING:
            print("Skipping feature engineering: Disabled in config.py.")
        fe_pipeline = Pipeline(steps=[("passthrough", "passthrough")])
    # --- [END FIX] ---


    # 6. Model Selection
    print("Getting model search space...")
    # models_to_try is a list of (name, model_pipeline, param_grid)
    models_to_try = model_selection.get_models_and_parameters()

    best_model = None
    best_score = -1
    best_model_name = ""

    # 7. Model Tuning (Loop through all models)
    print("Starting model tuning...")
    for name, model_pipeline, param_grid in models_to_try:
        
        # We build a *new*, full pipeline for GridSearchCV that includes
        # all steps: preprocessing, feature engineering, and the model.
        
        model_steps = model_pipeline.steps
        
        # Create the final, full pipeline to be tuned
        full_pipeline_to_tune = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('feature_eng', fe_pipeline)
        ] + model_steps)
        # ------------------------

        print(f"--- Tuning: {name} ---")
        try:
            # We pass the FULL pipeline and the RAW X_train / y_train
            grid_search = tuning.tune_model(full_pipeline_to_tune, param_grid, X_train, y_train)
            
            print(f"Best params for {name}: {grid_search.best_params_}")
            print(f"Best score (CV) for {name}: {grid_search.best_score_:.4f}")

            if grid_search.best_score_ > best_score:
                best_score = grid_search.best_score_
                best_model = grid_search.best_estimator_ # This is the best *full_pipeline_to_tune*
                best_model_name = name

        except Exception as e:
            print(f"Failed to tune {name}. Error: {e}")

    if not best_model:
        print("Error: No models were successfully tuned.")
        return

    print(f"\n--- Best Model Found: {best_model_name} ---")
    print(f"Best Cross-Validation Score: {best_score:.4f}")

    # 8. Final Evaluation
    print("Evaluating best model on test set...")
    
    # `best_model` is now the complete, fit pipeline.
    # We pass the RAW X_test, and the pipeline will correctly
    # .transform() and .predict() all in one, leak-free step.
    metrics = evaluation.evaluate_model(best_model, X_test, y_test)
    
    print("\n--- Test Set Performance ---")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nAutoML pipeline finished.")

if __name__ == "__main__":
    run_pipeline()


# deploy
# Automated problem detection
# Feature engineering (use of LLM)
# Predict which model will be best (training a new model for this)
# 