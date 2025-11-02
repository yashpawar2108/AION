# tuning.py
# Functions for hyperparameter tuning.

import config
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

def tune_model(pipeline, param_grid, X_train, y_train):
    """
    Performs hyperparameter tuning on a given model pipeline.
    
    Selects tuning strategy based on 'config.py'.

    Args:
        pipeline (Pipeline): The model pipeline to tune (includes preprocessor + model).
        param_grid (dict): The dictionary of hyperparameters to search.
        X_train: Training features.
        y_train: Training target.

    Returns:
        GridSearchCV or RandomizedSearchCV object: The fitted tuner object.
    """
    
    # Determine the scoring metric based on problem type
    if config.PROBLEM_TYPE == "classification":
        scoring = "accuracy" # Could also be 'f1', 'roc_auc', etc.
    else:
        scoring = "r2" # Could also be 'neg_mean_squared_error'

    if config.TUNING_STRATEGY == "GridSearchCV":
        tuner = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=config.CV_FOLDS,
            scoring=scoring,
            n_jobs=-1, # Use all available cores
            verbose=1
        )
    elif config.TUNING_STRATEGY == "RandomizedSearchCV":
        tuner = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_grid,
            n_iter=10, # Number of random combinations to try
            cv=config.CV_FOLDS,
            scoring=scoring,
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
    else:
        raise ValueError(f"Unknown TUNING_STRATEGY in config: {config.TUNING_STRATEGY}")

    # Fit the tuner
    tuner.fit(X_train, y_train)
    
    return tuner
