# model_selection.py
# Defines the models and hyperparameter search spaces to be tried.

import config
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR

def get_models_and_parameters() -> list:
    """
    Defines the models and their corresponding hyperparameter grids
    to be searched during tuning.
    
    Returns:
        list: A list of tuples. Each tuple contains:
              (model_name, model_pipeline, parameter_grid)
    """
    models_list = []

    if config.PROBLEM_TYPE == "classification":
        # --- Classification Models ---

        # 1. Logistic Regression
        lr_pipe = Pipeline([('model', LogisticRegression(max_iter=1000, random_state=42))])
        lr_params = {
            'model__C': [0.1, 1.0, 10.0],
            'model__solver': ['liblinear', 'saga']
        }
        models_list.append(("LogisticRegression", lr_pipe, lr_params))

        # 2. Random Forest Classifier
        rf_pipe = Pipeline([('model', RandomForestClassifier(random_state=42))])
        rf_params = {
            'model__n_estimators': [50, 100],
            'model__max_depth': [None, 10, 20],
            'model__min_samples_leaf': [1, 2]
        }
        models_list.append(("RandomForestClassifier", rf_pipe, rf_params))

        # 3. Support Vector Machine (SVC) - can be slow on large data
        # svc_pipe = Pipeline([('model', SVC(probability=True, random_state=42))])
        # svc_params = {
        #     'model__C': [0.1, 1.0],
        #     'model__kernel': ['rbf', 'linear']
        # }
        # models_list.append(("SupportVectorMachine", svc_pipe, svc_params))

    elif config.PROBLEM_TYPE == "regression":
        # --- Regression Models ---

        # 1. Linear Regression
        lin_pipe = Pipeline([('model', LinearRegression())])
        lin_params = {
            # Linear Regression has few hyperparameters to tune
            'model__fit_intercept': [True, False]
        }
        models_list.append(("LinearRegression", lin_pipe, lin_params))
        
        # 2. Random Forest Regressor
        rf_pipe = Pipeline([('model', RandomForestRegressor(random_state=42))])
        rf_params = {
            'model__n_estimators': [50, 100],
            'model__max_depth': [None, 10, 20],
            'model__min_samples_leaf': [1, 2]
        }
        models_list.append(("RandomForestRegressor", rf_pipe, rf_params))

    else:
        raise ValueError(f"Unknown PROBLEM_TYPE in config: {config.PROBLEM_TYPE}")

    return models_list
