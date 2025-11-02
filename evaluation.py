# evaluation.py
# Functions for evaluating the final model.

import config
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

def evaluate_model(model, X_test, y_test) -> dict:
    """
    Evaluates the performance of the final model on the test set.

    Args:
        model (Pipeline): The trained final model pipeline.
        X_test: Test features.
        y_test: Test target.

    Returns:
        dict: A dictionary of performance metrics.
    """
    
    y_pred = model.predict(X_test)
    metrics = {}

    if config.PROBLEM_TYPE == "classification":
        metrics["accuracy"] = accuracy_score(y_test, y_pred)
        
        # F1 score (good for imbalanced classes)
        # Use 'macro' average to treat all classes equally
        metrics["f1_score_macro"] = f1_score(y_test, y_pred, average="macro")
        
        # ROC-AUC (if possible)
        try:
            # Requires probability predictions
            y_pred_proba = model.predict_proba(X_test)
            # Handle binary vs. multiclass
            if len(np.unique(y_test)) == 2:
                metrics["roc_auc"] = roc_auc_score(y_test, y_pred_proba[:, 1])
            else:
                metrics["roc_auc_ovr"] = roc_auc_score(y_test, y_pred_proba, multi_class="ovr")
        except Exception as e:
            print(f"Could not calculate ROC-AUC: {e}")

    elif config.PROBLEM_TYPE == "regression":
        metrics["r2_score"] = r2_score(y_test, y_pred)
        metrics["mean_squared_error"] = mean_squared_error(y_test, y_pred)
        metrics["mean_absolute_error"] = mean_absolute_error(y_test, y_pred)
        metrics["root_mean_squared_error"] = np.sqrt(metrics["mean_squared_error"])

    return metrics
