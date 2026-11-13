"""Evaluation metrics for regression accuracy, feature selection precision/recall/F1,
and ANOVA variance attribution.
"""

import numpy as np
import torch


def compute_regression_metrics(y_true, y_pred):
    """Computes MSE, RMSE, MAE, and R^2 score."""
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy().flatten()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy().flatten()
        
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    
    total_var = np.var(y_true)
    r2 = 1.0 - (mse / (total_var + 1e-8))
    
    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2)
    }


def compute_feature_selection_metrics(selected_indices, true_indices, total_features):
    """Computes Precision, Recall, F1-Score, and False Discovery Proportion (FDP)."""
    selected_set = set(selected_indices)
    true_set = set(true_indices)
    
    tp = len(selected_set.intersection(true_set))
    fp = len(selected_set - true_set)
    fn = len(true_set - selected_set)
    
    precision = tp / len(selected_set) if len(selected_set) > 0 else 0.0
    recall = tp / len(true_set) if len(true_set) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fdp = fp / len(selected_set) if len(selected_set) > 0 else 0.0
    
    return {
        'num_selected': len(selected_set),
        'true_positives': tp,
        'false_positives': fp,
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'fdp': float(fdp)
    }
