import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_probs: np.ndarray = None) -> dict:
    """Compute classification metrics.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_probs: Predicted probabilities (optional).
    
    Returns:
        Dictionary containing metrics.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.ndim != 1 or y_pred.shape != y_true.shape or len(y_true) == 0:
        raise ValueError("Expected nonempty, aligned one-dimensional labels")
    if not np.isin(y_true, [0, 1]).all() or not np.isin(y_pred, [0, 1]).all():
        raise ValueError("Binary labels must be 0 or 1")
    if y_probs is not None:
        y_probs = np.asarray(y_probs)
        if y_probs.shape != y_true.shape or not np.isfinite(y_probs).all() or ((y_probs < 0) | (y_probs > 1)).any():
            raise ValueError("Expected aligned finite positive-class probabilities in [0, 1]")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel().tolist()
        specificity = (tn / (tn + fp)) if (tn + fp) > 0 else None
    else:
        specificity = None

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="binary", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="binary", zero_division=0),
        "specificity": specificity,
        "f1": f1_score(y_true, y_pred, average="binary", zero_division=0),
        "confusion_matrix": cm.tolist(),
    }

    if y_probs is not None:
        if len(np.unique(y_true)) < 2:
            metrics.update(roc_auc=None, auprc=None)
            return metrics
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_probs)
        except ValueError:
            metrics["roc_auc"] = None
        try:
            metrics["auprc"] = average_precision_score(y_true, y_probs)
        except ValueError:
            metrics["auprc"] = None
    
    return metrics


def print_metrics(metrics: dict) -> None:
    """Pretty print metrics.
    
    Args:
        metrics: Dictionary of metrics.
    """
    print("=" * 50)
    print("Classification Metrics")
    print("=" * 50)
    for key, value in metrics.items():
        if key == "confusion_matrix":
            print(f"\n{key}:")
            cm = np.array(value)
            print(cm)
        elif isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    print("=" * 50)
