import numpy as np


def compute_f1_score(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    p = compute_precision(y_pred, y_true)
    r = compute_recall(y_pred, y_true)
    try:
        f1 = 2 * (p * r) / (p + r)
    except ZeroDivisionError:
        f1 = 0
    return f1


def compute_true_positives(y_pred: np.ndarray, y_true: np.ndarray) -> int:
    return int(np.sum(np.logical_and(y_pred, y_true)))


def compute_false_positives(y_pred: np.ndarray, y_true: np.ndarray) -> int:
    return int(np.sum(np.logical_and(y_pred, np.logical_not(y_true))))


def compute_false_negatives(y_pred: np.ndarray, y_true: np.ndarray):
    return int(np.sum(np.logical_and(np.logical_not(y_pred), y_true)))


def compute_precision(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    tp = compute_true_positives(y_pred, y_true)
    fp = compute_false_positives(y_pred, y_true)
    return float(tp / (tp + fp))


def compute_recall(y_pred: np.ndarray, y_true: np.ndarray):
    tp = compute_true_positives(y_pred, y_true)
    fn = compute_false_negatives(y_pred, y_true)
    return float(tp / (tp + fn))