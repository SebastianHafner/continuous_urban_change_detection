import numpy as np


def compute_change_f1_score(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    p = compute_change_precision(y_pred, y_true)
    r = compute_change_recall(y_pred, y_true)
    return 2 * (p * r) / (p + r)


def compute_true_positives(y_pred: np.ndarray, y_true: np.ndarray) -> int:
    return int(np.sum(np.logical_and(y_pred, y_true)))


def compute_false_positives(y_pred: np.ndarray, y_true: np.ndarray) -> int:
    return int(np.sum(np.logical_and(y_pred, np.logical_not(y_true))))


def compute_false_negatives(y_pred: np.ndarray, y_true: np.ndarray):
    return int(np.sum(np.logical_and(np.logical_not(y_pred), y_true)))


def compute_change_precision(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    y_pred = y_pred > 1
    y_true = y_true > 1
    tp = compute_true_positives(y_pred, y_true)
    fp = compute_false_positives(y_pred, y_true)
    return float(tp / (tp + fp))


def compute_change_recall(y_pred: np.ndarray, y_true: np.ndarray):
    y_pred = y_pred > 1
    y_true = y_true > 1
    tp = compute_true_positives(y_pred, y_true)
    fn = compute_false_negatives(y_pred, y_true)
    return float(tp / (tp + fn))