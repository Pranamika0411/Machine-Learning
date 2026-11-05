import numpy as np
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

def compute_metrics(y_true, y_pred, y_proba):
    classes = np.unique(y_true)
    multi = len(classes) > 2
    avg = "weighted" if multi else "binary"
    auc = (roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted",
                         labels=sorted(classes)) if multi
           else roc_auc_score(y_true, y_proba[:, 1]))
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, average=avg, zero_division=0),
        "Recall": recall_score(y_true, y_pred, average=avg, zero_division=0),
        "F1": f1_score(y_true, y_pred, average=avg, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
