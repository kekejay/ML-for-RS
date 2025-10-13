import random
import numpy as np
import torch
import yaml
import joblib
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, confusion_matrix,
    recall_score, roc_auc_score, average_precision_score
)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_yaml(d, path):
    with open(path, "w") as f:
        yaml.dump(d, f)

def save_dl_model(model, path):
    torch.save(model.state_dict(), path)

def save_ml_model(model, path):
    joblib.dump(model, path)
    
def evaluate(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics = {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average='binary'),
        "precision": precision_score(y_true, y_pred),
        "sensitivity": recall_score(y_true, y_pred),
        "specificity": tn / (tn + fp),
        "auroc": roc_auc_score(y_true, y_prob),
        "auprc": average_precision_score(y_true, y_prob)
    }    
    return metrics

def defined_metric_score(score_metric, y_true, y_pred, y_proba):
    tn, fp, _, _ = confusion_matrix(y_true, y_pred).ravel()
    if score_metric == "auroc":
        return roc_auc_score(y_true, y_proba)
    elif score_metric == "accuracy":
        return accuracy_score(y_true, y_pred)
    elif score_metric == "f1":
        return f1_score(y_true, y_pred, average="binary")
    elif score_metric == "auprc":
        return average_precision_score(y_true, y_proba)
    elif score_metric == "precision":
        return precision_score(y_true, y_pred)
    elif score_metric == "sensitivity":
        return recall_score(y_true, y_pred)
    elif score_metric == "specificity":
        return tn / (tn + fp)                          
    else:
        raise ValueError(f"Unsupported score_metric: {score_metric}")
