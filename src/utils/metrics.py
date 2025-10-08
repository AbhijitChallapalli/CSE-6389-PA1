import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score

def compute_basic_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    acc = accuracy_score(y_true, y_pred)
    return {"confusion_matrix": cm, "accuracy": acc}
