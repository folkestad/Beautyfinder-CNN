from sklearn.metrics import classification_report

def classification_report(pred_targets, true_targets):
    return classification_report(true_targets, pred_targets)