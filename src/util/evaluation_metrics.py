import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, recall_score, roc_auc_score, accuracy_score


def compute_classification_report(y_true: np.array, y_pred: np.array):
    prestations = classification_report(y_true, y_pred)
    matrix = pd.crosstab(y_true, y_pred, rownames=['Real'], colnames=['Predicted'], margins=True)
    return matrix


def get_metric_classification(scoring_metric, n_classes, average='macro'):

    if n_classes == 2:
        return scoring_metric
    else: # multiclass
        if scoring_metric == 'roc_auc':
            return '{}_{}'.format(scoring_metric, 'ovo')
        elif scoring_metric == 'f1':
            return '{}_{}'.format(scoring_metric, average)


def compute_classification_prestations_v2(y_true: np.array,
                                          y_pred: np.array,
                                          class_names: np.array,
                                          average='micro',
                                          verbose=False,
                                          save_confusion_matrix=False
                                          ) -> (float, float, float, float):

    if len(np.unique(y_pred)) != len(class_names):
        y_pred = np.where(y_pred >= 0.5, 1.0, y_pred)
        y_pred = np.where(y_pred < 0.5, 0.0, y_pred)

    if verbose:
        print(classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)

    if save_confusion_matrix:
        plot_confusion_matrix(cm, class_names, save_confusion_matrix)

    n_classes = len(set(y_true))

    if n_classes == 2:

        return {'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred),
                'specificity': specificity_score(y_true, y_pred),
                'recall': recall_score(y_true, y_pred),
                'roc_auc': roc_auc_score(y_true, y_pred),
                'f1': f1_score(y_true, y_pred)
        }

    else:
        return compute_multiclass_metrics(y_true, y_pred, average)



# CNN model utils
def compute_classification_prestations_cnn(y_true: np.array, y_pred: np.array) -> (float, float, float, float):
    tn, fp, fn, tp = confusion_matrix(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1)).ravel()
    acc_val = accuracy_score(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))
    specificity_val = tn / (tn + fp)
    recall_val = recall_score(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))
    roc_val = roc_auc_score(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))

    return acc_val, specificity_val, recall_val, roc_val


def compute_classification_prestations(y_true: np.array, y_pred: np.array) -> (float, float, float, float):

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    acc_val = accuracy_score(y_true, y_pred)
    specificity_val = tn / (tn + fp)
    recall_val = recall_score(y_true, y_pred)
    roc_val = roc_auc_score(y_true, y_pred)

    return acc_val, specificity_val, recall_val, roc_val