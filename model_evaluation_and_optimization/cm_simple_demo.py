import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.utils.multiclass import unique_labels
from sklearn import metrics


def print_confusion_matrix_seaborn(
    confusion_matrix,
    class_names,
    figsize = (10,7),
    fontsize=14):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.

    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix.
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.

    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig

def print_CM_as_heatmap(cm):
    # print with seaborn - needs df:
    df_cm = pd.DataFrame(cm, index = class_values, columns=class_values)
    # print_confusion_matrix_seaborn(cm, class_names, figsize = (10,7), fontsize=14)

    sns.set(font_scale=1)#for label size
    sns.heatmap(df_cm, annot=True, annot_kws={"size": 16})# font size

    plt.show()

def calc_precision(tp, y_pred):
    ''' Precision:
        the fraction of relevant instances (TP) among the retrieved instances (TP+FP)

        TP+FP = all positive classes, predicted my the model
     '''
    return tp / tp+fp

def calc_recall(tp, y_pred):
    ''' Recall:
        the fraction of the total amount of relevant instances that were actually retrieved

        TP + FN = All positive classes, as of ground truth.
    '''
    return tp / tp+fn

def calc_CM_params(y_true, y_pred):
    ''' Meaning:

        True  Positives (TP): classifier correctly predicted the positive (value=1) class
        False Positives (FP): classifier incorrectly predicted 1 (the ground truth was 0)
        True  Negative  (TN): classifier correctly predicted the negative (value=0) class
        False Negative  (FN): classifier incorrectly predicted 0 (the ground truth was 1)
    '''
    tp = ( (y_true==1) & (y_pred==1) ).sum()
    tn = ( (y_true==0) & (y_pred==0) ).sum()
    fp = ( (y_true==0) & (y_pred==1) ).sum()
    fn = ( (y_true==1) & (y_pred==0) ).sum()

    return {
        'TN': tn,
        'FP': fp,
        'FN': fn,
        'TP': tp,
    }

def print_CM(y_true, y_pred):
    ''' print CM in console

        sklearn ordering:
        CM = [
                [ TN FP ]
                [ FN TP ]
             ]
        Rows: Predicted
            0 - negative
            1 - positive
        Cols: Ground truth
            0 - negative
            1 - positive
    '''
    params = calc_CM_params(y_true, y_pred)
    for k, v in params.items():
        print(f"{k}: {v}")

    cm = confusion_matrix(y_true, y_pred)
    print(f"CM: \n{cm}")

if __name__ == "__main__":
    # imagine, we've train a classifier to clasify documents as :
    # relevant      - the positive class (value 1)
    # irrelevant    - the negative class (value 0)

    class_names  = ['irrelevant', 'relevant']
    class_values = [0, 1]

    # the ground truth:  3 - positive, 7 - negative:
    # the predicted:     4 - positive, 6 - negative
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    y_pred = np.array([1, 0, 0, 1, 1, 1, 0, 0, 0, 0])


    # print accuracy:
    acs = accuracy_score(y_true, y_pred)
    print(f'acs: {acs}')

    # print Confusion Matrix:
    print_CM(y_true, y_pred)

    d = calc_CM_params(y_true, y_pred)

    print(f"PR: { d['TP']/d['TP'] + d['TP'] }")
    print(f"PR: { d['TP']/d['TP'] + d['FN'] }")


