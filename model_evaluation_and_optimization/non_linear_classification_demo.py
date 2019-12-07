import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn import metrics

import itertools

def plot_scater(**kwargs):
    colors_map = ['red', 'green']
    colors = list(map(lambda el: colors_map[el], kwargs['y']))

    if all(arg in kwargs for arg in ("x1","x2","y")):
        fig, ax = plt.subplots()

        ax.scatter(x=kwargs["x1"], y=kwargs["x2"],c=colors, cmap='viridis')

        ax.set(xlabel='x1', ylabel='x2')
        ax.legend(loc='center right')

        plt.show()
    else:
        print('Error')

def evaluate(y_pred, y_test, clf_name):
    # Model Accuracy:
    print(f"{clf_name} Accuracy:", metrics.accuracy_score(y_test, y_pred))

    # Model Precision:
    print(f"clf_name Precision:",metrics.precision_score(y_test, y_pred))

    # Model Recall:
    print(f"clf_name Recall:",metrics.recall_score(y_test, y_pred))


def print_confusion_matrix(clf_name, y_true, y_pred):
    print(f"Confusion matrix for {clf_name}:")
    print(metrics.confusion_matrix(y_true, y_pred, labels=[0,1]))


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def plot_confusion_matrix_iep(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    scikit-learn CM follows next convention:

                    Predicted
            ----------------------------
            |
    True:   ----------------------------


    """


    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        # print(format(cm[i, j], fmt))
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 verticalalignment = "bottom",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    ### Prepate the data
    X = np.array([ [1,1], [5,1], [1,5], [5,5]])
    y = [1, 0, 0, 1]

    class_names = [1,0]


    # print(f'X shape: {X.shape}')
    # plot_scater(x1=X[:, 0], x2 = X[:, 1], y = y)

    ### prepare Train and Test sets:
    X_train = X_test = X
    y_train = y_test = y

    ### fit and predict:
    clfs = {
        'LR': LogisticRegression(random_state=0),
        'KNN': KNeighborsClassifier(n_neighbors=1)
    }

    for clf_name,clf in clfs.items():
        y_pred = clf.fit(X_train, y_train).predict(X_test)
        print(f"\nEvaluate {clf_name}:")
        print(f'y_pred: {y_pred}')
        print(f'y_test: {y_test}')

        # plot_confusion_matrix(y_test, y_pred, classes=[1,0], title='Confusion matrix, without normalization')
        # cm = confusion_matrix(y_test, y_pred)
        # print(cm)
        # np.set_printoptions(precision=2)
        # plt.figure()


        # plot_confusion_matrix_iep(cm, classes=class_names, title=f"CM for {clf_name}")

        from sklearn.metrics import plot_confusion_matrix
        disp = plot_confusion_matrix(clf, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 normalize=None)

        print(disp.confusion_matrix)
        plt.show()

        evaluate(y_pred, y_test, clf_name)


    # # print_confusion_matrix('LR', y_test, lr_y_pred)





