# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 0.8.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.7.0
# ---

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn import datasets
from sklearn.linear_model import SGDClassifier

from mlxtend.plotting import plot_decision_regions

# get the data
iris = datasets.load_iris()
# print(iris.data)

# preprocess
## Get dataset with only the first two attributes, and only each 5th rows of the first 100rows(for the 2 target classes)
X = iris.data[:, :2]
print(X.shape)

## Get the respective target classes, but only the first two of them (0 and 1)
y = iris.target[:]
print(y.shape)

from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(criterion='gini',
                                n_estimators=25,
                                random_state=1,
                                n_jobs=2)

forest.fit(X, y)

plot_decision_regions(X, y,
                      clf=forest)

plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc='upper left')
plt.show()


