import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix
from sklearn import metrics


# %matplotlib inline
def load_data_as_nparrays(db):
    X = db.data[start:end, 0:2]
    y = db.target[start:end]

    print(f'X shape: {X.shape}')
    print(f'y shape: {y.shape}')

    return (X,y)


def load_data_as_df(db):
    df = pd.DataFrame(data={
        'x1': db.data[start:end,0],
        'x2': db.data[start:end,1],
        'y':db.target[start:end]})

    print(f'df.shape: {df.shape}')

    ### sustitute class 1 => 0, class 2 = > 1

    # with map and lambda - useful for more complicated cases
    # df.y = df.y.map( lambda x: 0 if x==1 else 1)

    # map can also accept disctionary
    df.y = df.y.map( {1: 0, 2:1})
    print(f'y unique values: {np.unique(df.y, return_counts=True)}')

    return df

def get_class_names():
    return ('non-terorist', 'terorist')

def plot_scater(**kwargs):
    if all(arg in kwargs for arg in ("x1","x2","y")):
        # print(f'x1 = {kwargs["x1"]}')
        # print(f'x2 = {kwargs["x2"]}')
        # print(f'y = {kwargs["y"]}')
        plt.scatter(x=kwargs["x1"], y=kwargs["x2"],c=kwargs["y"], cmap='viridis')
        plt.show()


if __name__ == "__main__":
    # will use the iris data, but in different context - more clear for CM:
    iris = datasets.load_iris()

    # start and end values for row indexes to take data from:
    start = 80
    end = 110

    df = load_data_as_df(iris)
    class_names = get_class_names()

    # print(df.iloc[:, 0])

    plot_scater(x1=df.iloc[:,0], x2=df.iloc[:,1],y=df.y)



