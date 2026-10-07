import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from IPython.display import display

def random_splitter(x, y, train_percentage, random_state=32):
    rng = np.random.default_rng(random_state)
    indices = np.arange(len(x))

    PERCENTAGE = train_percentage
    SIZE = int(indices.size * (PERCENTAGE / 100))

    train_indices = rng.choice(indices, size=SIZE, replace=False)
    test_indices = np.setdiff1d(indices, train_indices)

    x_train = x[train_indices]
    x_test = x[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    return x_train, x_test, y_train, y_test

iris = load_iris()
x = iris.data
feature_names = iris.feature_names
y = iris.target
df = pd.DataFrame(x, columns=feature_names)
df["target"] = y

x_train, x_test, y_train, y_test = random_splitter(x, y, 70)

x_train_df = pd.DataFrame(x_train, columns=iris.feature_names)
x_test_df = pd.DataFrame(x_test, columns=iris.feature_names)

y_train_df = pd.DataFrame(y_train, columns=["target"])
y_test_df = pd.DataFrame(y_test, columns=["target"])
display(x_train_df.tail())
display(y_train_df.tail())
display(x_test_df.tail())
display(y_test_df.tail())
