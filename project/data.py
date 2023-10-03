import pandas as pd
from sklearn.model_selection import train_test_split
from project.params import *

def get_dataframe():
    """
    Transform csv to dataframe and return it
    """
    path = DATA_PATH
    df = pd.read_csv(path)

    return df

def get_X_y():
    """
    Takes a df and return X & y
    """
    df = get_dataframe()
    X = df.drop(columns = "expenses")
    y = df.expenses

    return X, y

def get_X_train_etc(X, y, test_size = 0.2, random_state = 1):
    """
    Get X_train, X_test, y_train & y_test
    """
    X, y = get_X_y()
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=test_size,
                                                        random_state=random_state)

    return X_train, X_test, y_train, y_test
