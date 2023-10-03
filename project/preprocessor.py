import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder

from project.params import *
from project.data import get_X_train_etc

def preproc_data():
    """
    - Creates a pipeline
    - Robust scale & One Hot encode data
    - Returns X_train, X_test transformed and y_train, y_test
    """

    X_train, X_test, y_train, y_test = get_X_train_etc()
    num_pipeline = make_pipeline(RobustScaler())
    cat_pipeline = make_pipeline(OneHotEncoder(handle_unknown="ignore"))
    preproc_pipeline = make_column_transformer((num_pipeline, NUM_FEATURES),
                                           (cat_pipeline, CAT_FEATURES))
    preproc_pipeline.fit(X_train)
    X_train_preproc = preproc_pipeline.transform(X_train)
    X_test_preproc = preproc_pipeline.transform(X_test)

    return X_train_preproc, X_test_preproc, y_train, y_test
