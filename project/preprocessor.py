from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder

from project.params import *
from project.data import get_X_train_etc


def get_preproc_pipeline():
    """
    Create and return preprocessing pipeline
    """
    num_pipeline = make_pipeline(RobustScaler())
    cat_pipeline = make_pipeline(OneHotEncoder(handle_unknown="ignore"))
    preproc_pipeline = make_column_transformer((num_pipeline, NUM_FEATURES),
                                           (cat_pipeline, CAT_FEATURES))

    return preproc_pipeline


def fit_preproc_pipeline():
    """
    Fit preprocessing pipeline
    """
    preproc_pipeline = get_preproc_pipeline()
    X_train, X_test, y_train, y_test = get_X_train_etc()
    preproc_pipeline.fit(X_train)

    return preproc_pipeline


def transform_X(X):
    """
    Transform X with preprocessing fitted pipeline
    """
    preproc_pipeline = fit_preproc_pipeline()
    X_preproc = preproc_pipeline.transform(X)

    return X_preproc
