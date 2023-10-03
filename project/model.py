from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from project.data import get_X_train_etc
from project.preprocessor import transform_X


def initialize_model():
    """
    Return a random forest regressor
    """
    model = RandomForestRegressor(min_samples_leaf=3,
                                  min_samples_split=2,
                                  n_estimators=100)

    return model


def train_model():
    """
    Train model
    """
    X_train, X_test, y_train, y_test = get_X_train_etc()
    X_train_preproc = transform_X(X_train)
    X_test_preproc = transform_X(X_test)
    model = initialize_model()
    print("Model is fitting ⏳")
    model.fit(X_train_preproc, y_train)
    y_pred = model.predict(X_test_preproc)
    score = mean_absolute_error(y_test, y_pred)
    print(f"Model is well trained with a {round(score, 2)} score")

    return model
