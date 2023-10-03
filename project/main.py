import pandas as pd
import pickle

from project.data import get_X_train_etc
from project.preprocessor import transform_X
from project.model import initialize_model, train_model
from project.model import train_model


# def main():
#     X_train, X_test, y_train, y_test = get_X_train_etc()
#     X_train_preproc = transform_X(X_train)
#     X_test_preproc = transform_X(X_test)
#     model = train_model()

#     return model


def make_prediction():
    """
    Takes data and out a prediction
    """
    model = train_model()
    age = int(input("Enter age of client: "))
    sex = input("Enter sex of client [male/female]: ")
    bmi = float(input("Enter BMI of client [XX.X]: "))
    children = int(input("Enter number of childrens: "))
    smoker = input("Is client a smoker? [yes / no] ")
    region = input("Enter client's region: ")
    data = ({"age": [age],
        "sex": [sex],
        "bmi": [bmi],
        "children": [children],
        "smoker": [smoker],
        "region": [region]})
    X_new = pd.DataFrame(data = data)
    X_new_preproc = transform_X(X_new)
    prediction = model.predict(X_new_preproc)

    print(f"Healthcare costs prediction is: {round(prediction[0], 2)}$")


if __name__ == "__main__":
    make_prediction()
else:
    import traceback
    traceback.print_exc()
