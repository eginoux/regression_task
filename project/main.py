import pandas as pd

from project.preprocessor import transform_X
from project.model import train_model


def make_prediction():
    """
    - Asks for input data
    - Creates a data frame
    - Preprocess data
    - Makes a prediction
    """
    model = train_model()
    age = int(input("Enter age of client: "))
    while age not in [i for i in range(18, 100)]:
        age = int(input("Enter a valid age between 18 and 99: "))
    sex = input("Enter sex of client [male/female]: ")
    while sex not in ["male", "female"]:
        sex = input("Enter a valid sex [male/female]: ")
    bmi = float(input("Enter BMI of client [XX.X]: "))
    while bmi not in [i for i in range(10, 50)]:
        bmi = float(input("Enter a valid BMI [XX.X]: "))
    children = input("Enter a numeric number of childrens: ")
    while children.isdigit() == False:
        children = input("Enter a numeric value for children(s): ")
    children = int(children)
    while (children not in [i for i in range(0, 15)]):
        children = int(input("Enter a valid number of childrens: "))
    smoker = input("Is client a smoker? [yes / no] ")
    while smoker not in ["yes", "no"]:
        smoker = input("Enter a valid attribute [yes / no]: ")
    region = input("Enter client's region: ")
    while not region in ["northeast", "southeast", "northwest", "southwest"]:
        print("Accepted regions are northeast, southeast, northwest, southwest")
        region = input("Enter a valid region: ")

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
