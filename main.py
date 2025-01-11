print("Loading modules...")  # import modules
import numpy as np
import pandas as pd
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer


def gatherUserInfo():  # function to get user info
    user_name = input("Enter your name: ")
    user_class = int(input("What passenger class are you (1, 2 or 3): "))
    if user_class not in [1, 2, 3]:
        print("Invalid class. Please retry.")
        quit()
    user_gender = input("What is your gender (male or female): ").lower()
    valid_genders = ["male", "m", "man", "female", "f", "woman"]
    if user_gender not in valid_genders:
        print("Invalid gender. Please enter either \"male\" or \"female\".")
        quit()
    else:
        if user_gender == "male" or user_gender == "m":
            user_gender = 0
        else:
            user_gender = 1
    user_age = int(input("What is your age: "))
    result = [user_name, user_class, user_gender, user_age]
    return result


def cls():  # function to clear console
    os.system(
        'cls' if os.name == 'nt' else 'clear')  # has to use different clear commands depending on operating system


# main program
while True:
    cls()  # clears console
    print("Loading data...")
    data = pd.read_csv('data/train.csv')  # load dataset to data variable
    imp = SimpleImputer(missing_values=np.nan, strategy='median')  # set imputer to a variable to call on
    X = data.drop(columns=["PassengerId", "Name", "SibSp", "Parch", "Ticket", "Fare", "Cabin",
                           "Embarked"])  # set X to all columns not specified
    X["Age"] = imp.fit_transform(X[["Age"]]).ravel()  # fill all missing values in age column
    np_values = X.values  # set X to NumPy array
    np_new_values = np.delete(np_values, 0, 1)  # Delete unnecessary column
    X = pd.DataFrame(np_new_values, columns=["Pclass", "Sex", "Age"])  # convert back to pandas dataframe
    y = data["Survived"]  # set y value to the column of Survived
    # compare the data of the x and y values: who survived (y) and what their data was (x)

    print("Training model...")
    model = DecisionTreeClassifier()  # set the decision tree (neural network)
    model.fit(X.values, y.values)  # train the model
    user_data = gatherUserInfo()  # set user info to a variable
    predictions = model.predict([[user_data[1], user_data[2], user_data[3]]])  # get predictions
    if predictions[0] == 1:  # print them
        print(f"You are likely to survive the Titanic! Congratulations {user_data[0]}!")
    else:
        print(f"You are unlikely to survive the Titanic {user_data[0]}!")
    input("Press enter to reset!")