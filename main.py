print("Loading modules...")
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

print("Loading data...")
data = pd.read_csv('data/train.csv') #load dataset to data variable
X = data.drop(columns=["PassengerId", "Survived", "Name", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"]) #set X value to a table of all specified columns
y = data["Survived"] #set y value to the column of Survived

print("Training model...")
model = DecisionTreeClassifier()
model.fit(X.values,y.values) #train the model
user_name = input("Enter your name: ")
user_class = float(input("What passenger class are you (1, 2 or 3): "))
if user_class != 1 and user_class != 2 and user_class != 3:
  print("Invalid class. Please retry.")
  quit()
user_gender = input("What is your gender (male or female): ")
if user_gender != "male" and user_gender != "female" and user_gender != "m" and user_gender != "f":
  print("Invalid gender. Please enter either \"male\" or \"female\".")
  quit()
else:
  if user_gender == "male" or user_gender == "m":
    user_gender = 0
  else:
    user_gender = 1
user_age = float(input("What is your age: "))
predictions = model.predict([[user_class,user_gender,user_age]])
if predictions[0] == 1:
  print(f"You are likely to survive the Titanic! Congratulations {user_name}!")
else:
  print(f"You are unlikely to survive the Titanic {user_name}!")