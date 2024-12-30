print("Loading modules...")
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

def gatherUserInfo():
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

print("Loading data...")
data = pd.read_csv('data/train.csv') #load dataset to data variable
X = data.drop(columns=["PassengerId", "Survived", "Name", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"]) #set X to all columns not specified
y = data["Survived"] #set y value to the column of Survived
#compare the data of the x and y values: who survived (y) and what their data was (x)

print("Training model...")
model = DecisionTreeClassifier()
model.fit(X.values,y.values) #train the model
user_data = gatherUserInfo()
predictions = model.predict([[user_data[1],user_data[2],user_data[3]]])
if predictions[0] == 1:
  print(f"You are likely to survive the Titanic! Congratulations {user_data[0]}!")
else:
  print(f"You are unlikely to survive the Titanic {user_data[0]}!")