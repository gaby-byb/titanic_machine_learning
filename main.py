#imports

import pandas as pd
import numpy as np

#machine learning components
from sklearn.model_selection import train_test_split, GridSearchCV
#converter to scale of 0-1
from sklearn.preprocessing import MinMaxScaler
#the actual model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

#visualization tools
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("titanic.csv")
data.info()
#get the amount of missing data
print(data.isnull().sum())

# Data Cleaning 

def preprocess_data(df):
    df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True)

    df["Embarked"].fillna("S", inplace=True)
    df.drop(columns=["Embarked"], inplace=True)

    fill_missing_ages(df)

    # Convert Gender
    df["Sex"] = df["Sex"].map({'male':1, "female":0})

    ### Feature Engineering (creating new columns)
        #We create a 'IsAlone, 'FareBin' and 'AgeBin' to determine the social patterns of survivers
        #These can be use to predict how likely someone is to survive
    
    df["FamilySize"] = df["SibSp"] + df["Parch"]
    #is the passanger alone? if family size is 0, turn it to 0
    df["IsAlone"] = np.where(df["FamilySize"] == 0, 1, 0)
    #fill any missing fare data
    df["Fare"].fillna(df["Fare"].median(), inplace=True)
    df["FareBin"] = pd.qcut(df["Fare"], 4, labels=False)

    df["AgeBin"] = pd.cut(df["Age"], bins=[0,12,20,40,60, np.inf],labels=False)

    return df

# fill in missing ages - one median age per passenger class
def fill_missing_ages(df):

    #creating dictionary with the median age for each class
    age_fill_map = {}
    for pclass in df["Pclass"].unique():
        if pclass not in age_fill_map:
            age_fill_map[pclass] = df[df["Pclass"] == pclass]["Age"].median()
    # “Go through every row in the DataFrame.
    """If Age is missing, 
        replace it with the median age for that rowss passenger class.
        Otherwise, leave it as is.”
    """
    df["Age"] = df.apply(lambda row: age_fill_map[row["Pclass"]] 
    if pd.isnull(row["Age"]) else row["Age"], axis=1)

data = preprocess_data(data)

# Create Features / Target Variables (Make Flashcards)
X = data.drop(columns=["Survived"])
y = data["Survived"] #correct answer

#learning with 75% of the data, testing with 25%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# ML Preprocessing

scaler = MinMaxScaler() #makes all features comparable
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Hyperparemeter Tuning - knn 
def tune_model(X_train, y_train):
    param_grid = {
        "n_neighbors":range(1,21),
        "metric": ["euclidean", "manhattan", "minkowski"],
        "weights": ["uniform", "distance"]
    }

    model = KNeighborsClassifier()
    # find me the best settings with best results
    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_

best_model = tune_model(X_train, y_train)

# Predictions and Evaluate

def evaluate_model(model, X_test, y_test):
    #make predictions on  test set
    prediction = model.predict(X_test)
    #calculate the accuracy
    accuracy = accuracy_score(y_test, prediction)
    # return both and build a right vs wrong predictions
    matrix = confusion_matrix(y_test, prediction)
    return accuracy, matrix

accuracy, matrix = evaluate_model(best_model, X_test, y_test)

print(f'Accuracy: {accuracy*100:.2f}%')
print(f'Confusion Matrix: ')
print(matrix)

