import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold

titanic = pd.read_csv('train_cleaned.csv')

#Columns that we use for prediction
predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

#Initialize algorithm class
alg = LinearRegression()

#Generate cross validation folds. Returns the row indices corresponding to train and test.
#Set random_state to ensure we get the same splits every time we run this.
kf = KFold(titanic.shape[0], n_folds=3, random_state=1)

predictions = []
for train, test in kf:
    #The predictors we use to train. We only take rows from train.
    train_predictors = (titanic[predictors].iloc[train,:])
    #Target we're using to train.
    train_target = titanic['Survived'].iloc[train]
    #Train using the predictors and target
    alg.fit(train_predictors, train_target)
    #Now make predictions
    test_predictions = alg.predict(titanic[predictors].iloc[test,:])
    predictions.append(test_predictions)
    
#Concatenate predictions in three arrays into one
predictions = np.concatenate(predictions, axis=0)

#Map predictions to outcomes
predictions[predictions > 0.5] = 1
predictions[predictions <= 0.5] = 0

#Calculate accuracy
accuracy = sum(predictions[predictions == titanic['Survived']]) / len(predictions)
print(accuracy)
