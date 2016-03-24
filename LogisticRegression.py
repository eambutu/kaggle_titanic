import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation

titanic = pd.read_csv('train_cleaned.csv')
predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

#Initialize algorithm class
alg = LogisticRegression(random_state=1)

#Compute accuracy score across all folds
scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic['Survived'], cv=3)

#Mean of scores across three folds
print(scores.mean())
