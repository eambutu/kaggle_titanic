import pandas as pd
from sklearn.linear_model import LogisticRegression

titanic = pd.read_csv('train_cleaned.csv')
titanic_test = pd.read_csv('test.csv')

predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

alg = LogisticRegression(random_state=1)

# Train the model using all the training data
alg.fit(titanic[predictors], titanic['Survived'])

# Make predictions
predictions = alg.predict(titanic_test[predictors])

# Create new dataframe only with columns that Kaggle wants
submission = pd.DataFrame({
        'PassengerId': titanic_test['PassengerId'], 
        'Survived': predictions
    })

submission.to_csv('submission.csv', mode='w', index=False)
