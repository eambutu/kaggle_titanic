import pandas

#Create pandas dataframe and assigns it to titanic variable
titanic = pandas.read_csv('test.csv')

#Print the first 5 rows of the dataframe
print(titanic.describe())

#Replace missing age data with the median age
titanic['Age'] = titanic['Age'].fillna(titanic['Age'].median())

#Replace sexes with numerical values
titanic.loc[titanic['Sex'] == 'male', 'Sex'] = 0
titanic.loc[titanic['Sex'] == 'female', 'Sex'] = 1

#Replace missing values and numberize values of 'Embarked'
titanic['Embarked'] = titanic['Embarked'].fillna('S')
titanic.loc[titanic['Embarked'] == 'S', 'Embarked'] = 0
titanic.loc[titanic['Embarked'] == 'C', 'Embarked'] = 1
titanic.loc[titanic['Embarked'] == 'Q', 'Embarked'] = 2

titanic['Fare'] = titanic['Fare'].fillna(titanic['Fare'].median())

#Print result
titanic.to_csv('test.csv')
