#Todo fragezeichen vlt auschsließen oder so

import numpy, pandas as pd, sklearn

# Read data
dataset = pd.read_csv('adult.data', names = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','yearly-income'])
datasetTest = pd.read_csv('adultModified.test', names = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','yearly-income'])

from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
  
classifier = DecisionTreeClassifier(random_state=0)

#Convert string data into int data
col=list(dataset.columns)
for i in range(len(col)):
    if i not in (0,2,4,10,11,12):
        dataset[col[i]] = preprocessing.LabelEncoder().fit_transform(dataset[col[i]])
        
col2=list(datasetTest.columns)
for i in range(len(col2)):
    if i not in (0,2,4,10,11,12):
        datasetTest[col2[i]] = preprocessing.LabelEncoder().fit_transform(datasetTest[col2[i]])

print(dataset)

X = dataset.drop('yearly-income', axis=1) # select columns 0 through 13
Y = dataset['yearly-income'] # select column 14, the yearly income

classifier.fit(X, Y)

#cross_val_score(clf, X, Y, cv=10)

X_test = datasetTest.drop('yearly-income', axis=1) # select columns 0 through 1
y_test = datasetTest['yearly-income'] # select column 14, the yearly income
print(X_test)
y_pred = classifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
