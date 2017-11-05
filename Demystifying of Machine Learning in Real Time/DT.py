# import the packages
import os
import pandas as pd
from sklearn import tree
import io
import pydot
 
# Create current directory
os.getcwd
os.chdir("C:/Users/sukonda/Desktop/GDG")
 
# Read the train data.
titanic_train = pd.read_csv("train.csv")
print (titanic_train.info())
 
# EDA on train data
# apply one hot encoding for the below columns
titanic_train1 = pd.get_dummies(titanic_train, columns=['Pclass', 'Sex', 'Embarked'])
 
# verify the train directory
titanic_train1.shape
titanic_train1.info()
 
# Drop the columns and create the actual train data.
X_train = titanic_train1.drop(['PassengerId','Age','Cabin','Ticket', 'Name','Survived'], 1)
 
# Target column.
y_train = titanic_train['Survived']
 
# Build the model.
dt = tree.DecisionTreeClassifier()
 
# fit the model with train data.
dt.fit(X_train,y_train)
 
# Create Graphical view of the tree.
dot_data = io.StringIO()
tree.export_graphviz(dt, out_file = dot_data, feature_names = X_train.columns)
graph = pydot.graph_from_dot_data(dot_data.getvalue())[0]
graph.write_pdf("DescisionTree.pdf")
 
# Read test data.
# EDA on test data
titanic_test = pd.read_csv("test.csv")
titanic_test1 = pd.get_dummies(titanic_test, columns=['Pclass', 'Sex', 'Embarked'])
x_test = titanic_test1.drop(['PassengerId','Age','Cabin','Ticket', 'Name'], 1)
 
# run the model on test data.
titanic_test['Survived'] = dt.predict(x_test)
 
# export test data.
titanic_test.to_csv("Submission.csv", columns = ['PassengerId','Survived'] , index = False)