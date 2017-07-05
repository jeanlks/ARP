import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import  accuracy_score
from sklearn.metrics import roc_auc_score
dataset = pd.read_csv("crimes2016X.csv",sep=",")
from sklearn.metrics import confusion_matrix

#Splitting dependent and independent variables
X = dataset.iloc[:,1:10].values
y = dataset.iloc[:,0].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean',axis=0)
imputer = imputer.fit(X[:,1:10])
X[:,1:10] = imputer.transform(X[:,1:10])

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 1)


classifier = XGBClassifier(max_delta_step=1)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

score = accuracy_score(y_test,y_pred)

print(score)

# # Applying k-Fold Cross Validation
# from sklearn.model_selection import cross_val_score
# accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
# print(accuracies.mean())
# print(accuracies.std())


