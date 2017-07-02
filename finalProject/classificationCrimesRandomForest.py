import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

dataset = pd.read_csv("crimes2016THEFTandBATTERY.csv",sep=",")

#Splitting dependent and independent variables
X = dataset.iloc[:,1:13].values
y = dataset.iloc[:,0].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 1)

classifier = RandomForestClassifier(n_estimators=200, class_weight='balanced')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
score = accuracy_score(y_pred,y_test)

print(score)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print(accuracies.mean())
print(accuracies.std())