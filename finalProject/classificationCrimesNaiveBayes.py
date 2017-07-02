import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv("crimes2016THEFTandBATTERYNOTNULL.csv",sep=",")

print("Dataset size",len(df))

#Splitting dependent and independent variables
X = df.iloc[:,1:9].values
y = df.iloc[:,0].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean',axis=0)
imputer = imputer.fit(X[:,1:9])
X[:,1:9] = imputer.transform(X[:,1:9])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 1)

classifier = GaussianNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print(accuracies.mean())
print(accuracies.std())