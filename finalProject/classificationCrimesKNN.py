import pandas as pd

dataset = pd.read_csv("crimes2016THEFTandBATTERYandASSAULT.csv",sep=",")


#Splitting dependent and independent variables
X = dataset.iloc[:,1:13].values
y = dataset.iloc[:,0].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 1)

#KNN Classifier
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 100, metric = 'minkowski',p=1)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)


# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print(accuracies.mean())
print(accuracies.std())


