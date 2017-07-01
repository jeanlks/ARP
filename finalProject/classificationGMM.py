
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
import numpy as np

dataset = pd.read_csv("crimes2016OnlyLocations.csv",sep=",")

X = dataset.iloc[:,1:9].values
y = dataset.iloc[:,0].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 1)


lowest_bic = np.infty
bic = []

for n_components in range(1,3):
    classifier = GaussianMixture(n_components=n_components, covariance_type='full')
    classifier.fit(X_train, y_train)
    bic.append(classifier.bic(X))
    if bic[-1] < lowest_bic:
        lowest_bic = bic[-1]
        best_gmm = classifier


y_pred = best_gmm.predict(X_test)
score = accuracy_score(y_pred,y_test)

print(score)

# # Applying k-Fold Cross Validation
# from sklearn.model_selection import cross_val_score
# accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
# print(accuracies.mean())
# print(accuracies.std())