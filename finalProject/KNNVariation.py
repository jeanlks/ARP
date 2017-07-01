import pandas as pd
from  sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

dataset = pd.read_csv("crimes2016THEFTandBATTERYandASSAULT.csv",sep=",")


#Splitting dependent and independent variables
X = dataset.iloc[:,1:13].values
y = dataset.iloc[:,0].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 1)


accuracies  = []

#KNN Classifier
for i in range(1,100):
    classifier = KNeighborsClassifier(n_neighbors = i, metric = 'minkowski',p=1)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracies.append(accuracy_score(y_test,y_pred))


plt.plot(accuracies)
plt.title("Theft and Battery and ASSAULT K variation")
plt.show()