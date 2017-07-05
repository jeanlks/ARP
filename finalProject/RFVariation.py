import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import  matplotlib.pyplot as plt

dataset = pd.read_csv("crimes2016THEFTandBATTERYandASSAULT.csv",sep=",")
#Splitting dependent and independent variables
X = dataset.iloc[:,1:9].values
y = dataset.iloc[:,0].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean',axis=0)
imputer = imputer.fit(X[:,1:9])
X[:,1:9] = imputer.transform(X[:,1:9])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)


accuracies = []
for i in range(1,100):
    classifier = RandomForestClassifier(n_estimators=i, class_weight='balanced',criterion='entropy',random_state=0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracies.append(accuracy_score(y_pred,y_test))

plt.plot(accuracies)
plt.title("THEFT and BATTERY and ASSAULT Ktree variation")
plt.show()


