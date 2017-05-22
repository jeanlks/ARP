import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.metrics import accuracy_score

dataset = pd.read_csv("/Users/Jean/Documents/Software Engineering/UFG/mestrado/ARP/datasets/crimes-in-chicago/Chicago_Crimes_2012_to_2017.csv",sep=",")
#dataset = pd.read_csv("/Users/Jean/Documents/Software Engineering/UFG/mestrado/ARP/datasets/smalldatasetcrimes.csv",sep=",")

#Exclude not needed columns
columnsForExclusion = ['Ward',
                       'FBI Code',
                       "Arrest",
                       "Description",
                       "Case Number",
                       "IUCR",
                       "Beat",
                       "Updated On",
                       "Date",
                       "Block",
                       "Location Description",
                       "Unnamed: 0",
                       "Location",
                       "ID"]

dataset = dataset.drop(columnsForExclusion,axis=1)




#Get dummies for columns
columnsForDummies = ["Domestic"]
dataset = pd.get_dummies(dataset,columns=columnsForDummies)

dataset['Primary Type'] = dataset['Primary Type'].astype('category')
dataset['Primary Type'] = dataset['Primary Type'].cat.codes



#Rearranging columns
cols = ['Primary Type'] + [col for col in dataset if col != 'Primary Type']
dataset = dataset[cols]

#Remove empty rows

dataset.dropna(inplace=True)

#Splitting dependent and independent variables
X = dataset.iloc[:,1:10].values
y = dataset.iloc[:,0].values

print(dataset.columns.tolist())
print(dataset.columns[dataset.isnull().any()])


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# # Feature Scaling
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)

# # Fitting Naive Bayes to the Training set
# from sklearn.naive_bayes import GaussianNB
# classifier = GaussianNB()
# classifier.fit(X_train, y_train)
# Fitting SVM to the Training set

from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)



print(accuracy_score(y_test,y_pred))
print(len(y_train))
print(len(y_test))

