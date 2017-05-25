import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.metrics import accuracy_score


dataset = pd.read_csv("/Users/Jean/Documents/Software Engineering/UFG/mestrado/ARP/datasets/crimes-in-chicago/Chicago_Crimes_2012_to_2017.csv",sep=",")
#dataset = pd.read_csv("/Users/Jean/Documents/Software Engineering/UFG/mestrado/ARP/datasets/smalldatasetcrimes.csv",sep=",")

# convert dates to pandas datetime format
dataset.Date = pd.to_datetime(dataset.Date, format='%m/%d/%Y %I:%M:%S %p')
# setting the index to be the date will help us a lot later on

dataset.index = pd.DatetimeIndex(dataset.Date)
#
# dataset['date'] = str(dataset['Date'].dt.date)
# dataset['time'] = str(dataset['Date'].dt.time)


#Select Year
#dataset = dataset[dataset.Year == 2016]

#Exclude not needed columns
columnsForExclusion = ['Ward',
                       'FBI Code',
                       "Arrest",
                       "Case Number",
                       "IUCR",
                       "Beat",
                       "Updated On",
                       "Unnamed: 0",
                       "Block",
                       "Date",
                       "Location",
                       "ID"]

dataset = dataset.drop(columnsForExclusion,axis=1)



#Get dummies and categorical values for columns
columnsForDummies = ["Domestic"]
dataset = pd.get_dummies(dataset,columns=columnsForDummies)

dataset['Primary Type'] = pd.Categorical(dataset['Primary Type'])
dataset['Location Description'] = pd.Categorical(dataset['Location Description'])
dataset['Description']          = pd.Categorical(dataset['Description'])

dataset['Primary Type'] = dataset['Primary Type'].cat.codes
dataset['Location Description']  = dataset['Location Description'] .cat.codes
dataset['Description']  = dataset['Description'] .cat.codes


#Rearranging columns
cols = ['Primary Type'] + [col for col in dataset if col != 'Primary Type']
dataset = dataset[cols]


print(dataset.columns.tolist())
dataset.info()

print("Dataset size",len(dataset))
#Remove empty rows

dataset.dropna(inplace=True)

#Splitting dependent and independent variables
X = dataset.iloc[:,1:11].values
y = dataset.iloc[:,0].values



# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# # Feature Scaling
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)
#
#
# #Applying PCA
# from sklearn.decomposition import PCA
# pca = PCA(n_components = 2)
# X_train = pca.fit_transform(X_train)
# X_test = pca.transform(X_test)
# explained_variance = pca.explained_variance_ratio_
#
#
# #Fitting Naive Bayes to the Training set
# from sklearn.naive_bayes import GaussianNB
# classifier = GaussianNB()
# classifier.fit(X_train, y_train)
#
#
# # Predicting the Test set results
# y_pred = classifier.predict(X_test)
#
# print(accuracy_score(y_test,y_pred))
#
#
#
#
# import matplotlib.pyplot as plt
# mu, sigma = 200, 25
# df = dataset.sort(columns='Primary Type',ascending=True).copy()
#
# n, bins, patches = plt.hist(df['Primary Type'])
# plt.show()