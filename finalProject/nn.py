import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score
def removeOutliers(data, m=3):
    new_data = data
    clean_data = new_data[np.abs(new_data-new_data.mean())<=(m*new_data.std())]
    return clean_data

#PERIODS OF THE DAY 1 = MORNING 2 = AFTERNOON AND 3 = NIGHT
def getPeriodOfTheDay(vectorHour):
    periods = []
    for hour in vectorHour:
        if(hour>12 and hour<=20):
            periods.append(2)
        if(hour > 6 and hour<=12):
            periods.append(1)
        if(hour >20 and hour<=23):
            periods.append(3)
        if(hour>=0 and hour<=6):
            periods.append(3)
    return  periods


dataset = pd.read_csv("/Users/Jean/Documents/Software Engineering/UFG/mestrado/ARP/datasets/crimes-in-chicago/Chicago_Crimes_2012_to_2017.csv",sep=",")
#df = pd.read_csv("/Users/Jean/Documents/Software Engineering/UFG/mestrado/ARP/finalProject/datasets/smalldatasetcrimes.csv",sep=",")
dataset = dataset[dataset.Year == 2016]

# types_to_save = ["THEFT",
#                     "HOMICIDE"]
#
# dataset = dataset[dataset['Primary Type'].isin(types_to_save)]


#convert dates to pandas datetime format
dataset.Date = pd.to_datetime(dataset.Date, format='%m/%d/%Y %I:%M:%S %p')
# setting the index to be the date will help us a lot later on



dataset.loc[:, 'month'] = dataset['Date'].dt.month
dataset.loc[:,'day'] = dataset['Date'].dt.day
dataset.loc[:,'Period'] = getPeriodOfTheDay(dataset['Date'].dt.hour)

# print(len(dataset['Date'].dt.hour))
# print(len( getPeriodOfTheDay(dataset['Date'].dt.hour)))
#Exclude not needed columns
columnsForExclusion = ['Ward',
                       'FBI Code',
                       "Arrest",
                       "Case Number",
                       "IUCR",
                       "Beat",
                       "Updated On",
                       "Unnamed: 0",
                       "Date",
                       "Location",
                       "ID",
                        "Block",
                       "X Coordinate",
                       "Y Coordinate",
                       "District",
                       "Community Area",
                       "Description"]

dataset = dataset.drop(columnsForExclusion,axis=1)


#Get dummies and categorical values for columns
columnsForDummies = ["Period"]
dataset = pd.get_dummies(dataset,columns=columnsForDummies)

dataset['Primary Type'] = pd.Categorical(dataset['Primary Type'])
dataset['Location Description'] = pd.Categorical(dataset['Location Description'])
dataset['Domestic'] = pd.Categorical(dataset["Domestic"])


dataset['Primary Type'] = dataset['Primary Type'].cat.codes
dataset['Location Description']  = dataset['Location Description'].cat.codes
dataset['Domestic'] = dataset['Domestic'].cat.codes



#Rearranging columns
cols = ['Primary Type'] + [col for col in dataset if col != 'Primary Type']
dataset = dataset[cols]


print(dataset.columns.tolist())

print("Dataset size",len(dataset))
#Remove empty rows


dataset.dropna(inplace=True)

#Splitting dependent and independent variables
X = dataset.iloc[:,1:10].values
y = dataset.iloc[:,0].values



# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 1)



# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 7, kernel_initializer = 'uniform', activation = 'relu', input_dim = 9))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 7, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)


# Predicting the Test set results
y_pred = classifier.predict(X_test)


print(accuracy_score(y_test,y_pred))
