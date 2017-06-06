

import pandas as pd
import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar
from math import radians, cos, sin, asin, sqrt
from sklearn.metrics.scorer import  accuracy_score

def removeOutliers(data, m=3):
    new_data = data
    clean_data = new_data[np.abs(new_data-new_data.mean())<=(m*new_data.std())]
    return clean_data

def getHolidays(dates):
    holidaysVectorReturn  = []
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start='2015-01-01', end='2017-12-31').to_pydatetime()
    for idx, date in enumerate(dates):
         if date in holidays:
             holidaysVectorReturn.append(1)
         else:
             holidaysVectorReturn.append(0)
    return  holidaysVectorReturn

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


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles

    return c * r

def getCrimesByRadius(lon, lat, df, radius):
    for idx, row in df.iterrows():
        lon2 = df.iloc[idx]['Longitude'].astype(float)
        lat2 = df.iloc[idx]['Latitude'].astype(float)
        distance = haversine(lon,lat,lon2,lat2)
        if(distance>radius):
            df = df.drop(idx,axis=0)
    return df

def printAccuracy(y_test,y_pred):
    print("Accuracy: ",accuracy_score(y_test,y_pred))

#Fitting Naive Bayes to the Training set
def getClassifier(X_train, y_train):
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    return classifier

dataset = pd.read_csv("/Users/Jean/Documents/Software Engineering/UFG/mestrado/ARP/datasets/crimes-in-chicago/Chicago_Crimes_2012_to_2017.csv",sep=",")
#df = pd.read_csv("/Users/Jean/Documents/Software Engineering/UFG/mestrado/ARP/finalProject/datasets/smalldatasetcrimes.csv",sep=",")
dataset = dataset[dataset.Year == 2016]

types_to_save = ["THEFT",
                 "NARCOTICS",
                ]

dataset = dataset[dataset['Primary Type'].isin(types_to_save)]


#convert dates to pandas datetime format
dataset.Date = pd.to_datetime(dataset.Date, format='%m/%d/%Y %I:%M:%S %p')
# setting the index to be the date will help us a lot later on

#Get holidays
dataset.loc[:,'holiday'] = getHolidays(dataset['Date'])
dataset.loc[:, 'month'] = dataset['Date'].dt.month
dataset.loc[:,'day'] = dataset['Date'].dt.day
dataset.loc[:,'Period'] = getPeriodOfTheDay(dataset['Date'].dt.hour)


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
                       "Description",
                       "Year"]

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



#Remove empty rows

dataset = dataset.drop("Period_3",axis=1)
dataset.dropna(inplace=True)

print("Dataset size",len(dataset))
print(dataset.columns.tolist())


#Here we can  separate by radius and create a classifier for each radius
radius  = 5

y_predAccuracy  = []
y_testAccuracy = []

columnsLatLong = ["Latitude",
                  "Longitude"]

for idx,row in dataset.iterrows():
    size = len(dataset);
    lon = dataset.iloc[idx]['Longitude'].astype(float)
    lat = dataset.iloc[idx]['Latitude'].astype(float)
    X_test = row.iloc[1:11]
    y_test = row.iloc[0]
    y_testAccuracy.append(y_test)
    df = dataset[dataset.apply(lambda x: haversine(lon,lat,x['Longitude'],x['Latitude'])<radius, axis=1)]
   # df = dataset.drop(columnsLatLong, axis=1)
    #Splitting dependent and independent variables
    X_train = df.iloc[:,1:11].values
    y_train = df.iloc[:,0].values
    classifier = getClassifier(X_train,y_train)

    y_predAccuracy.append(classifier.predict(X_test))

    print("Idx: ",idx, " of a total of ",size)
    printAccuracy(y_predAccuracy,y_testAccuracy)

