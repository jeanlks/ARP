import pandas as pd
from pandas.tseries.holiday import  USFederalHolidayCalendar
from math import radians, cos, sin, asin, sqrt

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

def removeOutliers(df,classes,radius=10):
 dfReturn = pd.DataFrame()
 for classe in classes:
    print(classe)
    dfMean =  df[df['Primary Type']==classe]
    meanLon = dfMean['Longitude'].mean()
    meanLat = dfMean['Latitude'].mean()
    filtered = df[df.apply(lambda x: haversine(meanLon, meanLat, x['Longitude'], x['Latitude']) < radius and x['Primary Type'] == classe, axis=1)]
    dfReturn = pd.concat([dfReturn, filtered], axis=0)
 return dfReturn


dataset = pd.read_csv("/Users/Jean/Documents/Software Engineering/UFG/mestrado/dataset/CrimesChicago.csv",sep=",")

dataset = dataset[dataset.Year == 2016]

classes = ["THEFT",
            "BATTERY"]

dataset = dataset[dataset['Primary Type'].isin(classes)]

#convert dates to pandas datetime format
dataset.Date = pd.to_datetime(dataset.Date, format='%m/%d/%Y %I:%M:%S %p')

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

#Erase period column because of the dummy trap
dataset = dataset.drop("Period_3",axis=1)

dataset['Primary Type'] = pd.Categorical(dataset['Primary Type'])
dataset['Location Description'] = pd.Categorical(dataset['Location Description'])
dataset['Domestic'] = pd.Categorical(dataset["Domestic"])

dataset['Primary Type']  = dataset['Primary Type'].cat.codes
dataset['Location Description']  = dataset['Location Description'].cat.codes
dataset['Domestic'] = dataset['Domestic'].cat.codes

#Rearranging columns
cols = ['Primary Type'] + [col for col in dataset if col != 'Primary Type']
dataset = dataset[cols]

print(dataset.columns.tolist())

print("Dataset size",len(dataset))
#Remove empty rows

dataset.dropna(inplace=True)

dataset.to_csv('crimes2016THEFTandBATTERY.csv', index=False, encoding='utf-8')
print("CSV FILE GENERATED")

