
# coding: utf-8

# In[1]:


#Import pandas
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error


# In[2]:


#Import weather data
weather = pd.read_csv("W:/Ckalib.Nelson/HackCville/Machine Learning Project/ML Competition Materials.csv")


# In[3]:


#View first five results of weather data
weather.head(15)


# In[4]:


#View the data size
weather.size


# In[5]:


#View the shape of the dataset
weather.shape


# In[6]:


#View columns of datset
weather.columns


# In[7]:


#Convert date string to float
weather['Date'] = pd.to_datetime(arg=weather['Date'])
#Separating the days, months, and years into descrete columns as this data may be valuable in determining rain
weather["Month"] = weather["Date"].dt.month
weather["Day"] = weather["Date"].dt.day
weather["Year"] = weather["Date"].dt.year
#Delete the 'Date' column as it causes problems later on in scaling
del weather['Date']


# In[8]:


#Check for null values
print('Let\'s check for null values\n')
print(weather.isnull().sum())


# In[9]:


#Handle missing values. Fill the missing values with “mean” of the respective column.
from sklearn.preprocessing import Imputer

missingValuesImputer = Imputer (missing_values = 'NaN', strategy= 'mean', axis =0) #Initializing object

missingValuesImputer = missingValuesImputer.fit(weather[['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
       'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
       'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am',
       'Temp3pm']])
weather[['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
       'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
       'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am',
       'Temp3pm']] = missingValuesImputer.transform(weather[['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
       'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
       'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am',
       'Temp3pm']])


# In[10]:


#Check for null values after replacing the null values w/ the mean of the column
print('Let\'s check for null values\n')
print(weather.isnull().sum())


# In[11]:


#View data types of all columns in dataset
weather.dtypes


# In[12]:


#Convert objects to strings
from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
weather['WindGustDir'] = enc.fit_transform(weather['WindGustDir'].astype(str))
weather['WindDir9am'] = enc.fit_transform(weather['WindDir9am'].astype(str))
weather['WindDir3pm'] = enc.fit_transform(weather['WindDir3pm'].astype(str))
weather['RainToday'] = enc.fit_transform(weather['RainToday'].astype(str))
weather['RainTomorrow'] = enc.fit_transform(weather['RainTomorrow'].astype(str))


# In[13]:


#Ensure that LabelEncoder works properly
print('Let\'s ensure that all non-numerical columns are now categorized appropriately')
print(weather['WindGustDir'].unique())
print(weather['WindDir9am'].unique())
print(weather['WindDir3pm'].unique())
print(weather['RainToday'].unique())
print(weather['RainTomorrow'].unique())


# In[14]:


#View data types of all columns in dataset after conversion
weather.dtypes


# In[15]:


#Encode categorical data - Convert categorical column in the dataset to numerical data.
weather['WindGustDir'] = enc.fit_transform(weather['WindGustDir'])
weather['WindDir9am'] = enc.fit_transform(weather['WindDir9am'])
weather['WindDir3pm'] = enc.fit_transform(weather['WindDir3pm'])
weather['RainToday'] = enc.fit_transform(weather['RainToday'])
weather['RainTomorrow'] = enc.fit_transform(weather['RainTomorrow'])

enc.classes_ #Maintains the information of the encoded values. Encode a non-numerical data into a numerical data

weather.head(15)
#Notice how all of the non-numerical columns are now numerical


# In[16]:


#Visualize distribution of results
weather.hist(bins=50, figsize=(20,20), color = 'green');


# In[17]:


#Scale data (0 to 1) to normalize the relationships
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
scaler.fit(weather)
weather_scaled = pd.DataFrame(scaler.transform(weather), index=weather.index, columns=weather.columns)
#View scaled data
weather_scaled.head()


# In[18]:


#View columns of dataset
weather_scaled.columns


# In[19]:


#Create a feature df from the columns
features = weather_scaled[['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
       'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm',
       'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
       'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am',
       'Temp3pm', 'Month', 'Day', 'Year']]


# In[20]:


#Create a target
y_target = weather_scaled[['RainTomorrow']]


# In[21]:


#Split test and training data for target object 1
#By default 75% training data and 25% testing data but we will do 80% training data and 20% testing data
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(features,
                                                    y_target,
                                                    test_size=.2,
                                                    random_state=1)


# In[22]:


#View shape of train and test data sets for both feature and response
print (x_train.shape, x_test.shape, y_train.shape, y_test.shape)


# In[23]:


#Train a logistic regression model on the training set
logReg = LogisticRegression()
logReg.fit(x_train,y_train)


# In[24]:


#make predictions for the testing set
y_predict = logReg.predict(x_test)
y_predict


# In[25]:


# Probabilities
y_proba = logReg.predict_proba(x_test)
y_proba


# In[26]:


#check for accuracy
from sklearn import metrics
print (metrics.accuracy_score(y_test,y_predict))


# In[27]:


print(accuracy_score(y_predict, y_test)) # instead of r2_score, categorical variables use the accuracy_score to check accuracy
logReg.score(x_test, y_test)


# In[28]:


y_predict = pd.DataFrame(y_predict)
comparison = pd.concat([y_test.reset_index(), y_predict], axis=1).set_index('index') # to show where the differences are
comparison.head(15)

