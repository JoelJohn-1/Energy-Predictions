#!/usr/bin/env python
# coding: utf-8

# In[1]:


from asyncio import new_event_loop
import math
from bitarray import test
from numpy import tri
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from pyparsing import col
from pytz import HOUR
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from datetime import date, datetime
from sympy import re
pd.options.mode.chained_assignment = None  # default='warn'

#1.) Examine the data, parse the time fields wherever necessary. Take the sum of the energy usage (Use [kW]) to get per day usage and merge it with weather data 
energy_data = pd.read_csv("energy_data.csv")
weather_data = pd.read_csv("weather_data.csv")
unique_energy_days = energy_data["Date & Time"]
#Get all unique dates
weather_days = weather_data["time"]
unique_weather_days = weather_data["time"]
#Remove the H:M:S part from every Date & Time
for x in range (0, len(unique_energy_days)):
    energy_data["Date & Time"][x] = pd.to_datetime(unique_energy_days[x]).date()
for x in range (0, len(weather_days)):
    weather_data["time"][x] = datetime.fromtimestamp(weather_days[x]).date()
    unique_weather_days[x] = (weather_days[x])
unique_weather_days = unique_weather_days.unique() # Get all unique days from weather data
#Set the icon, and summary for what appeared most that day for all rows with the same day,
#couldn't get mode to work with the merge function, so this is a workaorund
x = 0
for days in range (0, len(unique_weather_days)):
    subData = weather_data[weather_data["time"] == unique_weather_days[days]]
    for all in subData:
        weather_data["icon"][x] = (subData["icon"].mode()).values[0]
        weather_data["summary"][x] = (subData["summary"].mode()).values[0]
        x = x + 1
#Define how all rows with the same Date & Time should be merged
sum_energy_rows = {'Date & Time': 'first', 'use [kW]': 'sum', 'gen [kW]': 'sum', 'Grid [kW]': 'sum', 'AC [kW]': 'sum',
 'Furnace [kW]': 'sum', 'Cellar Lights [kW]': 'sum', 'Washer [kW]': 'sum', 'First Floor lights [kW]': 'sum',
  'Utility Rm + Basement Bath [kW]': 'sum', 'Garage outlets [kW]': 'sum', 'MBed + KBed outlets [kW]': 'sum', 
  'Dryer + egauge [kW]': 'sum', 'Panel GFI (central vac) [kW]': 'sum', 'Home Office (R) [kW]': 'sum',
   'Dining room (R) [kW]': 'sum', 'Microwave (R) [kW]': 'sum', 'Fridge (R) [kW]': 'sum'}
sum_weather_rows = {'temperature': 'mean', 'icon': 'first', 'humidity': 'mean', 'visibility': 'mean',
 'summary': 'first', 'pressure': 'mean', 'windSpeed': 'mean', 'cloudCover': 'mean', 'time': 'first',
  'windBearing': 'mean','precipIntensity': 'mean', 'dewPoint': 'mean', 'precipProbability': 'mean'}
#Merge all rows
new_energy_data = energy_data.groupby(energy_data["Date & Time"]).aggregate(sum_energy_rows)
new_weather_data = weather_data.groupby(weather_data["time"]).aggregate(sum_weather_rows)
del new_energy_data["Date & Time"]
del new_weather_data["time"]
#New_energy_data contains all rows with the same days merged into one, with averaged out values
new_energy_data.to_csv("merged_energy_data.csv")
new_weather_data = new_weather_data.iloc[1: , :] #Want to get rid of the 2013 date as there is no match
new_weather_data["use [kW]"] = new_energy_data["use [kW]"] #Add energy usage column to weather
#New_weather contains all rows with the same days merged into one, with averaged out values,
#for icon/summary the icon/summary that appeared the most was used
new_weather_data = new_weather_data[['use [kW]', 'temperature', 'icon', 'humidity', 'visibility', 'summary', 'pressure',
       'windSpeed', 'cloudCover', 'windBearing', 'precipIntensity', 'dewPoint',
       'precipProbability']]
new_weather_data.to_csv("merged_weather_data.csv")


# In[8]:


# 2.)Split the data obtained from step 1, into training and testing sets. The aim is to predict the usage 
# for each day in the month of December using the weather data, so split accordingly. The usage as per 
# devices should be dropped, only the “use [kW]” column is to be used for prediction from the dataset
# Split data based on if its in Decemeber or not
weather_data = pd.read_csv("merged_weather_data.csv")
training_data = pd.DataFrame(columns  = weather_data.columns)
testing_data = pd.DataFrame(columns  = weather_data.columns)
for index, row in (weather_data.iterrows()):
    if pd.to_datetime(row["time"]).date().month < 12:
        training_data.loc[len(training_data.index)] = row
    else:
        testing_data.loc[len(testing_data.index)] = row

#Create two csv files for training and testing date
training_data.to_csv("training_data.csv")
testing_data.to_csv("testing_data.csv")


# In[9]:


# 3.) Linear Regression - Predicting Energy Usage:
# Set up a simple linear regression model to train, and then predict energy usage for each day in the month 
# of December using features from weather data (Note that you need to drop the “use [kW]” column in the 
# test set first). How well/badly does the model work? (Evaluate the correctness of your predictions based 
# on the original “use [kW]” column). Calculate the Root mean squared error of your model.
# Finally generate a csv dump of the predicted values. Format of csv: Two columns, first should 
# be the date and second should be the predicted value. 

#Load data set
training_data = pd.read_csv('training_data.csv') 
testing_data = pd.read_csv('testing_data.csv')
predicted_data_set = pd.DataFrame()
predicted_data_set["time"] = testing_data["time"]

#Convert datetime to epoch so it can be scaled
training_data["time"] = pd.to_datetime(training_data["time"])
testing_data["time"] = pd.to_datetime(testing_data["time"])
training_data["time"] = (training_data["time"] - datetime(1970,1,1)).dt.total_seconds()
testing_data["time"] = (testing_data["time"] - datetime(1970, 1, 1)).dt.total_seconds() 
training_data["time"] = training_data["time"] / (math.pow(10,7))
testing_data["time"] = testing_data["time"] / (math.pow(10, 7))

#Train the Linear Regression based off of the training data
epoch_axis = training_data.iloc[:, 1].values.reshape(-1, 1) 
energy_axis = training_data.iloc[:, 2].values.reshape(-1, 1) 
LR = LinearRegression() 
LR.fit(epoch_axis, energy_axis) 

#Predict the dates of decemeber using the Linear Regression Model
testing_axis = testing_data.iloc[:, 1].values.reshape(-1, 1)
energy_predict = LR.predict(testing_axis)
predicted_data_set["Predicted"] = energy_predict
predicted_data_set.to_csv("cse351_hw2_john_joel_113377079_linear_regression.csv")

#Calculate Root Mean Squared Error
root_squared_mean_error = 0
for x in range (0, len(energy_predict)):
    root_squared_mean_error += pow((predicted_data_set["Predicted"][x] - testing_data["use [kW]"][x]),2)
root_squared_mean_error = root_squared_mean_error / len(energy_predict)
root_squared_mean_error = math.sqrt(root_squared_mean_error)
print("The root mean squared error of the linear regression is: ", root_squared_mean_error)


# In[10]:


# 4. Logistic Regression - Temperature classification:
# Using only weather data we want to classify if the temperature is high or low. 
# Let's assume temperature greater than or equal to 35 is ‘high’ and below 35 is ‘low’.
#  Set up a logistic regression model to classify the temperature for each day in the month of December. 
# Calculate the F1 score for the model.
# Finally generate a csv dump of the classification (1 for high, 0 for low)
# Format: Two columns, first should be the date and second should be the classification (1/0).

#Classify low and high temperatues to 0 and 1
training_data = pd.read_csv('training_data.csv') 
testing_data = pd.read_csv('testing_data.csv')
training_data.loc[training_data["temperature"] >= 35, "temperature"] = 1
training_data.loc[training_data["temperature"] != 1, "temperature"] = 0

#Take in weather data that has integer/float values and not string values
temperature_axis = training_data["temperature"]
training_axis = training_data[['humidity',
       'visibility', 'pressure', 'windSpeed', 'cloudCover',
       'windBearing', 'precipIntensity', 'dewPoint', 'precipProbability']]
testing_data = testing_data[['humidity',
       'visibility', 'pressure', 'windSpeed', 'cloudCover',
       'windBearing', 'precipIntensity', 'dewPoint', 'precipProbability']]

#Create Logistical Regression Model and Predict
LoR= LogisticRegression()
LoR.fit(training_axis,temperature_axis)
temperature_predict = LoR.predict(testing_data)
predicted_values = pd.DataFrame()
testing_data = pd.read_csv('testing_data.csv')
predicted_values["date"] = testing_data["time"]
predicted_values["classification"] = temperature_predict
predicted_values.to_csv("cse351_hw2_john_joel_113377079_logistic_regression.csv")

#Calculate F1 Score
testing_data.loc[testing_data["temperature"] >= 35, "temperature"] = 1
testing_data.loc[testing_data["temperature"] != 1, "temperature"] = 0
real_value = testing_data["temperature"]
predicted_values = predicted_values["classification"]
TP = 0
FP = 0
TN = 0
FN = 0
for x in range (0, len(real_value)):
    if (predicted_values[x] == 1 and real_value[x] == 1):
        TP = TP + 1
    elif (predicted_values[x] == 1 and real_value[x] == 0):
        FP = FP + 1
    elif (predicted_values[x] == 0 and real_value[x] == 0):
        TN = TN + 1
    elif (predicted_values[x] == 1 and real_value[x] == 0):
        FN = FN + 1
precision = TP / (TP + FP)
recall = TP / (TP + FN)
F_score = (2 * precision * recall) / (precision + recall)

print("The F1 Score of the Logisitical Regression Model is:", F_score)


# In[11]:


# 5. Energy usage data Analysis:
# We want to analyze how different devices are being used in different times of the day.
# - Is the washer being used only during the day?
# - During what time of the day is AC used most?
# There are a number of questions that can be asked.
# For simplicity, let’s divide a day in two parts:
# - Day: 6AM - 7PM
# - Night: 7PM - 6AM
# Analyze the usage of any two devices of your choice during the ‘day’ and ‘night’.
#  Plot these trends. Explain your findings.

#Separate the values into Day/Night where: Day = 0 and Night = 1
energy_data = pd.read_csv("energy_data.csv")
date_time = pd.DataFrame()
date_time["Date & Time"] = energy_data['Date & Time'].apply(lambda x: pd.to_datetime(x))
date_time = date_time['Date & Time'].apply(lambda x: 0 if (x.hour >= 6 and x.hour < 19) else 1)
cellar_lights = energy_data["Cellar Lights [kW]"]
furnace = energy_data["Furnace [kW]"]

#Creating a counter for both furnace and energy usage, with a counter for both day and night
usage_data = pd.DataFrame()
usage_data["Furnace [kW]"] = furnace
usage_data["Cellar Lights [kW]"] = cellar_lights
usage_data["Day/Night"] = date_time
cellar_day_usage = 0
cellar_night_usage = 0
furnace_day_usage = 0
furnace_night_usage = 0
for x in range (0, len(usage_data["Day/Night"])):
    if (usage_data["Day/Night"][x] == 0):
        cellar_day_usage = cellar_day_usage + usage_data["Cellar Lights [kW]"][x]
        furnace_day_usage = furnace_day_usage + usage_data["Furnace [kW]"][x]
    else:
        cellar_night_usage = cellar_night_usage + usage_data["Cellar Lights [kW]"][x]
        furnace_night_usage = furnace_night_usage + usage_data["Furnace [kW]"][x]

#Create Box Plot
plt.bar(["Day Usage", "Night Usage"], [cellar_day_usage, cellar_night_usage], color=['red', 'blue'])
plt.xlabel("Time of Day")
plt.ylabel("Total Usage")
plt.title("Usage of the Cellar Lights from Day to Night")
print("Looking at the graph it is obvious a majority of families turn their light off at night and is more likely to use it when they are awake")
plt.show()

#Create pie Chart
pie_chart = pd.DataFrame({"Furnace Usage": [furnace_day_usage, furnace_night_usage]},
                   index=["Day", "night"])
plot = pie_chart.plot.pie(y= "Furnace Usage", figsize=(5, 5))
legend = plot.get_legend()
plt.title("Percentage of Total Furnace Usage: Day and Night")
print("As seen from looking on the graph, furnace usage between day and night is relatively equivalent which could mean that it is on 24/7 since it focuses on keeping the house a certain temperature")
plt.show()

