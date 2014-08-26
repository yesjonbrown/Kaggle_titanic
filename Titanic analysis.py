
# coding: utf-8

# In[39]:

import pandas as pd
import numpy as np
import matplotlib as plt

#read in our Titanic data set using pandas
df = pd.read_csv("Desktop/My Kaggle Folder/Titanic/train.csv") 

#df.head(15)    #prints out first 10 rows

#df.describe()  #prints out stats from numerical fields

#df["Age"].median()   #computes the median age from the data set

#df["Sex"].unique()   #looks at unique values to see whether they make sense or not

#looking at distribution of age 
fig = plt.pyplot.figure()
ax = fig.add_subplot(111)
ax.hist(df["Age"], bins = 10, range = (df["Age"].min(), df["Age"].max()))

plt.pyplot.title("Age distribution")
plt.pyplot.xlabel("Age")
plt.pyplot.ylabel("Number of Passengers")
plt.pyplot.show()

#distribution of fare
fig = plt.pyplot.figure()
ax = fig.add_subplot(111)
ax.hist(df["Fare"], bins = 10, range = (df["Fare"].min(), df["Fare"].max()))

plt.pyplot.title("Fare distribution")
plt.pyplot.xlabel("Fare")
plt.pyplot.ylabel("Number of Passengers")
plt.pyplot.show()

df.boxplot(column="Fare")    #boxplot of passengers by fare paid

df.boxplot(column="Fare", by = "Pclass")    #dividing the fares into classes

#plotting survival via what class passengers were in
t1 = df.groupby("Pclass").Survived.count()    
t2 = df.groupby("Pclass").Survived.sum()/df.groupby("Pclass").Survived.count()    #dividing the survivors by total count to get probability of survival

fig = plt.pyplot.figure(figsize=(8,4))
ax1 = fig.add_subplot(121)
ax1.set_xlabel("Passenger class")
ax1.set_ylabel("Number of Passengers")
ax1.set_title("Passengers divided by their class")
t1.plot(kind = "bar")    #gives us a bar chart for our temp1 data plot

ax2 = fig.add_subplot(122)
t2.plot(kind = "bar")    #gives us a bar chart for our temp2 data plot
ax2.set_xlabel("Pclass")
ax2.set_ylabel("Probability of Survival")
ax2.set_title("Prob of survival by class")







