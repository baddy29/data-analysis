#Importing libraries#
import numpy as np
import pandas as pd

#Loading csv file#
df=pd.read_csv('C:\\Users\\AQdmin\\Desktop\\Extra_docs\\PetInsurance_Model_Data1.csv')
print(df)

#Removing fields where DMCFlaf=0#
df=df.drop(df[df.DMCFlag == 0].index)

#Removing fileds where Days left for renewal < 0#
df=df.drop(df[df.DaysLefttoRenewal<0].index)

#How many attributes have missing values#
df.isnull().sum()

#Filled missing values using WEKA TOOLS and loading new file#
df=pd.read_csv('C:\\Users\\AQdmin\\Desktop\\PetInsurance_Model_Data1.2.1.csv')

#Now checking wether their are anymore missing values or not#
df.isnull().sum()
print(df)
-----------------------------------------------------------------------------------
