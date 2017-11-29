import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score

#Loading csv file
dataset = pd.read_csv('C:\\Users\\AQdmin\\Desktop\\PetInsurance_Model_Data1.2.1.csv')
-------------------------------------------------------------------------------------------------

#finding out average no. of instances of all variables on basis of QuoteFlag = 0 or QuoteFlag=1
dataset['QuoteFlag']= (dataset.QuoteFlag>0).astype(int)
dataset.groupby('QuoteFlag').mean()

-------------------------------------------------------------------------------------------------
#VISUALISATION of dataset

%matplotlib inline


#Visualising corelation between variables 
df1 = pd.read_csv('C:\\Users\\AQdmin\\Desktop\\PetInsurance_Model_Data1.2.1.csv')
correlations = df1.corr()
 #plot correlation matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,8,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
#ax.set_xticklabels(names)
#ax.set_yticklabels(names)
plt.show()



#On basis of priority
pd.crosstab(dataset.Priority_Segment,dataset.QuoteFlag.astype(bool)).plot(kind='bar')
plt.title('Priority Distribution by Quote value')
plt.xlabel('Priority')
plt.ylabel('Frequency')

#On basis of Region
pd.crosstab(dataset.Postcode_Region,dataset.QuoteFlag.astype(bool)).plot(kind='bar')
plt.title('Region Distribution by Quote value')
plt.xlabel('Region')
plt.ylabel('Frequency')

#On basis of Lead Age
pd.crosstab(dataset.Lead_Age_N,dataset.QuoteFlag.astype(bool)).plot(kind='bar')
plt.title('Lead Age Distribution by Quote value')
plt.xlabel('Lead Age')
plt.ylabel('Frequency')

#On basis of Gender
pd.crosstab(dataset.Combined_Gender,dataset.QuoteFlag.astype(bool)).plot(kind='bar')
lplt.title('Gender Distribution by Quote value')
plt.xlabel('Gender')
plt.ylabel('Frequency')

#On basis of Days left for renewal
pd.crosstab(dataset.DaysLefttoRenewal,dataset.QuoteFlag.astype(bool)).plot(kind='bar')
plt.title('Days for renewal Distribution by Quote value')
plt.xlabel('Days left')
plt.ylabel('Frequency')

--------------------------------------------------------------------------------------------------