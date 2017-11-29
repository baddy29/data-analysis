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
#LOGISTIC REGRESSION

#Preparing data for logistic regression
y , X = dmatrices('QuoteFlag ~ Postcode_Region + Priority_Segment + Lead_Age_N + Combined_Gender + DaysLefttoRenewal + Age',dataset, return_type ="dataframe")


#flatten y into 1-d array
y= np.ravel(y)


#running logistic regression
model = LogisticRegression()
model =model.fit(X,y)

#check accuracy
model.score(X,y) #94.2% accurrate

#percentage of Quotes
y.mean() #57% "Quoted" which means only 43% accuracy when "Not Quoted"

#Examining the coefficients
pd.DataFrame(list(zip(X.columns, np.transpose(model.coef_)))) 
------------------------------------------------------------------------------------------------------

#training and testing

X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.3,random_state=0)

model2 = LogisticRegression()
model2.fit(X_train,y_train)


# predict class labels for the test set
predicted = model2.predict(X_test)
print (predicted)

#Geneerate class probabilites
probs = model2.predict_proba(X_test)
print(probs)#classifier is giving Quote as 1 or TRUE whenever probability in second column is greater than 0.6

#Evaluation metrics
print (metrics.accuracy_score(y_test,predicted))
print (metrics.roc_auc_score(y_test,probs[:,1]))

#confusion matrix
print (metrics.confusion_matrix(y_test,predicted))
print (metrics.classification_report(y_test,predicted))

#cross_validation
scores = cross_val_score(LogisticRegression(), X, y, scoring='accuracy', cv=10)
print (scores)
print (scores.mean())
-------------------------------------------------------------------------------------------------------

#Generating ouput from the model#

model.predict_proba(np.array([1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 1, 1, 20,
                              50]))

#Predicting randomly A "Man" of "50 years of age" from "South West" having priority "2" with lead age "1" and days left 
#for renewal is 20 #His probabilty of giving a quote is 6%

model.predict_proba(np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 1, 1, 1, 21,
                              65]))

#Predicting randomly A "Man" of "65 years of age" from "Wales" having priority "1" with lead age "1" and days left 
#for renewal are "21 days" #His probabilty of giving a quote is 12%
--------------------------------------------------------------------------------------------------------