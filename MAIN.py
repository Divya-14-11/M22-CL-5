import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler #standardizre the 
result=pd.read_csv("Churn_Modelling.csv")
print(result.head())
print(result.info())
print(result.dtypes)
print(result.columns)
Lb = LabelEncoder() #converts text into number  
result["Geography"] = Lb.fit_transform(result["Geography"])
result["Gender"] = Lb.fit_transform(result["Gender"])
print(result.head())
print(result.info())
print(result.shape)
result = result.drop(["RowNumber","CustomerId","Surname"],axis = 1)
print(result.shape)
y = result.pop("Excited")
x = result
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size = 0.2, random_state = 0)
s = StandardScaler()
xtrain = s.fit_transform(xtrain)
xtest = s.fit_transform(xtest)
print(xtrain)