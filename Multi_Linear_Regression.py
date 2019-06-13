# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# ___categorical feture
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
encoder_x= LabelEncoder()
X[:,3]= encoder_x.fit_transform(X[:,3])
onehotencoder_x= OneHotEncoder(categorical_features= [3])
X= onehotencoder_x.fit_transform(X) .toarray()
#avoiding dummy variable
X= X[:,1:]

#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#fitting dataset into multiple linear regression model.

from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(X_train, y_train)

y_pred= regressor.predict(X_test)




#adding the constant as per the multiple linear regression equation.

import statsmodels.formula.api as sm 
X= np.append(arr = np.ones((50,1)).astype(int), values= X, axis=1)


x_opt = X[:,[0,1,2,3,4,5]]
regressor_ols= sm.OLS(endog= y, exog= x_opt).fit()
print(regressor_ols.summary())
# removing variable whose p value is the highest and greater than significance value.
x_opt = X[:,[0,1,3,4,5]]
regressor_ols= sm.OLS(endog= y, exog= x_opt).fit()
print(regressor_ols.summary())

x_opt = X[:,[0,3,4,5]]
regressor_ols= sm.OLS(endog= y, exog= x_opt).fit()
print(regressor_ols.summary())

x_opt = X[:,[0,3,5]]
regressor_ols= sm.OLS(endog= y, exog= x_opt).fit()
print(regressor_ols.summary())