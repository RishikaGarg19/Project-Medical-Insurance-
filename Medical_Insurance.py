import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score

dataset = pd.read_csv("./Medical Insurance.csv")
dataset = pd.get_dummies(dataset, columns=["Sex", "Region", "Smoker?"])

X = dataset.iloc[:, 1:12].values
Y = dataset.iloc[:, 0:1].values

from sklearn.preprocessing import StandardScaler
stdscale = StandardScaler()
X[:, 0:2] = stdscale.fit_transform(X[:, 0:2])

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

print()
print(dataset)
print()

#-----------------------------------------------LINEAR REGRESSION------------------------------------------

from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X_train, Y_train)

Y_pred = linreg.predict(X_test)

from sklearn.metrics import r2_score
print()
print("Accuracy with Linear Regression:")
print(r2_score(Y_test, Y_pred))
print(cross_val_score(linreg, X, Y.ravel(), cv=5))
print()
print()


#--------------------------------------- POLYNOMIAL REGRESSION --------------------------------------------

from sklearn.preprocessing import PolynomialFeatures
polreg = PolynomialFeatures(degree=2)
#Got the highest with 2, got 87.513% with 3
Xpoly_train = polreg.fit_transform(X_train)
PolyReg = LinearRegression()
PolyReg.fit(Xpoly_train, Y_train)
Xpoly_test = polreg.fit_transform(X_test)
#Had to transform both X_train and X_test into polynomial versions separately to get the result

Ypoly_pred = PolyReg.predict(Xpoly_test)

print("Accuracy with Polynomial Regression:")
print(r2_score(Y_test, Ypoly_pred))
print(cross_val_score(PolyReg, X, Y.ravel(), cv=5))
print()
print()


#--------------------------------------DECISION TREE REGRESSOR--------------------------------------------

from sklearn.tree import DecisionTreeRegressor
dtreg = DecisionTreeRegressor(max_leaf_nodes=10)
#Got 88.197% with 9 and 87.494% with 11, best at 10
dtreg.fit(X_train, Y_train)

Ydt_pred = dtreg.predict(X_test)

print("Accuracy with Decision Tree Regressor:")
print(r2_score(Y_test, Ydt_pred))
print(cross_val_score(dtreg, X, Y.ravel(), cv=5))
print()
print()


#--------------------------------------SUPPORT VECTOR MACHINE----------------------------------------------

from sklearn.svm import SVR
svreg = SVR(kernel="poly")
Y_train_scaled = StandardScaler().fit_transform(Y_train)
#It needed all the Y data to be scaled to match with the scaled X values
svreg.fit(X_train, Y_train_scaled.ravel())
Y_scaled = StandardScaler().fit_transform(Y)

Ysvm_pred = svreg.predict(X_test)

print("Accuracy with Support Vector Machine:")
Y_test_scaled = StandardScaler().fit_transform(Y_test)
print(r2_score(Y_test_scaled, Ysvm_pred))
print(cross_val_score(svreg, X, Y_scaled.ravel(), cv=5))
print()
print()


#---------------------------------------RANDOM FOREST REGRESSION-------------------------------------------

from sklearn.ensemble import RandomForestRegressor
rfreg = RandomForestRegressor(n_estimators=90, max_leaf_nodes=10)
#Worked best with 10 nodes and 90 n-estimators, maintained an accuracy close to 89%
rfreg.fit(X_train, Y_train.ravel())

Yrf_pred = rfreg.predict(X_test)

print("Accuracy with Random Forest Regressor:")
print(r2_score(Y_test, Yrf_pred))
print(cross_val_score(rfreg, X, Y.ravel(), cv=5))
print()
print()
