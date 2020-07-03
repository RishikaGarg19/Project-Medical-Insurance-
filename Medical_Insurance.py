import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

"""
#Flattening X to get 1D arrays for easy plotting
X_trainF = X_train.flatten()
X_testF = X_test.flatten()

#A-ranging Y to get longer 1D arrays of size equal to X
Y_trainR = np.arange(min(Y_train), max(Y_train+1), 58.555/11) 
Y_testR = np.arange(min(Y_test), max(Y_test+1), 192.05/11)
Y_predR = np.arange(min(Y_pred), max(Y_pred+1), 149.49/11)


plt.scatter(X_trainF, Y_trainR, color="red")
plt.scatter(X_testF, Y_testR, color="blue")
plt.scatter(X_testF, Y_predR, color="green")
plt.plot(X_testF, Y_predR, color="green")
plt.show()

While this reshaping worked well, the graph was messy, so I decided to apply PCA and just use the most
impactful column to plot against the output
"""

from sklearn.metrics import r2_score
print()
print("Accuracy with Linear Regression:")
print(r2_score(Y_test, Y_pred))
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

"""
plt.plot(X, Ypoly_pred, color="blue")
plt.show()
"""

print("Accuracy with Polynomial Regression:")
print(r2_score(Y_test, Ypoly_pred))
print()
print()


#--------------------------------------SUPPORT VECTOR MACHINE----------------------------------------------

from sklearn.svm import SVR
svreg = SVR(kernel="poly")
Y_train_scaled = StandardScaler().fit_transform(Y_train)
#It needed all the data (including Y_train and Y_test) to be scaled to give a readable r2 value
svreg.fit(X_train, Y_train_scaled.ravel())

Ysvm_pred = svreg.predict(X_test)

print("Accuracy with Support Vector Machine:")
Y_test_scaled = StandardScaler().fit_transform(Y_test)
print(r2_score(Y_test_scaled, Ysvm_pred))
print()
print()


#--------------------------------------DECISION TREE CLASSIFIER--------------------------------------------

from sklearn.tree import DecisionTreeRegressor
dtreg = DecisionTreeRegressor(max_leaf_nodes=10)
#Got 88.197% with 9 and 87.494% with 11, best at 10
dtreg.fit(X_train, Y_train)

Ydt_pred = dtreg.predict(X_test)

print("Accuracy with Decision Tree Classifier:")
print(r2_score(Y_test, Ydt_pred))
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
print()
print()











