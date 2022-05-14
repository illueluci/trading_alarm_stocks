# Importing the libraries
import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('mod_csv/BBCA.JK_mod.csv')
x = dataset.iloc[:, 5: -2].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 0)

# Training the Polynomial Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.transform(x_test)
regressor = LinearRegression()
regressor.fit(x_train_poly, y_train)

# predicting result
y_pred = regressor.predict(x_test_poly)
np.set_printoptions(precision=2, threshold=sys.maxsize, suppress=True)
y_pred_2d = y_pred.reshape(len(y_pred), 1)
y_test_2d = y_test.reshape(len(y_test), 1)
with open("poly_linear_result.txt", "w") as f:
    print(np.concatenate((y_pred_2d, y_test_2d), 1), file=f)

# Evaluating regressions
from sklearn.metrics import r2_score
score = r2_score(y_test, y_pred)
print(f"R2 score: {score}")
adj_score = 1 - (1-score)*(len(x_train)-1)/(len(x_train)-1-8)
print(f"adjusted R2 score: {adj_score}")
print("-" * 50)