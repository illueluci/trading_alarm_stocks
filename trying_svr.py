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

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
scaled_x_train = sc_x.fit_transform(x_train)
scaled_x_test = sc_x.transform(x_test)
y_train_2d = y_train.reshape(len(y_train), 1)
y_test_2d = y_test.reshape(len(y_test), 1)
scaled_y_train_2d = sc_y.fit_transform(y_train_2d)
scaled_y_test_2d = sc_y.transform(y_test_2d)
scaled_y_train_1d = scaled_y_train_2d.reshape(len(scaled_y_train_2d))
scaled_y_test_1d = scaled_y_test_2d.reshape(len(scaled_y_test_2d))

# Training the SVR model on the Training set
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(scaled_x_train, scaled_y_train_1d)

# predicting results
scaled_y_pred_1d = regressor.predict(scaled_x_test)
scaled_y_pred_2d = scaled_y_pred_1d.reshape(len(scaled_y_pred_1d), 1)
y_pred_2d = sc_y.inverse_transform(scaled_y_pred_2d)
y_pred_1d = y_pred_2d.reshape(len(y_pred_2d))
np.set_printoptions(precision=2, threshold=sys.maxsize, suppress=True)
with open("svr_result.txt", "w") as f:
    print(np.concatenate((y_pred_2d, y_test_2d),1), file=f)

# Evaluating regressions
from sklearn.metrics import r2_score
score = r2_score(y_test, y_pred_1d)
print(f"R2 score: {score}")
adj_score = 1 - (1-score)*(len(x_train)-1)/(len(x_train)-1-8)
print(f"adjusted R2 score: {adj_score}")
# print(len(x_train))
# print(len(y_train))
print("-" * 50)

