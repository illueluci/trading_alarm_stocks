# Importing the libraries
import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('test_folder/BBCA.JK_mod.csv')
x = dataset.iloc[:, 5:-2].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2,
                                                    random_state = 0
                                                    )

# Feature Scaling
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# scaled_x_train = sc.fit_transform(x_train)
# scaled_x_test = sc.transform(x_test)


# Training the Multiple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor0 = LinearRegression()
regressor0.fit(x_train, y_train)

# Training the Decision Tree Regression model on the Training set
from sklearn.tree import DecisionTreeRegressor
regressor1 = DecisionTreeRegressor(random_state=0)
regressor1.fit(x_train, y_train)

# Training the Random Forest Regression model on the Training set
from sklearn.ensemble import RandomForestRegressor
regressor2 = RandomForestRegressor(n_estimators=100,
                                   random_state=0
                                   )
regressor2.fit(x_train, y_train)

# grouping up regressors
regressor_list = [regressor0, regressor1, regressor2,]

# predicting results
y_pred_list = []
for reg in regressor_list:
    temp = reg.predict(x_test)
    y_pred_list.append(temp)

# y_pred = regressor.predict(x_test)
# np.set_printoptions(precision=2)
# print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
#

# Evaluating regressions
from sklearn.metrics import r2_score
for i in range(len(y_pred_list)):
    print(f"regressor: {regressor_list[i]}")
    score = r2_score(y_test, y_pred_list[i])
    print(f"R2 score: {score}")
    adj_score = 1 - (1-score)*(len(x_train)-1)/(len(x_train)-1-8)
    print(f"adjusted R2 score: {adj_score}")
    # print(len(x_train))
    # print(len(y_train))
    print("-" * 50)

# predicting result
result_filename_list = ["linear_reg_result.txt", "decision_tree_result.txt", "random_forest_result.txt"]
for i, regressor in enumerate(regressor_list):
    y_pred = regressor.predict(x_test)
    np.set_printoptions(precision=2, threshold=sys.maxsize, suppress=True)
    y_pred_2d = y_pred.reshape(len(y_pred), 1)
    y_test_2d = y_test.reshape(len(y_test), 1)
    with open(result_filename_list[i], "w") as f:
        print(np.concatenate((y_pred_2d, y_test_2d), 1), file=f)

# single result
# last_row_dataset = pd.read_csv('last_row_csv/BBNI.JK_last_row.csv')
# single_x = last_row_dataset.iloc[:, 5:-2].values
# print(single_x)
# single_y_pred = regressor_list[2].predict(single_x)
# print(single_y_pred)
# val = single_y_pred[0]
# print(val)
# print(f"{val:.2f}")
# val_string = f"{val:.2f}"
# print(val_string)
# print(type(val_string))
