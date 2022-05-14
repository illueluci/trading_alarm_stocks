# Testing all classification

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('test_folder/BBCA.JK_mod.csv')
x = dataset.iloc[:, 5:-3].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2,) # has random state
# print(x_train)
# print(x_test)
# print(y_train)
# print(y_test)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train_scaled = sc.fit_transform(x_train)
x_test_scaled = sc.transform(x_test)
# print(x_train_scaled)
# print(x_test_scaled)


# Training the Logistic Regression model on the Training set
from sklearn.linear_model import LogisticRegression
classifier0 = LogisticRegression()  # has random state
classifier0.fit(x_train_scaled, y_train)

# KNN
from sklearn.neighbors import KNeighborsClassifier
classifier1 = KNeighborsClassifier(n_neighbors=5, metric="minkowski",p=2)
classifier1.fit(x_train_scaled, y_train)

# SVM
from sklearn.svm import SVC
classifier2 = SVC(kernel="linear") # has random state
classifier2.fit(x_train_scaled, y_train)

# Kernel SVM
classifier3 = SVC(kernel="rbf") # has random state
classifier3.fit(x_train_scaled, y_train)

# Naive Bayes
from sklearn.naive_bayes import GaussianNB
classifier4 = GaussianNB()
classifier4.fit(x_train_scaled, y_train)

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
classifier5 = DecisionTreeClassifier(criterion="entropy")  # has random state
classifier5.fit(x_train_scaled, y_train)

# Random Forest
from sklearn.ensemble import RandomForestClassifier
classifier6 = RandomForestClassifier(n_estimators=70, criterion="entropy")  # has random state
classifier6.fit(x_train_scaled, y_train)

# XGBoost
from xgboost import XGBClassifier
classifier7 = XGBClassifier()
classifier7.fit(x_train_scaled, y_train)

# Catboost
from catboost import CatBoostClassifier
classifier8 = CatBoostClassifier()
classifier8.fit(x_train_scaled, y_train)

# tidying up
classifier_list = [classifier0, classifier1, classifier2,
                   classifier3, classifier4, classifier5,
                   classifier6, classifier7, classifier8]

# Predicting the Test set results
y_pred_2d_list = []
for classifier in classifier_list:
    temp = classifier.predict(x_test_scaled)
    temp_2d = temp.reshape(len(temp), 1)
    y_pred_2d_list.append(temp)

y_test_2d = y_test.reshape(len(y_test), 1)

# for _ in y_pred_2d_list:
#     print(_)

# Making the Confusion Matrix & accuracy score
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
for i, single_y in enumerate(y_pred_2d_list):
    print(f"Classifier type: {classifier_list[i]}")
    cf_m = confusion_matrix(y_test_2d, single_y)
    print(f"confusion matrix of classifier{i}: ")
    print(cf_m)
    acc_score = accuracy_score(y_test_2d, single_y)
    print(f"accuracy score of classifier{i}: {acc_score:.4f}")
    prc_score = precision_score(y_test_2d, single_y, zero_division=0)
    print(f"precision score of classifier{i}: {prc_score:.4f}")
    print("-" * 50)
