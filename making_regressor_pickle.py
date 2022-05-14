# Importing the libraries
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

stock_list = ["ADHI.JK", "ADRO.JK", "AKRA.JK", "ANTM.JK", "ASII.JK",
              "ASRI.JK", "BBCA.JK", "BBNI.JK", "BBRI.JK", "BBTN.JK",
              "BKSL.JK", "BMRI.JK", "BSDE.JK", "CPIN.JK", "ELSA.JK",
              "EXCL.JK", "ICBP.JK", "INCO.JK", "INDF.JK", "INDY.JK",
              "INKP.JK", "ITMG.JK", "JSMR.JK", "KLBF.JK", "LPKR.JK",
              "LPPF.JK", "MEDC.JK", "MNCN.JK", "PGAS.JK", "PTBA.JK",
              "PTPP.JK", "SCMA.JK", "SMGR.JK", "SSMS.JK", "TLKM.JK",
              "TPIA.JK", "UNTR.JK", "UNVR.JK", "WIKA.JK", "WSBP.JK",
              "WSKT.JK", ]

regressor_list = []
score_list = []

for stock in stock_list:
    # Importing the dataset
    dataset = pd.read_csv('mod_csv/' + stock + '_mod.csv')
    x = dataset.iloc[:, 5: -2].values
    y = dataset.iloc[:, -1].values

    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    # Training the Random Forest Regression model on the Training set
    from sklearn.ensemble import RandomForestRegressor
    regressor = RandomForestRegressor(n_estimators=100, random_state=0)
    regressor.fit(x_train, y_train)

    # predicting result
    y_pred = regressor.predict(x_test)
    # np.set_printoptions(precision=2, threshold=sys.maxsize, suppress=True)
    # y_pred_2d = y_pred.reshape(len(y_pred), 1)
    # y_test_2d = y_test.reshape(len(y_test), 1)
    # with open("poly_linear_result.txt", "w") as f:
    #     print(np.concatenate((y_pred_2d, y_test_2d), 1), file=f)

    # Evaluating regressions
    from sklearn.metrics import r2_score
    print(stock)
    score = r2_score(y_test, y_pred)
    print(f"R2 score: {score}")
    adj_score = 1 - (1-score)*(len(x_train)-1)/(len(x_train)-1-8)
    print(f"adjusted R2 score: {adj_score}")
    print("-" * 50)

    # appending poly and regressor
    regressor_list.append(regressor)
    score_list.append(adj_score)

with open("pickled_objects/regressor_list_pickle.pickle", "wb") as regressor_pickle_file:
    pickle.dump(regressor_list, regressor_pickle_file)
with open("pickled_objects/score_list_pickle.pickle", "wb") as score_pickle_file:
    pickle.dump(score_list, score_pickle_file)