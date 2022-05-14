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

classifier_list = []
score_list = []

for stock in stock_list:
    # Importing the dataset
    dataset = pd.read_csv('mod_classifier/' + stock + '_mod.csv')
    x = dataset.iloc[:, 5: -3].values
    y = dataset.iloc[:, -1].values

    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # Training the Random Forest Regression model on the Training set
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators=100, criterion="entropy")
    classifier.fit(x_train, y_train)

    # predicting result
    y_pred = classifier.predict(x_test)
    # np.set_printoptions(precision=2, threshold=sys.maxsize, suppress=True)
    # y_pred_2d = y_pred.reshape(len(y_pred), 1)
    # y_test_2d = y_test.reshape(len(y_test), 1)
    # with open("poly_linear_result.txt", "w") as f:
    #     print(np.concatenate((y_pred_2d, y_test_2d), 1), file=f)

    # Evaluating regressions
    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
    print(stock)
    cf_m = confusion_matrix(y_test, y_pred)
    print(f"confusion matrix: ")
    print(cf_m)
    acc_score = accuracy_score(y_test, y_pred)
    print(f"accuracy score: {acc_score:.4f}")
    prc_score = precision_score(y_test, y_pred, zero_division=0)
    print(f"precision score of: {prc_score:.4f}")
    print("-" * 50)

    # appending poly and regressor
    classifier_list.append(classifier)
    score_list.append(prc_score)

with open("pickled_objects/classifier_list_pickle.pickle", "wb") as classifier_pickle_file:
    pickle.dump(classifier_list, classifier_pickle_file)
with open("pickled_objects/prc_score_list_pickle.pickle", "wb") as score_pickle_file:
    pickle.dump(score_list, score_pickle_file)

print(f"average precision score: {sum(score_list)/len(score_list):.4f}")