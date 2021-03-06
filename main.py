from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import yfinance as yf
import time
import datetime
import smtplib
import pickle
import csv_processing_functions_ver

stock_list = ["ADHI.JK", "ADRO.JK", "AKRA.JK", "ANTM.JK", "ASII.JK",
              "ASRI.JK", "BBCA.JK", "BBNI.JK", "BBRI.JK", "BBTN.JK",
              "BKSL.JK", "BMRI.JK", "BSDE.JK", "CPIN.JK", "ELSA.JK",
              "EXCL.JK", "ICBP.JK", "INCO.JK", "INDF.JK", "INDY.JK",
              "INKP.JK", "ITMG.JK", "JSMR.JK", "KLBF.JK", "LPKR.JK",
              "LPPF.JK", "MEDC.JK", "MNCN.JK", "PGAS.JK", "PTBA.JK",
              "PTPP.JK", "SCMA.JK", "SMGR.JK", "SSMS.JK", "TLKM.JK",
              "TPIA.JK", "UNTR.JK", "UNVR.JK", "WIKA.JK", "WSBP.JK",
              "WSKT.JK", ]


def get_data_from_yf():
    for stock in stock_list:
        data = yf.Ticker(stock).history(period="1y")
        data.to_csv(path_or_buf="six_m_csv/" + stock + "_6m.csv")


def process_csv():
    for stock in stock_list:
        csv_processing_functions_ver.make_csv_mod_and_last_row("six_m_mod_csv/" + stock + "_6m_mod.csv",
                                                               "six_m_last_row_csv/" + stock + "_6m_last_row.csv",
                                                               "six_m_csv/" + stock + "_6m.csv",
                                                               method_of_download="auto")


def unpickle():
    with open("pickled_objects/regressor_list_pickle.pickle", "rb") as regressor_list_file:
        regressor_list = pickle.load(regressor_list_file)
    with open("pickled_objects/score_list_pickle.pickle", "rb") as score_list_file:
        score_list = pickle.load(score_list_file)
    return regressor_list, score_list


def predict_with_regressors(regressor_list, score_list):
    prediction_list = []
    for i, stock in enumerate(stock_list):
        dataset = pd.read_csv('six_m_last_row_csv/' + stock + '_6m_last_row.csv')
        x = dataset.iloc[:, 5:-2].values
        y_pred = regressor_list[i].predict(x)
        if y_pred >= 1:
            value_pair = y_pred[0], stock, score_list[i]
            prediction_list.append(value_pair)

    prediction_list.sort(reverse=True)
    prediction_string = ""
    for a, b, c in prediction_list:
        prediction_string += f"{a:.2f}% increase for {b} (score: {c:.4f})\n"
    return prediction_list, prediction_string


def send_email(prediction_string, last_sent_time):
    password = "xxxxxxxxxxxxxxxxxxx"
    sender_email = "xxxxxxxxxxxxxxxxxxx"
    receiver_email = "xxxxxxxxxxxxxxxxxxxxxx"
    message = \
    f"""Subject: Stock picks {datetime.date.today()}\n\n
    {prediction_string}\n\n
    This message is sent from Python.
    """

    try:
        server = smtplib.SMTP_SSL("smtp.mail.yahoo.com", 465)
        server.ehlo()
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message)
        server.quit()
        print("%" * 50)
        print("Email has been sent")
        print(message)
        print("%" * 50)
        last_sent_time = datetime.datetime.utcnow()
    except:
        print("%" * 50)
        print("Can't send email")
        print("%" * 50)

    if last_sent_time:
        return last_sent_time


def main():
    last_sent_time = datetime.datetime.min
    while True:
        condition1 = 11 <= datetime.datetime.utcnow().hour <= 23 or datetime.datetime.utcnow().hour < 2
        # 18.00 - 06.59 or 07.00 - 08.59 WIB
        print(f"the hour now in utc: {datetime.datetime.utcnow().hour}")
        print(f"condition 1: {condition1}")
        condition2 = (datetime.datetime.utcnow() - last_sent_time).seconds > 86300 or \
                     (datetime.datetime.utcnow() - last_sent_time).days >= 1
        print(f"It has been {(datetime.datetime.utcnow() - last_sent_time).seconds} seconds and "
              f"{(datetime.datetime.utcnow() - last_sent_time).days} days since the program sent an email.")
        print(f"condition 2: {condition2}")
        # have not sent an email in 24 hours minus 100 seconds or 1 day or more

        if condition1 and condition2:
            print()
            print("getting data from yahoo finance...")
            print()
            get_data_from_yf()
            print()
            print("processing csv...")
            print()
            process_csv()
            print()
            print("unpickling regressors...")
            print()
            reg, score = unpickle()
            print()
            print("making predictions...")
            print()
            pred_list, pred_str = predict_with_regressors(reg, score)
            print()
            print("sending email...")
            print()
            last_sent_time = send_email(pred_str, last_sent_time)
            time.sleep(60000)
        else:
            time.sleep(300)


if __name__ == "__main__":
    while True:
        try:
            main()
        except Exception as e:
            print("An exception has occurred.")
            print(e)
            time.sleep(60)
