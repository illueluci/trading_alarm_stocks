import pandas as pd
import numpy as np


def make_list_obv(insert_close_price: list, insert_volume: list) -> list:  # OBV indicator
    obv_column = [0.0]
    temp = 0
    for i, x in enumerate(insert_close_price[1:]):
        actual_index = ii = i+1
        if insert_close_price[ii] > insert_close_price[i]:
            temp += insert_volume[ii]
        elif insert_close_price[ii] < insert_close_price[i]:
            temp -= insert_volume[ii]
        obv_column.append(temp)
    # print(obv_column)
    obv_delta = [0.0] * 5
    for i in range(5, len(obv_column)):
        temp = (obv_column[i] - obv_column[i-4])/5
        obv_delta.append(temp)
    # print(obv_delta)
    # print(len(obv_delta))
    return obv_delta


def make_list_accudist(insert_close_price: list,    # Accumulation/Distribution Indicator
                       insert_low_price: list,
                       insert_high_price: list,
                       insert_volume: list,
                       ) -> list:
    ad_column = []
    temp_ad = 0
    for i in range(len(insert_close_price)):
        if insert_high_price[i] != insert_low_price[i]:
            mfm = ((insert_close_price[i] - insert_low_price[i]) -
                   (insert_high_price[i] - insert_close_price[i])) / (insert_high_price[i] - insert_low_price[i])
            mfv = mfm * insert_volume[i]
            temp_ad += mfv
        ad_column.append(temp_ad)
    # print(ad_column)
    ad_delta = [0.0] * 5
    for i in range(5, len(ad_column)):
        temp = (ad_column[i] - ad_column[i-4])/5
        ad_delta.append(temp)
    # print(ad_delta)
    # print(len(ad_delta))
    return ad_delta


def make_list_aroon(insert_close_price: list) -> list:  # Aroon Indicator
    aroon_up = [0.0] * 24
    aroon_down = [0.0] * 24
    aroon_oscilator = [0.0] * 24
    for i in range(24, len(insert_close_price)):
        partial_close_list = insert_close_price[i-24: i+1]
        max_finder = 0
        min_finder = 10**12
        for j, value in enumerate(partial_close_list):
            if value > max_finder:
                max_finder = value
                max_finder_index = 24 - j
            if value < min_finder:
                min_finder = value
                min_finder_index = 24 - j
        temp1 = 100 * (25 - max_finder_index) / 25
        temp2 = 100 * (25 - min_finder_index) / 25
        aroon_up.append(temp1)
        aroon_down.append(temp2)
        aroon_oscilator.append(temp1 - temp2)
    return aroon_oscilator


def make_list_macd_signal_diff(insert_close_price: list) -> list:   # MACD (and its difference to signal)
    ma_12 = [0.0] * 11
    ma_26 = [0.0] * 25
    for i in range(11, len(insert_close_price)):
        temp_list = insert_close_price[i-11: i+1]
        temp = sum(temp_list)/12
        ma_12.append(temp)
    for i in range(25, len(insert_close_price)):
        temp_list = insert_close_price[i-25: i+1]
        temp = sum(temp_list)/26
        ma_26.append(temp)
    # print(close_price)
    # print(len(close_price))
    # print(ma_12)
    # print(len(ma_12))
    # print(ma_26)
    # print(len(ma_26))
    ema_12 = [0.0] * 12
    ema_26 = [0.0] * 26
    # first ema uses yesterday's ma as yesterday's ema
    temp = (insert_close_price[12] * 2 / (12+1)) + (ma_12[11] * (1 - (2 / (12+1))))
    ema_12.append(temp)
    for i in range(13, len(ma_12)):
        temp = (insert_close_price[i] * 2 / (12+1)) + (ema_12[-1] * (1 - (2 / (12+1))))
        ema_12.append(temp)
    temp = (insert_close_price[26] * 2 / (26+1)) + (ma_26[25] * (1 - (2 / (26+1))))
    ema_26.append(temp)
    for i in range(27, len(ma_12)):
        temp = (insert_close_price[i] * 2 / (26+1)) + (ema_26[-1] * (1 - (2 / (26+1))))
        ema_26.append(temp)
    # print(ema_12)
    # print(len(ema_12))
    # print(ema_26)
    # print(len(ema_26))
    macd = [0.0] * 26
    for i in range(26, len(ema_26)):
        temp = ema_12[i] - ema_26[i]
        macd.append(temp)
    # making signal line
    macd_ma_9 = [0.0] * 34
    macd_signal = [0.0] * 35  # = macd_ema_9
    for i in range(34, len(macd)):
        temp_list = insert_close_price[i-8: i+1]
        temp = sum(temp_list)/9
        macd_ma_9.append(temp)
    temp = (macd[35] * 2 / (9+1)) + (macd_ma_9[34] * (1 - (2 / (9+1))))
    macd_signal.append(temp)
    for i in range(36, len(ma_12)):
        temp = (macd[i] * 2 / (9+1)) + (macd_signal[-1] * (1 - (2 / (9+1))))
        macd_signal.append(temp)
    # print(macd_ma_9)
    # print(len(macd_ma_9))
    # print(macd_signal)
    # print(len(macd_signal))
    macd_to_signal_diff = [0.0] * 35
    for i in range(35, len(macd)):
        temp = macd[i] - macd_signal[i]
        macd_to_signal_diff.append(temp)
    # print(macd_to_signal_diff)
    # print(len(macd_to_signal_diff))
    return macd_to_signal_diff


def make_list_rsi(insert_close_price: list) -> list:  # Relative Strength Index (RSI)
    percent_change = [0.0]
    for i in range(1, len(insert_close_price)):
        temp = (insert_close_price[i] - insert_close_price[i-1]) / insert_close_price[i-1] * 100
        percent_change.append(temp)
    # print(percent_change)
    # print(len(percent_change))
    avg_gain = [0.0] * 14
    avg_loss = [0.0] * 14
    for i in range(14, len(percent_change)):
        gain = []
        loss = []
        for j in range(i-13, i+1):
            if percent_change[j] >= 0:
                gain.append(percent_change[j])
            else:
                loss.append(-percent_change[j])
        temp_avg_gain = sum(gain)/14
        temp_avg_loss = sum(loss)/14
        avg_gain.append(temp_avg_gain)
        avg_loss.append(temp_avg_loss)
    # print(avg_gain)
    # print(len(avg_gain))
    # print(avg_loss)
    # print(len(avg_loss))
    rsi = [0.0] * 14
    if avg_loss[14] == 0:
        temp = 100
    else:
        temp = 100 - (100/(1+(avg_gain[14]/avg_loss[14])))
    rsi.append(temp)
    for i in range(15, len(avg_gain)):
        if percent_change[i] > 0:
            current_gain = percent_change[i]
            current_loss = 0
        elif percent_change[i] < 0:
            current_gain = 0
            current_loss = -percent_change[i]
        else:
            current_gain = 0
            current_loss = 0

        if avg_loss[i-1] * 13 + current_loss == 0:
            temp = 100
        else:
            temp = 100 - (100 / (1 + ((avg_gain[i-1] * 13 + current_gain)/(avg_loss[i-1] * 13 + current_loss))))
        rsi.append(temp)
    # print(rsi)
    # print(len(rsi))
    # for x in rsi:
    #     if not 0 <= x <= 100:
    #         print(rsi)
    return rsi


def make_list_stochastic(insert_close_price: list) -> list:  # (fast) Stochastic Oscillator
    stochastic = [0.0] * 13
    for i in range(13, len(insert_close_price)):
        temp_list = insert_close_price[i-13: i+1]
        if max(temp_list) - min(temp_list) == 0:
            temp = 100
        else:
            temp = (insert_close_price[i] - min(temp_list))/(max(temp_list) - min(temp_list)) * 100
        stochastic.append(temp)
    # print(stochastic)
    # print(len(stochastic))
    # for x in stochastic:
    #     if not 0 <= x <= 100:
    #         print(x)
    return stochastic


def close_price_3d_and_percent(insert_close_price: list) -> tuple:
    close_price_3days = insert_close_price[3:]
    # print(close_price_3days)
    for i in range(3):
        close_price_3days.append("nan")
    # print(close_price)
    # print(close_price_3days)
    increase_percent = []
    for i, x in enumerate(insert_close_price):
        try:
            temp = (close_price_3days[i] - insert_close_price[i]) / insert_close_price[i] * 100
        except:
            temp = "nan"
        increase_percent.append(temp)
    return close_price_3days, increase_percent


def make_csv_mod_and_last_row(mod_destination, last_row_destination, input_csv_directory, method_of_download):
    """

    :param mod_destination: where to put the modified csv files
    :param last_row_destination: where to put the last row csv files
    :param input_csv_directory: where the original csv came from
    :param method_of_download: "manual" (going to the page and clicking download) or "auto" (using yfinance)
    :return:
    """
    filename_regular_csv = input_csv_directory
    dataset = pd.read_csv(filename_regular_csv)

    # remove adjusted close, dividend, stock split, because I don't need it
    if method_of_download == "manual":
        dataset.drop("Adj Close", inplace=True, axis=1)
    if method_of_download == "auto":
        dataset.drop("Dividends", inplace=True, axis=1)
        dataset.drop("Stock Splits", inplace=True, axis=1)

    # remove rows with null values
    dataset.dropna(inplace=True)

    open_price = list(dataset.iloc[:, 1].values)
    high_price = list(dataset.iloc[:, 2].values)
    low_price = list(dataset.iloc[:, 3].values)
    close_price = list(dataset.iloc[:, 4].values)
    volume = list(dataset.iloc[:, 5].values)

    dataset["OBV"] = make_list_obv(close_price, volume)
    dataset["A/D"] = make_list_accudist(close_price, low_price, high_price, volume)
    dataset["Aroon Oscillator"] = make_list_aroon(close_price)
    dataset["MACD-signal difference"] = make_list_macd_signal_diff(close_price)
    dataset["RSI"] = make_list_rsi(close_price)
    dataset["Stochastic Oscillator"] = make_list_stochastic(close_price)
    dataset["Close Price 3 days later"], dataset["Increase %"] = close_price_3d_and_percent(close_price)

    # remove row0-80 because macd
    # and also row -3 and -2 because nan value for % increase
    # row -1 is kept for single prediction
    dataset.drop(dataset.index[range(80 + 1)], inplace=True, axis=0)
    dataset.drop(dataset.index[range(-3, -1)], inplace=True, axis=0)
    last_row = dataset.iloc[-1,:].values
    last_row = last_row.reshape(1, len(last_row))
    last_row = pd.DataFrame(last_row)

    # dataset.drop("Volume", inplace=True, axis=1)

    print(last_row)
    dataset.drop(dataset.index[-1], inplace=True, axis=0)
    print(dataset)
    dataset.to_csv(path_or_buf=mod_destination)
    last_row.to_csv(path_or_buf=last_row_destination)


stock_list = [
              # "ADHI.JK", "ADRO.JK", "AKRA.JK", "ANTM.JK", "ASII.JK",
              # "ASRI.JK",
              "BBCA.JK",
              # "BBNI.JK", "BBRI.JK", "BBTN.JK",
              # "BKSL.JK", "BMRI.JK", "BSDE.JK", "CPIN.JK", "ELSA.JK",
              # "EXCL.JK", "ICBP.JK", "INCO.JK", "INDF.JK", "INDY.JK",
              # "INKP.JK", "ITMG.JK", "JSMR.JK", "KLBF.JK", "LPKR.JK",
              # "LPPF.JK", "MEDC.JK", "MNCN.JK", "PGAS.JK", "PTBA.JK",
              # "PTPP.JK", "SCMA.JK", "SMGR.JK", "SSMS.JK", "TLKM.JK",
              # "TPIA.JK", "UNTR.JK", "UNVR.JK", "WIKA.JK", "WSBP.JK",
              # "WSKT.JK",
              ]

if __name__ == "__main__":
    for stock in stock_list:
        make_csv_mod_and_last_row("test_folder/" + stock + "_mod.csv",
                                  "test_folder/" + stock + "_last_row.csv",
                                  "regular_csv/" + stock + ".csv",
                                  method_of_download="manual")
        # input()





