import numpy as np
import pandas as pd
import matplotlib
from ta.trend import EMAIndicator
import math as m
import matplotlib.pyplot as plt

matplotlib.use('Qt5Agg')

"""
    This file implements the scalar EMA strategy.
    Everything is done on log values
"""

DATA_PATH = "D:/nw-trading/Cobalt/Data/"


# --- Parameters Definition ---

FILE_TRAIN = "BTCEUR_5m_2019_10_01_2022_10_01_TRAIN_v1.csv"
FILE_TEST = "BTCEUR_5m_2022_10_02_2022_12_15_TEST_v1.csv"
N_OFFSET_TRAIN = 30000
N_TRAIN = 5000
N_TEST = 50
EWA_WINDOW = 10
SCALAR_WINDOW = 20


# --- Data Loading, Indicators, Normalisation and baselines ---

raw_df_train = pd.read_csv(DATA_PATH + FILE_TRAIN, sep=';')[N_OFFSET_TRAIN:N_OFFSET_TRAIN + N_TRAIN + SCALAR_WINDOW].reset_index()
raw_df_test = pd.read_csv(DATA_PATH + FILE_TEST, sep=';')[:N_TEST + SCALAR_WINDOW].reset_index()

raw_df_train["Log_close"] = raw_df_train["Close"].apply(m.log)
raw_df_test["Log_close"] = raw_df_test["Close"].apply(m.log)

raw_df_train["Log_close_diff"] = raw_df_train["Log_close"].diff(1).shift(-1)
raw_df_test["Log_close_diff"] = raw_df_test["Log_close"].diff(1).shift(-1)

baseline_name = f"BASELINE_LOG_{1}_{EWA_WINDOW}"
residual_name = f"EMA_LOG_{EWA_WINDOW}"

ema_train = EMAIndicator(close=raw_df_train.Log_close, window=EWA_WINDOW, fillna=True)
raw_df_train[residual_name] = ema_train.ema_indicator()
raw_df_train[baseline_name] = raw_df_train["Log_close"] - raw_df_train[residual_name]

ema_test = EMAIndicator(close=raw_df_test.Log_close, window=EWA_WINDOW, fillna=True)
raw_df_test[residual_name] = ema_test.ema_indicator()
raw_df_test[baseline_name] = raw_df_test["Log_close"] - raw_df_test[residual_name]

raw_df_train["Log_b_diff"] = raw_df_train[baseline_name].diff(1).shift(-1)
raw_df_test["Log_b_diff"] = raw_df_test[baseline_name].diff(1).shift(-1)

raw_df_test["current_res_diff"] = raw_df_test["Log_close"].diff(1)


# --- Implementation of the Algorithm ---

predicted_log_diff = [0 for _ in range(SCALAR_WINDOW-1)]
predicted_b_diff = [0 for _ in range(SCALAR_WINDOW-1)]
used_train_index = []
for n in range(N_TEST):
    b_current = np.array(raw_df_test[n:n+SCALAR_WINDOW][baseline_name])
    max_score = -1
    k_max_score = 0
    b_k_max = np.array(raw_df_train[:SCALAR_WINDOW][baseline_name])
    for k in range(N_TRAIN):
        b_k = np.array(raw_df_train[k:k+SCALAR_WINDOW][baseline_name])
        score = np.dot(b_k, b_current) / (np.linalg.norm(b_k)*np.linalg.norm(b_current))
        if score >= max_score:
            max_score = score
            k_max_score = k
            b_k_max = b_k
    predicted_var_b = (np.linalg.norm(b_current) / np.linalg.norm(b_k_max)) * raw_df_train["Log_close_diff"][k_max_score+SCALAR_WINDOW-1]
    predicted_var_R = raw_df_test["current_res_diff"][n+SCALAR_WINDOW-1]
    predicted_log_diff.append(predicted_var_b + predicted_var_R)
    predicted_b_diff.append(predicted_var_b)
    used_train_index.append(k_max_score)
    print(f"n = {n} prediction done, k_max_score = {k_max_score}, max_score = {max_score}")

raw_df_test["Predicted_log_diff"] = pd.Series(predicted_log_diff)
raw_df_test["Predicted_b_diff"] = pd.Series(predicted_b_diff)

plt.plot(pd.Series.cumsum(raw_df_test["Predicted_log_diff"]*raw_df_test["Log_close_diff"]))
plt.figure()
plt.plot(pd.Series.cumsum(raw_df_test["Predicted_b_diff"]*raw_df_test["Log_b_diff"]))
