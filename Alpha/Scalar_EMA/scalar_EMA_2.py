import numpy as np
import pandas as pd
import matplotlib
import math as m
import matplotlib.pyplot as plt
import datetime as dt

matplotlib.use('Qt5Agg')

"""
    This file implements the scalar EMA strategy.
    Everything is done on log values
    V2:
    - Optimisation with matrix multiplication
    - Mean of multiple max feature
"""

DATA_PATH = "D:/nw-trading/Cobalt/Data/"


# --- Parameters Definition ---

FILE_TRAIN = "BTCEUR_5m_2019_10_01_2022_10_01_TRAIN_v1.csv"
FILE_TEST = "BTCEUR_5m_2022_10_02_2022_12_15_TEST_v1.csv"
N_OFFSET_TRAIN = 180000
N_TRAIN = 20000
N_TEST = 1000
EWA_WINDOW = 10
SCALAR_WINDOW = 20
N_MAX = 10


# --- Data Loading, Indicators, Normalisation and baselines ---

n_tot_train = len(pd.read_csv(DATA_PATH + FILE_TRAIN, sep=';'))
n_tot_test = len(pd.read_csv(DATA_PATH + FILE_TEST, sep=';'))
raw_df_train = pd.read_csv(DATA_PATH + FILE_TRAIN, sep=';')[N_OFFSET_TRAIN:N_OFFSET_TRAIN + N_TRAIN].reset_index()
raw_df_test = pd.read_csv(DATA_PATH + FILE_TEST, sep=';')[:N_TEST].reset_index()

raw_df_train["Time_open"] = raw_df_train["Time_open"].apply(lambda x: dt.datetime.strptime(str(x), "%Y%m%d%H%M%S"))
raw_df_test["Time_open"] = raw_df_test["Time_open"].apply(lambda x: dt.datetime.strptime(str(x), "%Y%m%d%H%M%S"))

raw_df_train["L"] = raw_df_train["Close"].apply(m.log)
raw_df_test["L"] = raw_df_test["Close"].apply(m.log)

b_name = f"bline_L_{1}_{EWA_WINDOW}"
ema_name = f"ema_L_{EWA_WINDOW}"
b_diff_shift_name = f"diff_shifted_bline_L_{1}_{EWA_WINDOW}"
R_diff_name = f"diff_ema_L_{EWA_WINDOW}"

raw_df_train[ema_name] = raw_df_train["L"].ewm(EWA_WINDOW).mean()
raw_df_train[b_name] = raw_df_train["L"] - raw_df_train[ema_name]
raw_df_train[b_diff_shift_name] = raw_df_train[b_name].diff(1).shift(-1).fillna(0)

raw_df_test[ema_name] = raw_df_test["L"].ewm(EWA_WINDOW).mean()
raw_df_test[b_name] = raw_df_test["L"] - raw_df_test[ema_name]
raw_df_test[R_diff_name] = raw_df_test[ema_name].diff(1).fillna(0)

bline_L_array_train = []
for k in range(SCALAR_WINDOW, N_TRAIN+1):
    bline_L_array_train.append(raw_df_train[b_name].iloc[k-SCALAR_WINDOW: k])
bline_L_array_train = np.array(bline_L_array_train)

bline_L_diff_shift_array_train = []
for k in range(SCALAR_WINDOW-1, N_TRAIN):
    bline_L_diff_shift_array_train.append(raw_df_train[b_diff_shift_name].iloc[k])
bline_L_diff_shift_array_train = np.array(bline_L_diff_shift_array_train)


# --- Implementation of the Algorithm ---

predicted_moves = []
predicted_b_moves = []
predicted_R_moves = []
for n in range(SCALAR_WINDOW, N_TEST+1):
    b_current = np.array(raw_df_test[b_name][n-SCALAR_WINDOW:n])
    max_scores = [-1 for _ in range(N_MAX)]
    k_max_scores = [0 for _ in range(N_MAX)]
    scores = np.dot(bline_L_array_train, b_current) / (np.linalg.norm(bline_L_array_train, axis=1)*np.linalg.norm(b_current))
    for k in range(N_TRAIN - SCALAR_WINDOW):
        i = -1
        while i+1 < N_MAX and scores[k] >= max_scores[i+1]:
            i += 1
        if i >= 0:
            max_scores = max_scores[1:i+1] + [scores[k]] + max_scores[i+1:]
            k_max_scores = k_max_scores[1:i+1] + [k] + k_max_scores[i+1:]
    mean_move = np.mean(bline_L_diff_shift_array_train[k_max_scores])
    #adj_factor = 1
    adj_factor = np.linalg.norm(b_current) / np.mean(np.linalg.norm(bline_L_array_train[k_max_scores], axis=1))  # can be modified
    predicted_move_b = adj_factor * mean_move
    predicted_move_R = raw_df_test[R_diff_name][n-1]
    predicted_moves.append(predicted_move_b + predicted_move_R)
    predicted_b_moves.append(predicted_move_b)
    predicted_R_moves.append(predicted_move_R)
    if (n-1) % 100 == 0:
        print(f"n = {n-1} prediction done, k_max_scores = {k_max_scores}, max_scores = {max_scores}, {adj_factor}")

raw_df_test["Predicted_move"] = pd.Series([0 for _ in range(SCALAR_WINDOW-1)] + predicted_moves)
raw_df_test["Predicted_b_move"] = pd.Series([0 for _ in range(SCALAR_WINDOW-1)] + predicted_b_moves)
raw_df_test["Predicted_R_move"] = pd.Series([0 for _ in range(SCALAR_WINDOW-1)] + predicted_R_moves)


# --- Plotting and results ---

plt.figure()
a = raw_df_test["Predicted_move"]
b = raw_df_test["L"].diff(1).shift(-1).fillna(0)
plt.plot(pd.Series.cumsum(a*b/(a.std()*b.std()*N_TEST)))
plt.title("Normalized cumsum: predicted move / next move")

plt.figure()
a = raw_df_test["Predicted_b_move"]
b = raw_df_test[b_name].diff(1).shift(-1).fillna(0)
plt.plot(pd.Series.cumsum(a*b/(a.std()*b.std()*N_TEST)))
plt.title("Normalized cumsum: predicted b move / next b move")

plt.figure()
plt.plot(raw_df_train["Time_open"], raw_df_train["Close"])
plt.title("Train data")

plt.figure()
plt.plot(raw_df_test["Time_open"], raw_df_test["Close"])
plt.title("Test data")
