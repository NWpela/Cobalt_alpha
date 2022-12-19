from binance.client import Client
import pandas as pd
import datetime as dt
import dateparser

client = Client()

# Global parameters
COIN_CODES = ["BTC"]
REFERENCE = "EUR"
CREATE_TRAIN = True
CREATE_TEST = True
INTERVAL = Client.KLINE_INTERVAL_5MINUTE
start_date_train_str = "1 October, 2019"
end_date_train_str = "1 October, 2022"
start_date_test_str = "2 October, 2022"
end_date_test_str = "15 December, 2022"
VERSION = "v1"

column_names = ["Timestamp_open",
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
                "Close_time",
                "Quote_asset_volume",
                "Number_of_trades",
                "Taker_buy_base_asset_volume",
                "Taker_buy_quote_asset_volume",
                "Ignore"]

for COIN_CODE in COIN_CODES:
    PAIR = COIN_CODE + REFERENCE

    # Start retrieval using API
    if CREATE_TRAIN:
        raw_data_train = client.get_historical_klines(PAIR, INTERVAL, start_date_train_str, end_date_train_str)

        # Formatting the data and converting to CSV
        data_df = pd.DataFrame(data=raw_data_train, columns=column_names).drop(columns=["Ignore"])
        data_df["Time_open"] = data_df["Timestamp_open"].apply(lambda x: dt.datetime.fromtimestamp(x/1000))
        data_df["Time_open"] = data_df["Time_open"].dt.strftime("%Y%m%d%H%M%S")

        start_date_code_train = dateparser.parse(start_date_train_str, settings={'TIMEZONE': "UTC"}).strftime("%Y_%m_%d")
        end_date_code_train = dateparser.parse(end_date_train_str, settings={'TIMEZONE': "UTC"}).strftime("%Y_%m_%d")
        file_name = '_'.join([PAIR, INTERVAL, start_date_code_train + '__' + end_date_code_train, "TRAIN", VERSION]) + ".csv"
        data_df.to_csv(file_name, sep=";")
        print(f"{file_name} created")

    if CREATE_TEST:
        raw_data_test = client.get_historical_klines(PAIR, INTERVAL, start_date_test_str, end_date_test_str)

        # Formatting the data and converting to CSV
        data_df = pd.DataFrame(data=raw_data_test, columns=column_names).drop(columns=["Ignore"])
        data_df["Time_open"] = data_df["Timestamp_open"].apply(lambda x: dt.datetime.fromtimestamp(x / 1000))
        data_df["Time_open"] = data_df["Time_open"].dt.strftime("%Y%m%d%H%M%S")

        start_date_code_test = dateparser.parse(start_date_test_str, settings={'TIMEZONE': "UTC"}).strftime("%Y_%m_%d")
        end_date_code_test = dateparser.parse(end_date_test_str, settings={'TIMEZONE': "UTC"}).strftime("%Y_%m_%d")
        file_name = '_'.join([PAIR, INTERVAL, start_date_code_test + '__' + end_date_code_test, "TEST", VERSION]) + ".csv"
        data_df.to_csv(file_name, sep=";")
        print(f"{file_name} created")

print("DONE")
