import pytz
from datetime import datetime
import MetaTrader5 as mt5
import pandas as pd
pd.set_option('display.max_columns', 500)  # number of columns to be displayed
pd.set_option('display.width', 1500)      # max table width to display
# import pytz module for working with time zone

# establish connection to MetaTrader 5 terminal
mt5.initialize()
# file name to export to csv
file_name = 'EURUSD_H4_20220103_20220203.csv'
# set time zone to UTC
timezone = pytz.timezone("Etc/GMT-2")
# create 'datetime' objects in UTC time zone to avoid the implementation of a local time zone offset
utc_from = datetime(2022, 1, 3, tzinfo=timezone)
utc_to = datetime(2022, 2, 3, tzinfo=timezone)
# get bars from EURUSD H4 within the interval of 2021.05.03 00:00 - 2022.01.03 00:00 in GMT-2 time zone
rates = mt5.copy_rates_range("EURUSD", mt5.TIMEFRAME_H4, utc_from, utc_to)

# shut down connection to the MetaTrader 5 terminal
mt5.shutdown()

# display each element of obtained data in a new line
print("Display obtained data 'as is'")
counter = 0
for rate in rates:
    counter += 1
    if counter <= 10:
        print(rate)

# create DataFrame out of the obtained data
rates_frame = pd.DataFrame(rates)
# convert time in seconds into the 'datetime' format
rates_frame['time'] = pd.to_datetime(rates_frame['time'], unit='s')

# display data
print("\nDisplay dataframe with data")
print(rates_frame.head(10))
rates_frame.to_csv(file_name, encoding='utf-8', index=False)
