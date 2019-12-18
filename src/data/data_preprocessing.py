import os
import sys
import pprint
import numpy as np
import pandas as pd
from datetime import datetime
from warnings import simplefilter

simplefilter(action='ignore')

TRAINING_DATA = "training.txt"
VALIDATE_DATA = "validate.txt"
TEST_DATA = "test.txt"

result_content = []

days = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

def load_data():
    pd.set_option('expand_frame_repr', False)
    train = pd.read_csv(TRAINING_DATA)
    validate = pd.read_csv(VALIDATE_DATA)
    test = pd.read_csv(TEST_DATA)

    return [train, validate, test]


def date_str_to_date_obj(date_str):
    return datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')

def convert_dates(data_frame_list):
    for data_frame in data_frame_list:
        for i, date in enumerate(data_frame['Date']):
            data_frame.iloc[i, data_frame.columns.get_loc('Date')] = date_str_to_date_obj(date)

def add_weekend_and_working__hours(data_frame_list):
    for data_frame in data_frame_list:
        data_frame.loc[:, 'Weekend'] = 0
        data_frame.loc[:, 'WorkingHour'] = 0

        for i, date in enumerate(data_frame['Date']):
            if ((days[date.weekday()] == 'Sunday') or \
                    (days[date.weekday()] == 'Saturay')):
                data_frame.iloc[i, data_frame.columns.get_loc('Weekend')] = 1

            if ((date.time() >= datetime.strptime('07:30', '%H:%M').time()) and \
                    (date.time() >= datetime.strptime('07:30', '%H:%M').time())):
                data_frame.iloc[i, data_frame.columns.get_loc('WorkingHour')] = 1

def pre_processing(data_frame_list):
    convert_dates(data_frame_list)
    #add_weekend_and_working__hours(data_frame_list)

    train = data_frame_list[0]
    validate = data_frame_list[1]
    test = data_frame_list[2]

    columns = ["SerialNumber", "Date", "Temperature", "Humidity", "Light",
               "CO2", "HumidityRatio", "Occupancy"]

    if not os.path.exists('processed'):
        os.makedirs('processed')

    train.to_csv("processed/training.csv", index=False, columns=columns)
    validate.to_csv("processed/validate.csv", index=False, columns=columns)
    test.to_csv("processed/test.csv", index=False, columns=columns)

    return

if __name__ == "__main__":
    data_frame_list= load_data()
    feature_list = pre_processing(data_frame_list)
