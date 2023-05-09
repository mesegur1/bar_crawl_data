import sys
import struct
import numpy as np
import sklearn
from sklearn import preprocessing
import csv

TAC_LEVEL_0 = 0  # < 0.080 g/dl
TAC_LEVEL_1 = 1  # >= 0.080 g/dl

MS_PER_SEC = 1000


def tac_to_class(tac: float):
    if tac < 0:
        tac = 0

    tac = round(tac, 3) * 1000
    if tac < 80:
        return TAC_LEVEL_0
    else:
        return TAC_LEVEL_1


def load_data(
    pid: str,
    limit: int,
    offset: int,
    window: int,
    window_step: int,
    sample_rate: int = 20,
):
    print("Reading in Data for person %s" % (pid))
    # Read in accelerometer data
    accel_data_full = []
    with open("data/all_accelerometer_data_pids_13.csv", "r", newline="") as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            accel_data_full.append(row)
        file.close()
    # Read in clean TAC data
    tac_data = []
    with open("data/clean_tac/" + pid + "_clean_TAC.csv", "r", newline="") as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            tac_data.append(row)
        file.close()

    # Get specific accelerometer data
    accel_data_specific = [
        (int(v[0]), float(v[2]), float(v[3]), float(v[4]))
        for v in accel_data_full
        if v[1] == pid
    ]
    # Down sample accelerometer data
    accel_data = []
    sample_data = []
    current_time = accel_data_specific[0][0]
    for v in accel_data_specific:
        if v[0] < current_time + MS_PER_SEC:
            sample_data.append(v)
        else:
            sample_data_size = len(sample_data)
            if sample_data_size > sample_rate:
                step = int(sample_data_size / sample_rate)
                for i in range(0, sample_data_size, step):
                    accel_data.append(sample_data[i])
            else:
                accel_data.extend(sample_data)
            current_time += MS_PER_SEC
            sample_data = []
    accel_data_length = len(accel_data)

    # Get formatted TAC data
    tac_data = [(int(v[0]) * 1000, tac_to_class(float(v[1]))) for v in tac_data]
    tac_data_length = len(tac_data)

    tac_data_labels = []
    i = 0
    j = 0
    while i < accel_data_length and j < tac_data_length:
        if accel_data[i][0] < tac_data[j][0]:
            tac_data_labels.append(tac_data[j][1])
        else:
            j = j + 1  # Move to next TAC entry
        i = i + 1  # Go to next accel data

    print("Total Data length: %d" % (len(tac_data_labels)))

    if limit > len(tac_data_labels):
        limit = len(tac_data_labels)

    print("Creating data sets")
    # Create training set and test set
    # Split data into two parts
    max_data_length = limit - offset
    split_point = int((3 * max_data_length) / 4) + offset
    train_data_accel = accel_data[offset:split_point]
    train_data_accel = [
        train_data_accel[base : base + window]
        for base in range(0, len(train_data_accel), window_step)
    ]
    train_data_tac = tac_data_labels[offset:split_point]
    train_data_tac = [
        train_data_tac[base : base + window]
        for base in range(0, len(train_data_tac), window_step)
    ]
    train_set = tuple(zip(train_data_accel, train_data_tac))
    print("Data Length For Training: %d" % (split_point - offset))

    test_data_accel = accel_data[split_point:limit]
    # test_data_accel = [
    #     test_data_accel[base : base + window]
    #     for base in range(0, len(test_data_accel), window_step)
    # ]
    test_data_tac = tac_data_labels[split_point:limit]
    # test_data_tac = [
    #     test_data_tac[base : base + window]
    #     for base in range(0, len(test_data_tac), window_step)
    # ]
    test_set = tuple(zip(test_data_accel, test_data_tac))
    print("Data Length For Testing: %d" % (limit - split_point))

    return (train_set, test_set)
