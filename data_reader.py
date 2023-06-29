import sys
import struct
import numpy as np
import sklearn
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import csv

TAC_LEVEL_0 = 0  # < 0.080 g/dl
TAC_LEVEL_1 = 1  # >= 0.080 g/dl

MS_PER_SEC = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIDS = [
    "BK7610",
    "BU4707",
    "CC6740",
    "DC6359",
    "DK3500",
    "HV0618",
    "JB3156",
    "JR8022",
    "MC7070",
    "MJ8002",
    "PC6771",
    "SA0297",
    "SF3079",
]


# Convert TAC measurement to a class
def tac_to_class(tac: float):
    if tac < 0:
        tac = 0
    tac = round(tac, 3) * 1000
    if tac < 80:
        return TAC_LEVEL_0
    else:
        return TAC_LEVEL_1


accel_data_full = pd.DataFrame([])  # = []


# Load in accelerometer data into memory
def load_accel_data_full():
    global accel_data_full
    print("Read in accelerometer data")
    accel_data_full = pd.read_csv("data/all_accelerometer_data_pids_13.csv")
    accel_data_full["time"] = accel_data_full["time"].astype("datetime64[ms]")
    accel_data_full["pid"] = accel_data_full["pid"].astype(str)
    accel_data_full["x"] = accel_data_full["x"].astype(float)
    accel_data_full["y"] = accel_data_full["y"].astype(float)
    accel_data_full["z"] = accel_data_full["z"].astype(float)
    accel_data_full = accel_data_full.set_index("time")


# Load entire dataset
def load_data_combined(
    window: int,
    window_step: int,
    sample_rate: int = 20,
    test_ratio: float = 0.5,
):
    global accel_data_full

    accel_data = accel_data_full
    # skip two rows (dummy data)
    accel_data = accel_data.iloc[2:-1]

    # Down sample accelerometer data
    accel_data = accel_data.resample("%dL" % (MS_PER_SEC / sample_rate)).first()

    tac_data = {}
    for pid in PIDS:
        print("Reading in Data for person %s" % (pid))
        tac_data[pid] = pd.read_csv("data/clean_tac/%s_clean_TAC.csv" % pid)
        tac_data[pid]["timestamp"] = tac_data[pid]["timestamp"].astype("datetime64[s]")
        tac_data[pid]["TAC_Reading"] = tac_data[pid]["TAC_Reading"].astype(float)
        # Get formatted TAC data
        tac_data[pid]["TAC_Reading"] = (
            tac_data[pid]["TAC_Reading"].map(lambda tac: tac_to_class(tac)).astype(int)
        )
        tac_data[pid] = tac_data[pid].rename(columns={"timestamp": "time"})
        tac_data[pid] = tac_data[pid].set_index("time")

    # Combine Data Frames to perform interpolation and backfilling
    input_data = accel_data.join(list(tac_data.values()), how="outer")
    input_data = input_data.apply(pd.Series.interpolate, args=("time",))
    input_data = input_data.fillna(method="backfill")
    input_data["time"] = input_data.index
    input_data["time"] = input_data["time"].astype("int64")

    print("Total Data length: %d" % (len(input_data.index)))

    # Split data back into two parts for train/test set creation
    accel_data = input_data[["time", "x", "y", "z"]].to_numpy()
    tac_data_labels = input_data["TAC_Reading"].to_numpy()

    print("Creating data sets")
    # Split data into two parts
    train_data_accel, test_data_accel, train_data_tac, test_data_tac = train_test_split(
        accel_data,
        tac_data_labels,
        test_size=test_ratio,
        shuffle=False,
    )
    train_length = train_data_accel.shape[0]
    test_length = test_data_accel.shape[0]

    # Change training data to be windowed
    train_data_accel = [
        train_data_accel[base : base + window]
        for base in range(0, len(train_data_accel), window_step)
    ]
    train_data_tac = [
        train_data_tac[base : base + window]
        for base in range(0, len(train_data_tac), window_step)
    ]

    # Change test data to be windowed
    test_data_accel = [
        test_data_accel[base : base + window]
        for base in range(0, len(test_data_accel), window_step)
    ]
    test_data_tac = [
        test_data_tac[base : base + window]
        for base in range(0, len(test_data_tac), window_step)
    ]

    train_set = tuple(zip(train_data_accel, train_data_tac))
    print("Data Length For Training: %d" % (train_length))
    test_set = tuple(zip(test_data_accel, test_data_tac))
    print("Data Length For Testing: %d" % (test_length))

    return (train_set, test_set)


# Load data from CSVs
def load_data(
    pid: str,
    limit: int,
    offset: int,
    window: int,
    window_step: int,
    sample_rate: int = 20,
    test_ratio: float = 0.5,
):
    global accel_data_full
    print("Reading in Data for person %s" % (pid))
    tac_data = pd.read_csv("data/clean_tac/%s_clean_TAC.csv" % pid)
    tac_data["timestamp"] = tac_data["timestamp"].astype("datetime64[s]")
    tac_data["TAC_Reading"] = tac_data["TAC_Reading"].astype(float)
    # Get formatted TAC data
    tac_data["TAC_Reading"] = (
        tac_data["TAC_Reading"].map(lambda tac: tac_to_class(tac)).astype(int)
    )
    tac_data = tac_data.rename(columns={"timestamp": "time"})
    tac_data = tac_data.set_index("time")

    # Get specific accelerometer data
    accel_data_specific = accel_data_full.query("pid == @pid")
    if pid == "JB3156" or pid == "CC6740":
        # skip first row (dummy data)
        accel_data_specific = accel_data_specific.iloc[1:-1]

    # Down sample accelerometer data
    accel_data = accel_data_specific.resample(
        "%dL" % (MS_PER_SEC / sample_rate)
    ).first()

    # if limit > len(accel_data_specific.index):
    #     limit = len(accel_data_specific.index)
    # accel_data_specific = accel_data_specific.iloc[offset:limit]

    # Combine Data Frames to perform interpolation and backfilling
    input_data = accel_data.join(tac_data, how="outer")
    input_data = input_data.apply(pd.Series.interpolate, args=("time",))
    input_data = input_data.fillna(method="backfill")
    input_data["time"] = input_data.index
    input_data["time"] = input_data["time"].astype("int64")

    print("Total Data length: %d" % (len(input_data.index)))

    # Split data back into two parts for train/test set creation
    accel_data = input_data[["time", "x", "y", "z"]].to_numpy()
    tac_data_labels = input_data["TAC_Reading"].to_numpy()

    print("Creating data sets")
    # Split data into two parts
    train_data_accel, test_data_accel, train_data_tac, test_data_tac = train_test_split(
        accel_data,
        tac_data_labels,
        test_size=test_ratio,
        shuffle=False,
    )
    train_length = train_data_accel.shape[0]
    test_length = test_data_accel.shape[0]

    # Change training data to be windowed
    train_data_accel = [
        train_data_accel[base : base + window]
        for base in range(0, len(train_data_accel), window_step)
    ]
    train_data_tac = [
        train_data_tac[base : base + window]
        for base in range(0, len(train_data_tac), window_step)
    ]

    # Change test data to be windowed
    test_data_accel = [
        test_data_accel[base : base + window]
        for base in range(0, len(test_data_accel), window_step)
    ]
    test_data_tac = [
        test_data_tac[base : base + window]
        for base in range(0, len(test_data_tac), window_step)
    ]

    train_set = tuple(zip(train_data_accel, train_data_tac))
    print("Data Length For Training: %d" % (train_length))
    test_set = tuple(zip(test_data_accel, test_data_tac))
    print("Data Length For Testing: %d" % (test_length))

    return (train_set, test_set)
