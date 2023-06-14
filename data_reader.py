import sys
import struct
import numpy as np
import sklearn
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


accel_data_full = []


# Load in accelerometer data into memory
def load_accel_data_full():
    global accel_data_full
    print("Read in accelerometer data")
    with open("data/all_accelerometer_data_pids_13.csv", "r", newline="") as file:
        reader = csv.reader(file)
        next(reader)
        for row in tqdm(reader):
            accel_data_full.append(row)
        file.close()


# Load entire dataset
def load_data_combined(
    window: int,
    window_step: int,
    sample_rate: int = 20,
    test_ratio: float = 0.5,
):
    global accel_data_full

    accel_data = [
        (int(v[0]), str(v[1]), float(v[2]), float(v[3]), float(v[4]))
        for v in accel_data_full
    ]
    # Skip first two rows
    accel_data = accel_data[2:]

    tac_data = {}
    for pid in PIDS:
        print("Reading in Data for person %s" % (pid))
        tac_data[pid] = []
        with open("data/clean_tac/" + pid + "_clean_TAC.csv", "r", newline="") as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                tac_data[pid].append(row)
            file.close()
        tac_data[pid] = [
            (int(v[0]) * 1000, tac_to_class(float(v[1]))) for v in tac_data[pid]
        ]
    # Get TAC labels for each accel data sample
    tac_data_labels = []
    for a in accel_data:
        for d in tac_data[a[1]]:
            if d[0] > a[0]:
                tac_data_labels.append(d[1])
                break
    accel_data = accel_data[0 : len(tac_data_labels) : int(sample_rate / 10)]
    tac_data_labels = tac_data_labels[0 : len(tac_data_labels) : int(sample_rate / 10)]
    # Remove PID
    accel_data = [(v[0], v[2], v[3], v[4]) for v in accel_data]

    print("Creating data sets")
    # Split data into two parts
    train_data_accel, test_data_accel, train_data_tac, test_data_tac = train_test_split(
        np.array(accel_data),
        np.array(tac_data_labels),
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
    tac_data = []
    with open("data/clean_tac/%s_clean_TAC.csv" % pid, "r", newline="") as file:
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
    if limit > len(accel_data_specific):
        limit = len(accel_data_specific)
    accel_data_specific = accel_data_specific[offset:limit]
    if pid == "JB3156" or pid == "CC6740":
        # skip first row (dummy data)
        accel_data_specific = accel_data_specific[1:-1]

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
    if accel_data[0][0] > tac_data[0][0]:
        # Cut off data before start of Accel data
        while j < tac_data_length and accel_data[0][0] > tac_data[j][0]:
            j += 1
    elif accel_data[0][0] < tac_data[0][0]:
        # Cut off data before start of TAC data
        while i < accel_data_length and accel_data[i][0] < tac_data[0][0]:
            i += 1
    tac_data_labels.append(tac_data[j][1])
    acc_new_start = i
    while i < accel_data_length and j < tac_data_length:
        if accel_data[i][0] < tac_data[j][0]:
            tac_data_labels.append(tac_data[j][1])
        else:
            j = j + 1  # Move to next TAC entry
        i = i + 1  # Go to next accel data
    accel_data = accel_data[acc_new_start : acc_new_start + len(tac_data_labels)]
    print("Total Data length: %d" % (len(tac_data_labels)))

    print("Creating data sets")
    # Split data into two parts
    train_data_accel, test_data_accel, train_data_tac, test_data_tac = train_test_split(
        np.array(accel_data),
        np.array(tac_data_labels),
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
