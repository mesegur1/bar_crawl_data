import sys
import struct
import numpy as np
import sklearn
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import csv

MS_PER_SEC = 1000

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


def plot_pid_tac(pid: str):
    print("Reading in Data for person %s" % (pid))
    tac_data = []
    with open("data/clean_tac/" + pid + "_clean_TAC.csv", "r", newline="") as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            tac_data.append(row)
        file.close()

    # Get formatted TAC data
    tac_data_x = [int(v[0]) * 1000 for v in tac_data]
    tac_data_y = [float(v[1]) for v in tac_data]

    pid_plot_fig = plt.figure()
    plt.plot(tac_data_x, tac_data_y)
    plt.title("TAC Readings for %s" % pid)
    plt.xlabel("Time")
    plt.ylabel("TAC Reading")
    plt.savefig("data/plot_data/raw_data_plots/tac_readings_%s.png" % pid)


def plot_pid_acc(pid: str):
    # Get specific accelerometer data
    accel_data_specific = [
        (int(v[0]), float(v[2]), float(v[3]), float(v[4]))
        for v in accel_data_full
        if v[1] == pid
    ]
    if pid == "JB3156" or pid == "CC6740":
        # skip first row (dummy data)
        accel_data_specific = accel_data_specific[1:-1]
    acc_x = [v[0] for v in accel_data_specific]
    acc_y1 = [v[1] for v in accel_data_specific]
    acc_y2 = [v[2] for v in accel_data_specific]
    acc_y3 = [v[3] for v in accel_data_specific]

    pid_plot_fig = plt.figure()
    plt.plot(acc_x, acc_y1, label="X")
    plt.plot(acc_x, acc_y2, label="Y")
    plt.plot(acc_x, acc_y3, label="Z")
    plt.legend()
    plt.title("Acc Readings for %s" % pid)
    plt.xlabel("Time")
    plt.ylabel("Acc Amount")
    plt.savefig("data/plot_data/raw_data_plots/acc_readings_%s.png" % pid)


if __name__ == "__main__":
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
    load_accel_data_full()

    for pid in PIDS:
        plot_pid_tac(pid)
        plot_pid_acc(pid)
    print("Done")
