import sys
import struct
import numpy as np
import sklearn
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import csv

MS_PER_SEC = 1000

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


def plot_pid_tac(pid: str):
    print("Reading in Data for person %s" % (pid))
    tac_data = pd.read_csv("data/clean_tac/%s_clean_TAC.csv" % pid)
    tac_data["timestamp"] = tac_data["timestamp"].astype("datetime64[s]")
    tac_data["TAC_Reading"] = tac_data["TAC_Reading"].astype(float)
    tac_data = tac_data.rename(columns={"timestamp": "time"})
    tac_data = tac_data.set_index("time")

    tac_data_x = tac_data.index.astype("int64")
    tac_data_y = tac_data["TAC_Reading"].values

    pid_plot_fig = plt.figure()
    plt.plot(tac_data_x, tac_data_y)
    plt.title("TAC Readings for %s" % pid)
    plt.xlabel("Time")
    plt.ylabel("TAC Reading")
    plt.savefig("data/plot_data/raw_data_plots/tac_readings_%s.png" % pid)
    plt.close()


def plot_pid_acc(pid: str):
    # Get specific accelerometer data
    accel_data_specific = accel_data_full.query("pid == @pid")
    if pid == "JB3156" or pid == "CC6740":
        # skip first row (dummy data)
        accel_data_specific = accel_data_specific.iloc[1:-1]
    acc_x = accel_data_specific.index.astype("int64")
    acc_y1 = accel_data_specific["x"].values
    acc_y2 = accel_data_specific["y"].values
    acc_y3 = accel_data_specific["z"].values

    pid_plot_fig = plt.figure()
    plt.plot(acc_x, acc_y1, label="X")
    plt.plot(acc_x, acc_y2, label="Y")
    plt.plot(acc_x, acc_y3, label="Z")
    plt.legend()
    plt.title("Acc Readings for %s" % pid)
    plt.xlabel("Time")
    plt.ylabel("Acc Amount")
    plt.savefig("data/plot_data/raw_data_plots/acc_readings_%s.png" % pid)
    plt.close()


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
