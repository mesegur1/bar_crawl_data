import sys
import struct
import numpy as np
import sklearn
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import csv


def plot_pid_train(pid: str):
    print("Plotting Train Data for person %s" % (pid))
    train_data = []
    with open(
        "data/plot_data/train_data/%s_train_data_downsampled.csv" % pid, "r", newline=""
    ) as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            train_data.append(row)
        file.close()

    plot_x = [int(float(v[0])) for v in train_data]
    plot_y_x = [float(v[1]) for v in train_data]
    plot_y_y = [float(v[2]) for v in train_data]
    plot_y_z = [float(v[3]) for v in train_data]
    plot_y_tag = [int(float(v[4])) for v in train_data]

    pid_plot_fig = plt.figure()
    plt.plot(plot_x, plot_y_x, label="X")
    plt.plot(plot_x, plot_y_y, label="Y")
    plt.plot(plot_x, plot_y_z, label="Z")
    plt.plot(plot_x, plot_y_tag, label="Tag")
    plt.legend()
    plt.title("Train Data for %s" % pid)
    plt.xlabel("Time")
    plt.ylabel("Values")
    plt.savefig("data/plot_data/train_data/train_data_%s.png" % pid)


def plot_pid_test(pid: str):
    print("Plotting Test Data for person %s" % (pid))
    train_data = []
    with open(
        "data/plot_data/test_data/%s_test_data_downsampled.csv" % pid, "r", newline=""
    ) as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            train_data.append(row)
        file.close()

    plot_x = [int(float(v[0])) for v in train_data]
    plot_y_x = [float(v[1]) for v in train_data]
    plot_y_y = [float(v[2]) for v in train_data]
    plot_y_z = [float(v[3]) for v in train_data]
    plot_y_tag = [int(float(v[4])) for v in train_data]

    pid_plot_fig = plt.figure()
    plt.plot(plot_x, plot_y_x, label="X")
    plt.plot(plot_x, plot_y_y, label="Y")
    plt.plot(plot_x, plot_y_z, label="Z")
    plt.plot(plot_x, plot_y_tag, label="Tag")
    plt.legend()
    plt.title("Test Data for %s" % pid)
    plt.xlabel("Time")
    plt.ylabel("Values")
    plt.savefig("data/plot_data/test_data/test_data_%s.png" % pid)


if __name__ == "__main__":
    PIDS = [
        # "BK7610",
        # "BU4707",
        # "CC6740",
        # "DC6359",
        # "DK3500",
        # "HV0618",
        # "JB3156",
        # "JR8022",
        "MC7070",
        "MJ8002",
        # "PC6771",
        # "SA0297",
        # "SF3079",
    ]

    for pid in PIDS:
        plot_pid_train(pid)
        plot_pid_test(pid)
    print("Done")
