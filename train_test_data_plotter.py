import sys
import struct
import numpy as np
import sklearn
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import csv


def plot_train():
    print("Plotting Train Data")
    train_data = []
    with open(
        "data/plot_data/train_data/train_data_downsampled.csv", "r", newline=""
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

    plot_fig = plt.figure()
    plt.plot(plot_x, plot_y_x, label="X")
    plt.plot(plot_x, plot_y_y, label="Y")
    plt.plot(plot_x, plot_y_z, label="Z")
    plt.plot(plot_x, plot_y_tag, label="Tag")
    plt.legend()
    plt.title("Train Data")
    plt.xlabel("Time")
    plt.ylabel("Values")
    plt.savefig("data/plot_data/train_data/train_data.png")
    plt.close()


def plot_test():
    print("Plotting Test Data")
    train_data = []
    with open(
        "data/plot_data/test_data/test_data_downsampled.csv", "r", newline=""
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
    plot_y_pred = [int(float(v[5])) for v in train_data]

    plot_fig = plt.figure()
    plt.plot(plot_x, plot_y_x, label="X")
    plt.plot(plot_x, plot_y_y, label="Y")
    plt.plot(plot_x, plot_y_z, label="Z")
    plt.plot(plot_x, plot_y_tag, label="Tag")
    plt.plot(plot_x, plot_y_pred, label="Tag Pred")
    plt.legend()
    plt.title("Test Data")
    plt.xlabel("Time")
    plt.ylabel("Values")
    plt.savefig("data/plot_data/test_data/test_data.png")
    plt.close()


if __name__ == "__main__":
    plot_train()
    plot_test()
    print("Done")
