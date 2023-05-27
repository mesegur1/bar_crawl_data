# Experiment with barcrawl data
import torch
import numpy as np
from rctorch import *
import torchhd
from torchhd import embeddings
from torchhd import models
from data_reader import load_data
from data_reader import load_accel_data_full
import torchmetrics
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
from tqdm import tqdm
import csv
import getopt, sys

# Hyperparameters
# Changing these affects performance up or down depending on PID
DIMENSIONS = 6000
NUM_CHANNELS = 3
NUM_SIGNAL_LEVELS = 100
NUM_TAC_LEVELS = 2
LEARNING_RATE = 0.035
SIGNAL_X_MIN = -5
SIGNAL_X_MAX = 5
SIGNAL_Y_MIN = -5
SIGNAL_Y_MAX = 5
SIGNAL_Z_MIN = -5
SIGNAL_Z_MAX = 5

# Data windowing settings
WINDOW = 200
WINDOW_STEP = 50
START_OFFSET = 0
END_INDEX = 1200000
TRAINING_EPOCHS = 1
SAMPLE_RATE = 20
TEST_RATIO = 0.30

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

pid_data_sets = {}
PIDS = [
    "BK7610",
    # "BU4707",
    # "CC6740",
    # "DC6359",
    # "DK3500",
    # "HV0618",
    # "JB3156",
    # "JR8022",
    # "MC7070",
    # "MJ8002",
    # "PC6771",
    # "SA0297",
    # "SF3079",
]


# HDC Encoder for Bar Crawl Data
class HDCEncoder(torch.nn.Module):
    def __init__(self, levels: int, timestamps: int, out_dimension: int):
        super(HDCEncoder, self).__init__()

        self.signal_level_x = embeddings.Level(
            levels,
            out_dimension,
            dtype=torch.float64,
            low=SIGNAL_X_MIN,
            high=SIGNAL_X_MAX,
        )
        self.signal_level_y = embeddings.Level(
            levels,
            out_dimension,
            dtype=torch.float64,
            low=SIGNAL_Y_MIN,
            high=SIGNAL_Y_MAX,
        )
        self.signal_level_z = embeddings.Level(
            levels,
            out_dimension,
            dtype=torch.float64,
            low=SIGNAL_Z_MIN,
            high=SIGNAL_Z_MAX,
        )
        self.channel_basis = embeddings.Random(
            NUM_CHANNELS, out_dimension, dtype=torch.float64
        )
        self.timestamps = embeddings.Thermometer(
            timestamps, out_dimension, dtype=torch.float64, low=0, high=timestamps
        )

    # Encode window of feature vectors (x,y,z)
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Get level hypervectors for x, y, z samples
        x_signal = torch.nn.functional.normalize(input[:, 1], dim=0)
        y_signal = torch.nn.functional.normalize(input[:, 2], dim=0)
        z_signal = torch.nn.functional.normalize(input[:, 3], dim=0)
        x_levels = self.signal_level_x(x_signal)
        y_levels = self.signal_level_y(y_signal)
        z_levels = self.signal_level_z(z_signal)
        # Get time hypervectors
        times = self.timestamps(input[:, 0])
        # Bind time sequence for x, y, z samples
        x_hypervector = torchhd.multiset(torchhd.bind(x_levels, times))
        y_hypervector = torchhd.multiset(torchhd.bind(y_levels, times))
        z_hypervector = torchhd.multiset(torchhd.bind(z_levels, times))
        sample_hvs = torch.stack((x_hypervector, y_hypervector, z_hypervector))
        # Data fusion of channels
        sample_hvs = torchhd.bind(self.channel_basis.weight, sample_hvs)
        sample_hv = torchhd.multiset(sample_hvs)
        # Apply activation function
        # sample_hv = torch.tanh(sample_hv)
        return sample_hv


# Load all data for each pid
def load_all_pid_data():
    for pid in PIDS:
        # Load data from CSVs
        train_set, test_set = load_data(
            pid, END_INDEX, START_OFFSET, WINDOW, WINDOW_STEP, SAMPLE_RATE, TEST_RATIO
        )
        pid_data_sets[pid] = (train_set, test_set)


def train_rcn(pid: str):
    hps = {
        "n_nodes": 500,
        "connectivity": 0.5,
        "spectral_radius": 1.2,
        "regularization": 1.5,
        "leaking_rate": 0.01,
        "bias": 0.48,
    }
    plotting_args = {"ylabel_pred": r"$x$, $p$"}
    train_set, test_set = pid_data_sets[pid]
    rcn_x = RcNetwork(**hps, feedback=True)
    rcn_y = RcNetwork(**hps, feedback=True)
    rcn_z = RcNetwork(**hps, feedback=True)
    for x, y in tqdm(train_set[0:100], desc="Loading Training Data"):
        # Format for RCN
        d = torch.tensor(x, dtype=torch.float32, device=device)
        labels = torch.tensor(y, dtype=torch.float32, device=device)
        rcn_x.fit(X=labels, y=d[:, [0, 1]])
        rcn_y.fit(X=labels, y=d[:, [0, 2]])
        rcn_z.fit(X=labels, y=d[:, [0, 3]])

    for x, y in tqdm(test_set[0:50], desc="Loading Test Data"):
        # Format for RCN
        d = torch.tensor(x, dtype=torch.float32, device=device)
        labels = torch.tensor(y, dtype=torch.float32, device=device)
        score1, pred1 = rcn_x.test(X=labels, y=d[:, [0, 1]])
        score2, pred2 = rcn_y.test(X=labels, y=d[:, [0, 2]])
        score3, pred3 = rcn_z.test(X=labels, y=d[:, [0, 3]])
    rcn_x.combined_plot(**plotting_args)
    rcn_y.combined_plot(**plotting_args)
    rcn_z.combined_plot(**plotting_args)


# Run train for a given pid, with provided model and encoder
def run_train_for_pid(pid: str, model: models.Centroid, encode: HDCEncoder):
    train_set, _ = pid_data_sets[pid]

    # Train using training set half
    print("Begin training with pid %s data" % pid)
    with torch.no_grad():
        for e in range(0, TRAINING_EPOCHS):
            print("Training Epoch %d" % (e))
            for x, y in tqdm(train_set):
                input_tensor = torch.tensor(x, dtype=torch.float64, device=device)
                input_hypervector = encode(input_tensor)
                input_hypervector = input_hypervector.unsqueeze(0)
                label_tensor = torch.tensor(y[-1], dtype=torch.int64, device=device)
                label_tensor = label_tensor.unsqueeze(0)
                model.add_online(input_hypervector, label_tensor, lr=LEARNING_RATE)


def run_test_for_pid(pid: str, model: models.Centroid, encode: HDCEncoder):
    _, test_set = pid_data_sets[pid]

    # Test using test set half
    print("Begin Predicting for pid %s" % pid)
    accuracy = torchmetrics.Accuracy(
        "multiclass",
        num_classes=NUM_TAC_LEVELS,
    )
    accuracy = accuracy.to(device)
    with torch.no_grad():
        for x, y in tqdm(test_set):
            query_tensor = torch.tensor(x, dtype=torch.float64, device=device)
            query_hypervector = encode(query_tensor)
            output = model(query_hypervector, dot=False)
            y_pred = torch.argmax(output).unsqueeze(0).to(device)
            label_tensor = torch.tensor(y[-1], dtype=torch.int64, device=device)
            label_tensor = label_tensor.unsqueeze(0)
            accuracy.update(y_pred, label_tensor)

    print(f"Testing accuracy of model is {(accuracy.compute().item() * 100):.3f}%")
    return accuracy.compute().item() * 100


# Run a test for a pid, only training using that pid's data
def run_individual_train_and_test_for_pid(pid: str):
    # Create Centroid model
    model = models.Centroid(
        DIMENSIONS, NUM_TAC_LEVELS, dtype=torch.float64, device=device
    )
    # Create Encoder module
    encode = HDCEncoder(NUM_SIGNAL_LEVELS, WINDOW, DIMENSIONS)
    encode = encode.to(device)

    # Run training
    run_train_for_pid(pid, model, encode)

    # Run Testing
    accuracy = run_test_for_pid(pid, model, encode)

    return accuracy


if __name__ == "__main__":
    print("Using {} device".format(device))

    load_accel_data_full()
    load_all_pid_data()

    train_rcn(PIDS[0])

    # # Remove 1st argument from the
    # # list of command line arguments
    # argumentList = sys.argv[1:]

    # # Options
    # options = "m:"

    # # Long options
    # long_options = ["Mode="]

    # try:
    #     # Parsing argument
    #     arguments, values = getopt.getopt(argumentList, options, long_options)
    #     mode = 0
    #     # Checking each argument
    #     for currentArgument, currentValue in arguments:
    #         if currentArgument in ("-m", "--Mode"):
    #             if currentValue == str(0):
    #                 mode = 0
    #             elif currentValue == str(1):
    #                 mode = 1
    #             else:
    #                 mode = 0

    #     if mode == 0:
    #         print("Single-Train, Single-Test Mode")
    #         # Load datasets in windowed format
    #         load_accel_data_full()
    #         load_all_pid_data()

    #         with open("hdc_output_single.csv", "w", newline="") as file:
    #             writer = csv.writer(file)
    #             for pid in PIDS:
    #                 accuracy = run_individual_train_and_test_for_pid(pid)
    #                 writer.writerow([pid, accuracy])
    #                 file.flush()
    #             file.close()
    #         print("All tests done")
    #     elif mode == 1:
    #         print("Combined-Train, Combined-Test Mode")
    #         # Load datasets in windowed format
    #         load_accel_data_full()
    #         load_all_pid_data()

    #         # Create common Centroid model
    #         model = models.Centroid(
    #             DIMENSIONS, NUM_TAC_LEVELS, dtype=torch.float64, device=device
    #         )
    #         # Create common HDCEncoder module
    #         encode = HDCEncoder(NUM_SIGNAL_LEVELS, WINDOW, DIMENSIONS)
    #         encode = encode.to(device)

    #         with open("hdc_output_combined.csv", "w", newline="") as file:
    #             writer = csv.writer(file)
    #             for pid in PIDS:
    #                 run_train_for_pid(pid, model, encode)
    #             for pid in PIDS:
    #                 accuracy = run_test_for_pid(pid, model, encode)
    #                 writer.writerow([pid, accuracy])
    #                 file.flush()
    #             file.close()
    #         print("All tests done")

    # except getopt.error as err:
    #     # output error, and return with an error code
    #     print(str(err))
