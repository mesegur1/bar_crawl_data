# Experiment with barcrawl data
import torch
import torch.nn.functional as F
import numpy as np
import torchhd
from torchhd_custom import embeddings
from torchhd_custom import models  # from torchhd import models
from data_reader import load_data
from data_reader import load_accel_data_full
import torchmetrics
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import f1_score
from tqdm import tqdm
import csv
import getopt, sys

# Hyperparameters
# Changing these affects performance up or down depending on PID
DIMENSIONS = 6000
NUM_CHANNEL = 3
NUM_TAC_LEVELS = 2
LEARNING_RATE = 0.8

# Type of kernel trick
USE_TANH = True

# Data windowing settings
WINDOW = 100
WINDOW_STEP = 50
START_OFFSET = 0
END_INDEX = 1200000
TRAINING_EPOCHS = 1
SAMPLE_RATE = 20
TEST_RATIO = 0.30

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pid_data_sets = {}
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


# HDC Encoder for Bar Crawl Data
class HDCEncoder(torch.nn.Module):
    def __init__(self, timestamps: int, out_dimension: int):
        super(HDCEncoder, self).__init__()

        self.timestamps = timestamps
        # RBF Kernel Trick
        if USE_TANH == True:
            self.kernel = embeddings.HyperTangent(
                timestamps * NUM_CHANNEL, out_dimension, dtype=torch.float64
            )
        else:
            self.kernel = embeddings.Sinusoid(
                timestamps * NUM_CHANNEL, out_dimension, dtype=torch.float64
            )

    # Encode window of feature vectors (x,y,z)
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Get features from x, y, z samples
        window = input.size(0)  # sample count
        if window < self.timestamps:
            # Pad the inputs to required length
            padding = self.timestamps - window
            x_signal = F.pad(
                input=input[:, 1], pad=(0, padding), mode="constant", value=0
            )
            y_signal = F.pad(
                input=input[:, 2], pad=(0, padding), mode="constant", value=0
            )
            z_signal = F.pad(
                input=input[:, 3], pad=(0, padding), mode="constant", value=0
            )
        else:
            x_signal = input[:, 1]
            y_signal = input[:, 2]
            z_signal = input[:, 3]
        features = torch.cat((x_signal, y_signal, z_signal))
        # Use kernel encoder
        sample_hv = self.kernel(features)
        return sample_hv


# Load all data for each pid
def load_all_pid_data():
    for pid in PIDS:
        # Load data from CSVs
        train_set, test_set = load_data(
            pid, END_INDEX, START_OFFSET, WINDOW, WINDOW_STEP, SAMPLE_RATE, TEST_RATIO
        )
        pid_data_sets[pid] = (train_set, test_set)


# Run train for a given pid, with provided model and encoder
def run_train_for_pid(pid: str, model: models.Centroid, encode: HDCEncoder):
    train_set, _ = pid_data_sets[pid]

    # Train using training set half
    print("Begin training with pid %s data" % pid)
    with open(
        "data/plot_data/train_data/%s_train_data_downsampled.csv" % pid, "w", newline=""
    ) as file:
        writer = csv.writer(file)
        with torch.no_grad():
            for e in range(0, TRAINING_EPOCHS):
                print("Training Epoch %d" % (e))
                for x, y in tqdm(train_set):
                    input_tensor = torch.tensor(x, dtype=torch.float64, device=device)
                    input_hypervector = encode(input_tensor)
                    # input_hypervector = input_hypervector.unsqueeze(0)
                    label_tensor = torch.tensor(y[-1], dtype=torch.int64, device=device)
                    label_tensor = label_tensor.unsqueeze(0)
                    model.add_adjust_iterative(
                        input_hypervector, label_tensor, lr=LEARNING_RATE
                    )
                    writer.writerow((x[-1][0], x[-1][1], x[-1][2], x[-1][3], y[-1]))
                    file.flush()
        file.close()


def run_test_for_pid(pid: str, model: models.Centroid, encode: HDCEncoder):
    _, test_set = pid_data_sets[pid]

    # Test using test set half
    print("Begin Predicting for pid %s" % pid)
    accuracy = torchmetrics.Accuracy(
        "multiclass",
        num_classes=NUM_TAC_LEVELS,
    )
    accuracy = accuracy.to(device)
    with open(
        "data/plot_data/test_data/%s_test_data_downsampled.csv" % pid, "w", newline=""
    ) as file:
        writer = csv.writer(file)
        y_true = []
        preds = []
        with torch.no_grad():
            for x, y in tqdm(test_set):
                query_tensor = torch.tensor(x, dtype=torch.float64, device=device)
                query_hypervector = encode(query_tensor)
                output = model(query_hypervector, dot=False)
                y_pred = torch.argmax(output).unsqueeze(0).to(device)
                label_tensor = torch.tensor(y[-1], dtype=torch.int64, device=device)
                label_tensor = label_tensor.unsqueeze(0)
                accuracy.update(y_pred, label_tensor)
                preds.append(y_pred.item())
                y_true.append(label_tensor.item())
                writer.writerow(
                    (x[-1][0], x[-1][1], x[-1][2], x[-1][3], y[-1], y_pred.item())
                )
                file.flush()
        file.close()

    print(f"Testing accuracy of model is {(accuracy.compute().item() * 100):.3f}%")
    f1 = f1_score(y_true, preds, zero_division=1)
    print(f"Testing F1 Score of model is {(f1):.3f}")
    return (accuracy.compute().item() * 100, f1)


# Run a test for a pid, only training using that pid's data
def run_individual_train_and_test_for_pid(pid: str):
    # Create Centroid model
    model = models.Centroid(
        DIMENSIONS, NUM_TAC_LEVELS, dtype=torch.float64, device=device
    )
    # Create Encoder module
    encode = HDCEncoder(WINDOW, DIMENSIONS)
    encode = encode.to(device)

    # Run training
    run_train_for_pid(pid, model, encode)

    # Run Testing
    accuracy = run_test_for_pid(pid, model, encode)

    return accuracy


if __name__ == "__main__":
    print("Using {} device".format(device))

    # Remove 1st argument from the
    # list of command line arguments
    argumentList = sys.argv[1:]

    # Options
    options = "m:"

    # Long options
    long_options = ["Mode="]

    try:
        # Parsing argument
        arguments, values = getopt.getopt(argumentList, options, long_options)
        mode = 0
        # Checking each argument
        for currentArgument, currentValue in arguments:
            if currentArgument in ("-m", "--Mode"):
                if currentValue == str(0):
                    mode = 0
                elif currentValue == str(1):
                    mode = 1
                else:
                    mode = 0

        if mode == 0:
            print("Single-Train, Single-Test Mode")
            # Load datasets in windowed format
            load_accel_data_full()
            load_all_pid_data()

            with open("hdc_rbf_output_single.csv", "w", newline="") as file:
                writer = csv.writer(file)
                for pid in PIDS:
                    accuracy, f1 = run_individual_train_and_test_for_pid(pid)
                    writer.writerow([pid, accuracy, f1])
                    file.flush()
                file.close()
            print("All tests done")
        elif mode == 1:
            print("Combined-Train, Combined-Test Mode")
            print("TODO")
            print("Test done")

    except getopt.error as err:
        # output error, and return with an error code
        print(str(err))
