# Experiment with barcrawl data
import torch
import numpy as np
import torchhd
from torchhd import embeddings
from torchhd_custom import models  # from torchhd import models
from encoders import HdcLevelEncoder
from encoders import HdcRbfEncoder
from encoders import RcnHdcEncoder
from encoders import HdcSinusoidNgramEncoder
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
NUM_SIGNAL_LEVELS = 200
NUM_TAC_LEVELS = 2
LEARNING_RATE = 0.005

# Data windowing settings
WINDOW = 100
WINDOW_STEP = 50
START_OFFSET = 0
END_INDEX = np.inf
TRAINING_EPOCHS = 1
SAMPLE_RATE = 20
RCN_SAMPLE_RATE = 5
TEST_RATIO = 0.30

# Encoder options
USE_LEVEL_ENCODER = 0
USE_RBF_ENCODER = 1
USE_RCN_ENCODER = 2
USE_SINUSOID_NGRAM_ENCODER = 3


def encoder_mode_str(mode: int):
    if mode == USE_LEVEL_ENCODER:
        return "level"
    elif mode == USE_RBF_ENCODER:
        return "rbf"
    elif mode == USE_RCN_ENCODER:
        return "rcn"
    elif mode == USE_SINUSOID_NGRAM_ENCODER:
        return "sinusoid-ngram"
    else:
        return "unknown"


# Option for RBF encoder
USE_TANH = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pid_data_sets = {}
#                            This one  This one also
PIDS1 = ["BK7610", "MC7070", "MJ8002", "SF3079"]
#
PIDS2 = ["CC6740", "SA0297"]


# Load all data for each pid
def load_all_pid_data(mode: int):
    sample_rate = SAMPLE_RATE
    if mode == 2:
        # Lower sample rate for RCN encoder
        sample_rate = RCN_SAMPLE_RATE
    for pid in PIDS1:
        # Load data from CSVs
        train_set, test_set = load_data(
            pid, END_INDEX, START_OFFSET, WINDOW, WINDOW_STEP, sample_rate, TEST_RATIO
        )
        pid_data_sets[pid] = (train_set, test_set)
    # Get subset of data for these pids
    for pid in PIDS2:
        # Load data from CSVs
        if pid == "CC6740":
            end_index = 500000
        elif pid == "SA0297":
            end_index = 1000000
        else:
            end_index = END_INDEX
        train_set, test_set = load_data(
            pid, end_index, START_OFFSET, WINDOW, WINDOW_STEP, sample_rate, TEST_RATIO
        )
        pid_data_sets[pid] = (train_set, test_set)


# Run train for a given pid, with provided model and encoder
def run_train_for_pid(
    pid: str, model: models.Centroid, encode: torch.nn.Module, write_file: bool = True
):
    train_set, _ = pid_data_sets[pid]

    # Train using training set half
    print("Begin training with pid %s data" % pid)
    if write_file == True:
        with open(
            "data/plot_data/train_data/%s_train_data_downsampled.csv" % pid,
            "w",
            newline="",
        ) as file:
            writer = csv.writer(file)
            with torch.no_grad():
                for e in range(0, TRAINING_EPOCHS):
                    print("Training Epoch %d" % (e))
                    for x, y in tqdm(train_set):
                        input_tensor = torch.tensor(
                            x, dtype=torch.float64, device=device
                        )
                        input_hypervector = encode(input_tensor)
                        input_hypervector = input_hypervector.unsqueeze(0)
                        label_tensor = torch.tensor(y, dtype=torch.int64, device=device)
                        label_tensor = label_tensor.unsqueeze(0)
                        model.add_adjust_iterative(
                            input_hypervector, label_tensor, lr=LEARNING_RATE
                        )
                        writer.writerow((x[-1][0], x[-1][1], x[-1][2], x[-1][3], y))
                        file.flush()
            file.close()
    else:
        with torch.no_grad():
            for e in range(0, TRAINING_EPOCHS):
                print("Training Epoch %d" % (e))
                for x, y in tqdm(train_set):
                    input_tensor = torch.tensor(x, dtype=torch.float64, device=device)
                    input_hypervector = encode(input_tensor)
                    input_hypervector = input_hypervector.unsqueeze(0)
                    label_tensor = torch.tensor(y, dtype=torch.int64, device=device)
                    label_tensor = label_tensor.unsqueeze(0)
                    model.add_adjust_iterative(
                        input_hypervector, label_tensor, lr=LEARNING_RATE
                    )


def run_test_for_pid(
    pid: str, model: models.Centroid, encode: torch.nn.Module, write_file: bool = True
):
    _, test_set = pid_data_sets[pid]

    # Test using test set half
    print("Begin Predicting for pid %s" % pid)
    accuracy = torchmetrics.Accuracy(
        "multiclass",
        num_classes=NUM_TAC_LEVELS,
    )
    accuracy = accuracy.to(device)
    if write_file == True:
        with open(
            "data/plot_data/test_data/%s_test_data_downsampled.csv" % pid,
            "w",
            newline="",
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
                    label_tensor = torch.tensor(y, dtype=torch.int64, device=device)
                    label_tensor = label_tensor.unsqueeze(0)
                    accuracy.update(y_pred, label_tensor)
                    preds.append(y_pred.item())
                    y_true.append(label_tensor.item())
                    writer.writerow(
                        (x[-1][0], x[-1][1], x[-1][2], x[-1][3], y, y_pred.item())
                    )
                    file.flush()
            file.close()
    else:
        y_true = []
        preds = []
        with torch.no_grad():
            for x, y in tqdm(test_set):
                query_tensor = torch.tensor(x, dtype=torch.float64, device=device)
                query_hypervector = encode(query_tensor)
                output = model(query_hypervector, dot=False)
                y_pred = torch.argmax(output).unsqueeze(0).to(device)
                label_tensor = torch.tensor(y, dtype=torch.int64, device=device)
                label_tensor = label_tensor.unsqueeze(0)
                accuracy.update(y_pred, label_tensor)
                preds.append(y_pred.item())
                y_true.append(label_tensor.item())

    print(f"Testing accuracy of model is {(accuracy.compute().item() * 100):.3f}%")
    f1 = f1_score(y_true, preds, zero_division=0)
    print(f"Testing F1 Score of model is {(f1):.3f}")
    return (accuracy.compute().item() * 100, f1)


# Run a test for a pid, only training using that pid's data
def run_individual_train_and_test_for_pid(pid: str, encoder_option: int):
    # Create Centroid model
    model = models.Centroid(
        DIMENSIONS, NUM_TAC_LEVELS, dtype=torch.float64, device=device
    )
    # Create Encoder module
    if encoder_option == USE_LEVEL_ENCODER:
        encode = HdcLevelEncoder.HdcLevelEncoder(NUM_SIGNAL_LEVELS, WINDOW, DIMENSIONS)
    elif encoder_option == USE_RBF_ENCODER:
        encode = HdcRbfEncoder.HdcRbfEncoder(WINDOW, DIMENSIONS, USE_TANH)
    elif encoder_option == USE_RCN_ENCODER:
        encode = RcnHdcEncoder.RcnHdcEncoder(DIMENSIONS)
    elif encoder_option == USE_SINUSOID_NGRAM_ENCODER:
        encode = HdcSinusoidNgramEncoder.HdcSinusoidNgramEncoder(DIMENSIONS)
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
    options = "e:"

    # Long options
    long_options = ["Encoder="]

    try:
        # Parsing argument
        arguments, values = getopt.getopt(argumentList, options, long_options)
        mode = 0
        # Checking each argument
        for currentArgument, currentValue in arguments:
            if currentArgument in ("-e", "--Encoder"):
                if currentValue == str(0):
                    mode = USE_LEVEL_ENCODER
                elif currentValue == str(1):
                    mode = USE_RBF_ENCODER
                elif currentValue == str(2):
                    mode = USE_RCN_ENCODER
                elif currentValue == str(3):
                    mode = USE_SINUSOID_NGRAM_ENCODER
                else:
                    mode = USE_LEVEL_ENCODER

        print("Single-PID-Tests for %s encoder" % encoder_mode_str(mode))
        # Load datasets in windowed format
        load_accel_data_full()
        load_all_pid_data(mode)

        with open(
            "results/hdc_output_single_%s.csv" % encoder_mode_str(mode), "w", newline=""
        ) as file:
            writer = csv.writer(file)
            for pid in PIDS1 + PIDS2:
                accuracy, f1 = run_individual_train_and_test_for_pid(pid, mode)
                writer.writerow([pid, accuracy, f1])
                file.flush()
            file.close()
        print("All tests done")

    except getopt.error as err:
        # output error, and return with an error code
        print(str(err))
