# Experiment with barcrawl data
import torch
import numpy as np
import torchhd
import random
from torchhd import embeddings
from torchhd_custom import models  # from torchhd import models
from encoders import HdcLevelEncoder
from encoders import HdcRbfEncoder
from encoders import RcnHdcEncoder
from encoders import HdcSinusoidNgramEncoder
from encoders import HdcGenericEncoder
from data_combined_reader import load_accel_data_full
from data_combined_reader import is_greater_than
from data_combined_reader import load_combined_data
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
TRAINING_EPOCHS = 1

# Data windowing settings
WINDOW = 200  # 10 second window: 10 seconds * 20Hz = 200 samples per window
MOTION_EPSILON = 0.0

# Encoder options
USE_LEVEL_ENCODER = 0
USE_RBF_ENCODER = 1
USE_RCN_ENCODER = 2
USE_SINUSOID_NGRAM_ENCODER = 3
USE_GENERIC_ENCODER = 4


def encoder_mode_str(mode: int):
    if mode == USE_LEVEL_ENCODER:
        return "level"
    elif mode == USE_RBF_ENCODER:
        return "rbf"
    elif mode == USE_RCN_ENCODER:
        return "rcn"
    elif mode == USE_SINUSOID_NGRAM_ENCODER:
        return "sinusoid-ngram"
    elif mode == USE_GENERIC_ENCODER:
        return "generic"
    else:
        return "unknown"


# Option for RBF encoder
USE_TANH = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data_set = []
test_data_set = []
#                            This one  This one also
# PIDS1 = ["BK7610", "MC7070", "MJ8002", "SF3079"]
#
PIDS2 = ["CC6740", "SA0297"]

PIDS1 = [
    "BK7610",
    "BU4707",
    "DC6359",
    "DK3500",
    "HV0618",
    "JB3156",
    "JR8022",
    "MC7070",
    "MJ8002",
    "PC6771",
    "SF3079",
]


# Load all data for each pid
def load_all_pid_data(mode: int):
    global train_data_set
    global test_data_set

    train_data_set, test_data_set = load_combined_data(PIDS1 + PIDS2)


# Run train for a given pid, with provided model and encoder
def run_train(model: models.Centroid, encode: torch.nn.Module, write_file: bool = True):
    # Train using training set half
    print("Begin training")
    if write_file == True:
        with open(
            "data/plot_data/train_data/train_data_downsampled.csv",
            "w",
            newline="",
        ) as file:
            writer = csv.writer(file)
            with torch.no_grad():
                for e in range(0, TRAINING_EPOCHS):
                    print("Training Epoch %d" % (e))
                    for x, f, y in tqdm(train_data_set):
                        if is_greater_than(x[:, 1:], MOTION_EPSILON) == True:
                            input_tensor = torch.tensor(
                                x, dtype=torch.float64, device=device
                            )
                            input_feat_tensor = torch.tensor(
                                f, dtype=torch.float64, device=device
                            )
                            input_hypervector = encode(input_tensor, input_feat_tensor)
                            input_hypervector = input_hypervector.unsqueeze(0)
                            label_tensor = torch.tensor(
                                y, dtype=torch.int64, device=device
                            )
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
                for x, f, y in tqdm(train_data_set):
                    if is_greater_than(x[:, 1:], MOTION_EPSILON) == True:
                        input_tensor = torch.tensor(
                            x, dtype=torch.float64, device=device
                        )
                        input_feat_tensor = torch.tensor(
                            f, dtype=torch.float64, device=device
                        )
                        input_hypervector = encode(input_tensor, input_feat_tensor)
                        input_hypervector = input_hypervector.unsqueeze(0)
                        label_tensor = torch.tensor(y, dtype=torch.int64, device=device)
                        label_tensor = label_tensor.unsqueeze(0)
                        model.add_adjust_iterative(
                            input_hypervector, label_tensor, lr=LEARNING_RATE
                        )


def run_test(model: models.Centroid, encode: torch.nn.Module, write_file: bool = True):
    # Test using test set half
    print("Begin Predicting")
    accuracy = torchmetrics.Accuracy(
        "multiclass",
        num_classes=NUM_TAC_LEVELS,
    )
    accuracy = accuracy.to(device)
    if write_file == True:
        with open(
            "data/plot_data/test_data/test_data_downsampled.csv",
            "w",
            newline="",
        ) as file:
            writer = csv.writer(file)
            y_true = []
            preds = []
            with torch.no_grad():
                for x, f, y in tqdm(test_data_set):
                    if is_greater_than(x[:, 1:], MOTION_EPSILON) == True:
                        query_tensor = torch.tensor(
                            x, dtype=torch.float64, device=device
                        )
                        query_feat_tensor = torch.tensor(
                            f, dtype=torch.float64, device=device
                        )
                        query_hypervector = encode(query_tensor, query_feat_tensor)
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
            for x, f, y in tqdm(test_data_set):
                if is_greater_than(x[:, 1:], MOTION_EPSILON) == True:
                    query_tensor = torch.tensor(x, dtype=torch.float64, device=device)
                    query_feat_tensor = torch.tensor(
                        f, dtype=torch.float64, device=device
                    )
                    query_hypervector = encode(query_tensor, query_feat_tensor)
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


# Run a test
def run_train_and_test(encoder_option: int):
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
    elif encoder_option == USE_GENERIC_ENCODER:
        encode = HdcGenericEncoder.HdcGenericEncoder(NUM_SIGNAL_LEVELS, DIMENSIONS)
    encode = encode.to(device)

    # Run training
    run_train(model, encode)

    print("Normalizing model")
    model.normalize()

    # Run Testing
    accuracy = run_test(model, encode)

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
                    print("RCN encoder is not available for this experiment")
                    exit()
                elif currentValue == str(3):
                    mode = USE_SINUSOID_NGRAM_ENCODER
                elif currentValue == str(4):
                    mode = USE_GENERIC_ENCODER
                else:
                    mode = USE_LEVEL_ENCODER

        print("Multi-PID-Tests for %s encoder" % encoder_mode_str(mode))
        # Load datasets in windowed format
        # load_accel_data_full()
        load_all_pid_data(mode)

        with open(
            "results/hdc_output_combined_%s.csv" % encoder_mode_str(mode),
            "w",
            newline="",
        ) as file:
            writer = csv.writer(file)
            accuracy, f1 = run_train_and_test(mode)
            writer.writerow([accuracy, f1])
            file.flush()
            file.close()
        print("All tests done")

    except getopt.error as err:
        # output error, and return with an error code
        print(str(err))
