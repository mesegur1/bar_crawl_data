# Experiment with barcrawl data
import torch
import numpy as np
import torchhd
from torchhd import embeddings
import pandas as pd
from torchhd_custom import models  # from torchhd import models
from encoders import HdcLevelEncoder
from encoders import HdcRbfEncoder
from encoders import RcnHdcEncoder
from data_reader import load_train_test_data
import torchmetrics
from sklearn.metrics import f1_score
from tqdm import tqdm
import csv
import getopt, sys

# Hyperparameters
# Changing these affects performance up or down depending on PID
DIMENSIONS = 6000
NUM_SIGNAL_LEVELS = 100
NUM_TAC_LEVELS = 2
LEARNING_RATE = 0.005

# Data windowing settings
WINDOW = 100  # 5 second window: 5 seconds * 20Hz = 100 samples per window
WINDOW_STEP = 50
START_OFFSET = 0
END_INDEX = -1
TRAINING_EPOCHS = 1
SAMPLE_RATE = 20  # Hz
RCN_SAMPLE_RATE = 5  # Hz
TEST_RATIO = 0.30

# Encoder options
USE_LEVEL_ENCODER = 0
USE_RBF_ENCODER = 1
USE_RCN_ENCODER = 2


def encoder_mode_str(mode: int):
    if mode == USE_LEVEL_ENCODER:
        return "level"
    elif mode == USE_RBF_ENCODER:
        return "rbf"
    elif mode == USE_RCN_ENCODER:
        return "rcn"
    else:
        return "unknown"


# Option for RBF encoder
USE_TANH = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_PIDS = [
    "BK7610",
    "BU4707",
    "CC6740",
    "DC6359",
    "DK3500",
    "HV0618",
    "JB3156",
    "JR8022",
]

TEST_PIDS = [
    "MC7070",
    "MJ8002",
    "PC6771",
    "SA0297",
    "SF3079",
]

train_feature_set = np.array([])
train_raw_set = np.array([])
train_labels = np.array([])
test_feature_set = np.array([])
test_raw_set = np.array([])
test_labels = np.array([])
num_features = 1


# Load all data for each pid
def load_all_pid_data(mode: int):
    global train_feature_set
    global train_raw_set
    global train_labels
    global test_feature_set
    global test_raw_set
    global test_labels
    global num_features

    (
        num_features,
        train_feature_set,
        train_raw_set,
        train_labels,
        test_feature_set,
        test_raw_set,
        test_labels,
    ) = load_train_test_data()
    print("Num of features = %d" % num_features)


# Run train with provided model and encoder
def run_train(model: models.Centroid, encode: torch.nn.Module):
    # Train using training set half
    with torch.no_grad():
        for e in range(0, TRAINING_EPOCHS):
            print("Training Epoch %d" % (e))
            for r_x, f_x, y in tqdm(zip(train_raw_set, train_feature_set, train_labels)):
                r_input_tensor = torch.tensor(r_x, dtype=torch.float64, device=device)
                r_input_hypervector = encode(r_input_tensor)
                r_input_hypervector = r_input_hypervector.unsqueeze(0)
                label_tensor = torch.tensor(y, dtype=torch.int64, device=device)
                label_tensor = label_tensor.unsqueeze(0)
                model.add_adjust_iterative(
                    r_input_hypervector, label_tensor, lr=LEARNING_RATE
                )


def run_test(model: models.Centroid, encode: torch.nn.Module):
    # Test using test set half
    print("Begin Predicting")
    accuracy = torchmetrics.Accuracy(
        "multiclass",
        num_classes=NUM_TAC_LEVELS,
    )
    accuracy = accuracy.to(device)
    y_true = []
    preds = []
    with torch.no_grad():
        for r_x, f_x, y in tqdm(zip(test_raw_set, test_feature_set, test_labels)):
            r_query_tensor = torch.tensor(r_x, dtype=torch.float64, device=device)
            query_hypervector = encode(r_query_tensor)
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


def run_train_and_test(encoder_option: int):
    global num_features

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
    encode = encode.to(device)

    # Run training
    run_train(model, encode)

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
        mode = USE_LEVEL_ENCODER
        # Checking each argument
        for currentArgument, currentValue in arguments:
            if currentArgument in ("-e", "--Encoder"):
                if currentValue == str(0):
                    mode = USE_LEVEL_ENCODER
                elif currentValue == str(1):
                    mode = USE_RBF_ENCODER
                elif currentValue == str(2):
                    mode = USE_RCN_ENCODER
                else:
                    mode = USE_LEVEL_ENCODER

        print("Test with %s encoder" % encoder_mode_str(mode))
        # Load datasets in windowed format
        load_all_pid_data(mode)

        with open(
            "results/hdc_output_%s.csv" % encoder_mode_str(mode), "w", newline=""
        ) as file:
            writer = csv.writer(file)
            writer.writerow(["Train PIDs", TRAIN_PIDS])
            writer.writerow(["Test PIDs", TEST_PIDS])
            accuracy, f1 = run_train_and_test(mode)
            writer.writerow(["Accuracy and F1 Score", accuracy, f1])
            file.flush()
            file.close()
        print("All tests done")

    except getopt.error as err:
        # output error, and return with an error code
        print(str(err))
