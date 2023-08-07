# Experiment with barcrawl data
import torch
import numpy as np
import torchhd
import random
from torchhd import embeddings
from torchhd_custom import models  # from torchhd import models
from encoders.HdcLevelEncoder import HdcLevelEncoder
from encoders.HdcRbfEncoder import HdcRbfEncoder
from encoders.HdcSinusoidNgramEncoder import HdcSinusoidNgramEncoder
from encoders.HdcGenericEncoder import HdcGenericEncoder
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
DIMENSIONS = 10000
NUM_SIGNAL_LEVELS = 200
NUM_TAC_LEVELS = 2
DEFAULT_LEARNING_RATE = 0.005
DEFAULT_TRAINING_EPOCHS = 1

# Data windowing settings (this is actually read from PKL data files,
# but we provide a default here)
DEFAULT_WINDOW = 400  # 10 second window: 10 seconds * 40Hz = 400 samples per window

# Encoder options
USE_LEVEL_ENCODER = 0
USE_RBF_ENCODER = 1
USE_SINUSOID_NGRAM_ENCODER = 2
USE_GENERIC_ENCODER = 3

# Learning mode options
USE_ADD = 0
USE_ONLINEHD = 1
USE_ADAPTHD = 2
USE_ADJUSTHD = 3
USE_NEURALHD = 4


def encoder_mode_str(mode: int):
    if mode == USE_LEVEL_ENCODER:
        return "level"
    elif mode == USE_RBF_ENCODER:
        return "rbf"
    elif mode == USE_SINUSOID_NGRAM_ENCODER:
        return "sinusoid-ngram"
    elif mode == USE_GENERIC_ENCODER:
        return "generic"
    else:
        return "unknown"


def learning_mode_str(lmode: int):
    if lmode == USE_ADD:
        return "Add"
    elif lmode == USE_ADAPTHD:
        return "AdaptHD"
    elif lmode == USE_ONLINEHD:
        return "OnlineHD"
    elif lmode == USE_ADJUSTHD:
        return "AdjustHD"
    elif lmode == USE_NEURALHD:
        return "NeuralHD"
    else:
        return "Add"


# Option for RBF encoder
USE_BACKPROPAGATION = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data_set = []
test_data_set = []
window = DEFAULT_WINDOW

# PIDS1 = ["BK7610", "MC7070", "MJ8002", "SF3079"]

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
def load_all_pid_data():
    global train_data_set
    global test_data_set
    global window

    window, train_data_set, test_data_set = load_combined_data(PIDS1 + PIDS2)


# Run train for a given pid, with provided model and encoder
def run_train(
    model: models.Centroid,
    encode: torch.nn.Module,
    learning_mode: int,
    train_epochs: int = 1,
    lr: float = DEFAULT_LEARNING_RATE,
    write_file: bool = True,
):
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
                for e in range(0, train_epochs):
                    print("Training Epoch %d" % (e))
                    for x, f, y in tqdm(train_data_set):
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
                        if learning_mode == USE_ADD:
                            model.add(input_hypervector, label_tensor, lr=lr)
                        elif learning_mode == USE_ADAPTHD:
                            model.add_adapt(input_hypervector, label_tensor, lr=lr)
                        elif learning_mode == USE_ONLINEHD:
                            model.add_online(input_hypervector, label_tensor, lr=lr)
                        elif learning_mode == USE_ADJUSTHD:
                            model.add_adjust_iterative(
                                input_hypervector, label_tensor, lr=lr
                            )
                        elif learning_mode == USE_NEURALHD:
                            model.add_neural(input_hypervector, label_tensor, lr=lr)
                        writer.writerow((x[-1][0], x[-1][1], x[-1][2], x[-1][3], y))
                        file.flush()
            file.close()
    else:
        with torch.no_grad():
            for e in range(0, train_epochs):
                print("Training Epoch %d" % (e))
                for x, f, y in tqdm(train_data_set):
                    input_tensor = torch.tensor(x, dtype=torch.float64, device=device)
                    input_feat_tensor = torch.tensor(
                        f, dtype=torch.float64, device=device
                    )
                    input_hypervector = encode(input_tensor, input_feat_tensor)
                    input_hypervector = input_hypervector.unsqueeze(0)
                    label_tensor = torch.tensor(y, dtype=torch.int64, device=device)
                    label_tensor = label_tensor.unsqueeze(0)
                    model.add_adjust_iterative(input_hypervector, label_tensor, lr=lr)


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
                query_tensor = torch.tensor(x, dtype=torch.float64, device=device)
                query_feat_tensor = torch.tensor(f, dtype=torch.float64, device=device)
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
def run_train_and_test(
    encoder_option: int, learning_mode: int, train_epochs: int, lr: float
):
    # Create Centroid model
    model = models.Centroid(
        DIMENSIONS, NUM_TAC_LEVELS, dtype=torch.float64, device=device
    )
    # Create Encoder module
    if encoder_option == USE_LEVEL_ENCODER:
        encode = HdcLevelEncoder(NUM_SIGNAL_LEVELS, window, DIMENSIONS)
    elif encoder_option == USE_RBF_ENCODER:
        encode = HdcRbfEncoder(window, DIMENSIONS, USE_BACKPROPAGATION)
    elif encoder_option == USE_SINUSOID_NGRAM_ENCODER:
        encode = HdcSinusoidNgramEncoder(DIMENSIONS)
    elif encoder_option == USE_GENERIC_ENCODER:
        encode = HdcGenericEncoder(NUM_SIGNAL_LEVELS, DIMENSIONS)
    encode = encode.to(device)

    # Run training
    run_train(model, encode, learning_mode, train_epochs, lr)

    # print("Normalizing model")
    # model.normalize()

    # Run Testing
    accuracy = run_test(model, encode)

    return accuracy


if __name__ == "__main__":
    print("Using {} device".format(device))

    # Remove 1st argument from the
    # list of command line arguments
    argumentList = sys.argv[1:]

    # Options
    options = "e:t:m:l:"

    # Long options
    long_options = ["Encoder=", "Epochs=", "LearningMode=", "LearningRate="]

    try:
        # Parsing argument
        arguments, values = getopt.getopt(argumentList, options, long_options)
        encoder = USE_LEVEL_ENCODER
        train_epochs = DEFAULT_TRAINING_EPOCHS
        lmode = USE_ADD
        lr = DEFAULT_LEARNING_RATE
        # Checking each argument
        for currentArgument, currentValue in arguments:
            if currentArgument in ("-e", "--Encoder"):
                if currentValue == str(USE_LEVEL_ENCODER):
                    encoder = USE_LEVEL_ENCODER
                elif currentValue == str(USE_RBF_ENCODER):
                    encoder = USE_RBF_ENCODER
                elif currentValue == str(USE_SINUSOID_NGRAM_ENCODER):
                    encoder = USE_SINUSOID_NGRAM_ENCODER
                elif currentValue == str(USE_GENERIC_ENCODER):
                    encoder = USE_GENERIC_ENCODER
                else:
                    encoder = USE_LEVEL_ENCODER
            elif currentArgument in ("-t", "--Epochs"):
                if currentValue.isnumeric():
                    train_epochs = int(currentValue)
                    if train_epochs < DEFAULT_TRAINING_EPOCHS:
                        train_epochs = DEFAULT_TRAINING_EPOCHS
            elif currentArgument in ("-m", "--LearningMode"):
                if currentValue == str(USE_ADD):
                    lmode = USE_ADD
                elif currentValue == str(USE_ADAPTHD):
                    lmode = USE_ADAPTHD
                elif currentValue == str(USE_ONLINEHD):
                    lmode = USE_ONLINEHD
                elif currentValue == str(USE_ADJUSTHD):
                    lmode = USE_ADJUSTHD
                elif currentValue == str(USE_NEURALHD):
                    lmode = USE_NEURALHD
                else:
                    lmode = USE_ADD
            elif currentArgument in ("-l", "--LearningRate"):
                try:
                    lr = float(currentValue)
                except ValueError:
                    print("Defaulting learning rate to %.5f" % DEFAULT_LEARNING_RATE)
                    lr = DEFAULT_LEARNING_RATE
                if lr < 0:
                    lr = DEFAULT_LEARNING_RATE

        print(
            "Multi-PID-Tests for %s encoder and %s learning mode, with %d train epochs, learning mode, and lr=%.5f"
            % (encoder_mode_str(encoder), learning_mode_str(lmode), train_epochs, lr)
        )
        # Load datasets in windowed format
        load_all_pid_data()

        with open(
            "results/hdc_output_combined_%s_%s_%d_%.5f.csv"
            % (encoder_mode_str(encoder), learning_mode_str(lmode), train_epochs, lr),
            "w",
            newline="",
        ) as file:
            writer = csv.writer(file)
            accuracy, f1 = run_train_and_test(encoder, lmode, train_epochs, lr)
            writer.writerow([accuracy, f1])
            file.flush()
            file.close()
        print("All tests done")

    except getopt.error as err:
        # output error, and return with an error code
        print(str(err))
