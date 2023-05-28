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
NUM_RCN_NODES = 100
NUM_TAC_LEVELS = 2
LEARNING_RATE = 0.035

# Data windowing settings
WINDOW = 200
WINDOW_STEP = 190
START_OFFSET = 0
END_INDEX = 1200000
TRAINING_EPOCHS = 1
SAMPLE_RATE = 20
TEST_RATIO = 0.30

my_device = torch_device("cuda" if torch.cuda.is_available() else "cpu")

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
class RcnHdcEncoder(torch.nn.Module):
    def __init__(self, timestamps: int, out_dimension: int):
        super(RcnHdcEncoder, self).__init__()
        self.nodes = NUM_RCN_NODES
        self.hps = {
            "n_nodes": self.nodes,
            "n_inputs": NUM_CHANNELS,
            "n_outputs": NUM_CHANNELS,
            "connectivity": 0.2,
            "spectral_radius": 1.2,
            "regularization": 1.5,
            "leaking_rate": 0.01,
            "bias": 1.4,
        }
        self.rcn = RcNetwork(**self.hps, feedback=True)
        self.x_basis = self.generate_basis(self.nodes + NUM_CHANNELS, out_dimension)
        self.y_basis = self.generate_basis(self.nodes + NUM_CHANNELS, out_dimension)
        self.z_basis = self.generate_basis(self.nodes + NUM_CHANNELS, out_dimension)

        self.channel_basis = embeddings.Random(
            NUM_CHANNELS, out_dimension, device=my_device
        )
        self.timestamps = embeddings.Thermometer(
            timestamps,
            out_dimension,
            low=0,
            high=timestamps,
            device=my_device,
        )

    # Generate n x d matrix with orthogonal rows
    def generate_basis(self, features: int, dimension: int):
        # Generate random projection n x d matrix M using chosen probability distribution
        # Hyperdimensionality causes quasi-orthogonality
        M = np.random.normal(0, 1, (features, dimension))
        # return n x d matrix as a tensor
        return torch.tensor(M, device=my_device)

    # Encode window of feature vectors (x,y,z)
    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        # Feature extraction from x, y, z samples
        self.rcn.fit(X=signals[:-1, 1:], y=signals[1:, 1:])
        x_hvs = torch.matmul(
            self.rcn.LinOut.weight.data[0].float(), self.x_basis.float()
        )
        y_hvs = torch.matmul(
            self.rcn.LinOut.weight.data[1].float(), self.y_basis.float()
        )
        z_hvs = torch.matmul(
            self.rcn.LinOut.weight.data[2].float(), self.z_basis.float()
        )
        # Get time hypervectors
        times = self.timestamps(signals[:, 0])
        # Bind time sequence for x, y, z sample hypervectors
        x_hypervector = torchhd.multiset(torchhd.bind(x_hvs, times))
        y_hypervector = torchhd.multiset(torchhd.bind(y_hvs, times))
        z_hypervector = torchhd.multiset(torchhd.bind(z_hvs, times))
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


# Run train for a given pid, with provided model and encoder
def run_train_for_pid(pid: str, model: models.Centroid, encode: RcnHdcEncoder):
    train_set, _ = pid_data_sets[pid]

    # Train using training set half
    print("Begin training with pid %s data" % pid)
    with torch.no_grad():
        for e in range(0, TRAINING_EPOCHS):
            print("Training Epoch %d" % (e))
            for x, y in tqdm(train_set):
                input_tensor = torch.tensor(x, dtype=torch.float32, device=my_device)
                input_hypervector = encode(input_tensor)
                input_hypervector = input_hypervector.unsqueeze(0)
                label_tensor = torch.tensor(y[-1], dtype=torch.int64, device=my_device)
                label_tensor = label_tensor.unsqueeze(0)
                model.add_online(input_hypervector, label_tensor, lr=LEARNING_RATE)


def run_test_for_pid(pid: str, model: models.Centroid, encode: RcnHdcEncoder):
    _, test_set = pid_data_sets[pid]

    # Test using test set half
    print("Begin Predicting for pid %s" % pid)
    accuracy = torchmetrics.Accuracy(
        "multiclass",
        num_classes=NUM_TAC_LEVELS,
    )
    accuracy = accuracy.to(my_device)
    with torch.no_grad():
        for x, y in tqdm(test_set):
            query_tensor = torch.tensor(x, dtype=torch.float32, device=my_device)
            query_hypervector = encode(query_tensor)
            output = model(query_hypervector, dot=False)
            y_pred = torch.argmax(output).unsqueeze(0).to(my_device)
            label_tensor = torch.tensor(y[-1], dtype=torch.int64, device=my_device)
            label_tensor = label_tensor.unsqueeze(0)
            accuracy.update(y_pred, label_tensor)

    print(f"Testing accuracy of model is {(accuracy.compute().item() * 100):.3f}%")
    return accuracy.compute().item() * 100


# Run a test for a pid, only training using that pid's data
def run_individual_train_and_test_for_pid(pid: str):
    # Create Centroid model
    model = models.Centroid(DIMENSIONS, NUM_TAC_LEVELS, device=my_device)
    # Create Encoder module
    encode = RcnHdcEncoder(WINDOW, DIMENSIONS)
    encode = encode.to(my_device)

    # Run training
    run_train_for_pid(pid, model, encode)

    # Run Testing
    accuracy = run_test_for_pid(pid, model, encode)

    return accuracy


if __name__ == "__main__":
    print("Using {} device".format(my_device))

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

            with open("rcn_hdc_output_single.csv", "w", newline="") as file:
                writer = csv.writer(file)
                for pid in PIDS:
                    accuracy = run_individual_train_and_test_for_pid(pid)
                    writer.writerow([pid, accuracy])
                    file.flush()
                file.close()
            print("All tests done")
        elif mode == 1:
            print("Combined-Train, Combined-Test Mode")
            # Load datasets in windowed format
            load_accel_data_full()
            load_all_pid_data()

            # Create common Centroid model
            model = models.Centroid(DIMENSIONS, NUM_TAC_LEVELS, device=my_device)
            # Create common HDCEncoder module
            encode = RcnHdcEncoder(WINDOW, DIMENSIONS)
            encode = encode.to(my_device)

            with open("rcn_hdc_output_combined.csv", "w", newline="") as file:
                writer = csv.writer(file)
                for pid in PIDS:
                    run_train_for_pid(pid, model, encode)
                for pid in PIDS:
                    accuracy = run_test_for_pid(pid, model, encode)
                    writer.writerow([pid, accuracy])
                    file.flush()
                file.close()
            print("All tests done")

    except getopt.error as err:
        # output error, and return with an error code
        print(str(err))
