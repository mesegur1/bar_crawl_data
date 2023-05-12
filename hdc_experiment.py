# Experiment with barcrawl data
import torch
import numpy as np
import torchhd
from torchhd import embeddings
from torchhd import models
from data_reader import load_data
import torchmetrics
import csv

DIMENSIONS = 6000
NUM_CHANNELS = 3
NUM_SIGNAL_LEVELS = 100
NUM_TAC_LEVELS = 2
WINDOW = 100
WINDOW_STEP = 80
LEARNING_RATE = 0.5
START_OFFSET = 0
END_INDEX = 12000000
TRAINING_EPOCHS = 3
SAMPLE_RATE = 20
TEST_RATIO = 0.25


# Encoder for Bar Crawl Data
class Encoder(torch.nn.Module):
    def __init__(self, levels: int, timestamps: int, out_dimension: int):
        super(Encoder, self).__init__()

        self.signal_level_x = embeddings.Level(
            levels, out_dimension, dtype=torch.float64
        )
        self.signal_level_y = embeddings.Level(
            levels, out_dimension, dtype=torch.float64
        )
        self.signal_level_z = embeddings.Level(
            levels, out_dimension, dtype=torch.float64
        )
        self.channel_basis = embeddings.Random(
            NUM_CHANNELS, out_dimension, dtype=torch.float64
        )
        self.timestamps = embeddings.Thermometer(
            timestamps, out_dimension, dtype=torch.float64
        )

    # Encode window of feature vectors (x,y,z)
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Get level hypervectors for x, y, z samples
        x_levels = self.signal_level_x(input[:, 1])
        y_levels = self.signal_level_y(input[:, 2])
        z_levels = self.signal_level_z(input[:, 3])
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
        sample_hv = torch.tanh(sample_hv)
        return sample_hv


# Run a test for a pid
def run_test_for_pid(pid: str):
    # Load data from CSVs
    train_set, test_set = load_data(
        pid, END_INDEX, START_OFFSET, WINDOW, WINDOW_STEP, SAMPLE_RATE, TEST_RATIO
    )
    # Create Centroid model
    model = models.Centroid(DIMENSIONS, NUM_TAC_LEVELS, dtype=torch.float64)
    # Create Encoder module
    encode = Encoder(NUM_SIGNAL_LEVELS, WINDOW, DIMENSIONS)

    # Train using training set half
    print(
        "Begin training with length %d windows, with %d overlap"
        % (WINDOW, WINDOW - WINDOW_STEP)
    )
    with torch.no_grad():
        for e in range(0, TRAINING_EPOCHS):
            print("Training Epoch %d" % (e))
            for x, y in train_set:
                input_tensor = torch.tensor(x, dtype=torch.float64)
                input_hypervector = encode(input_tensor)
                input_hypervector = input_hypervector.unsqueeze(0)
                label_tensor = torch.round(
                    torch.tensor(y, dtype=torch.float64).mean()
                ).long()
                label_tensor = label_tensor.unsqueeze(0)
                model.add_online(input_hypervector, label_tensor, lr=LEARNING_RATE)

    # Normalize model
    print("Normalizing model")
    model.normalize()

    # Test using test set half
    print("Begin Predicting.")
    accuracy = torchmetrics.Accuracy("multiclass", num_classes=NUM_TAC_LEVELS)
    with torch.no_grad():
        for x, y in test_set:
            query_tensor = torch.tensor(x, dtype=torch.float64)
            query_tensor = query_tensor.unsqueeze(0)
            query_hypervector = encode(query_tensor)
            output = model(query_hypervector, dot=False)
            y_pred = torch.argmax(output).unsqueeze(0)
            accuracy.update(y_pred, torch.tensor(y, dtype=torch.int64).unsqueeze(0))

    print(f"Testing accuracy of model is {(accuracy.compute().item() * 100):.3f}%")
    return accuracy.compute().item() * 100


if __name__ == "__main__":
    pids = [
        # "BK7610",
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

    with open("hdc_output.csv", "w", newline="") as file:
        writer = csv.writer(file)
        for pid in pids:
            accuracy = run_test_for_pid(pid)
            writer.writerow([pid, accuracy])
        file.close()
    print("All tests done")
