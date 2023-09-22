import warnings
import numpy as np
import pandas as pd
import torch
import pickle
from sklearn.model_selection import train_test_split
import feature_engineering.eda as eda
import feature_engineering.feature_engineering as fe
import feature_engineering.preprocessing as pre
from tqdm import tqdm

tqdm.pandas()

TAC_LEVEL_0 = 0  # < 0.080 g/dl
TAC_LEVEL_1 = 1  # >= 0.080 g/dl

MS_PER_SEC = 1000

# Data windowing settings
WINDOW = 400  # 10 second window: 10 seconds * 40Hz = 400 samples per window
WINDOW_STEP = 360  # 8 second step: 9 seconds * 40Hz = 360 samples per step
START_OFFSET = 0
END_INDEX = np.inf
TRAINING_EPOCHS = 1
SAMPLE_RATE = 40  # Hz
TEST_RATIO = 0.25
MOTION_EPSILON = 0.0001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a function to convert row to numpy matrix
def convert_to_window(row):
    return np.array([c for c in row]).T

def split_dataset(dataset, seed=1):
    """
    Split dataset into train (70%), test (30%).
    """
    raw_X = dataset[['time', 'x', 'y', 'z']]
    raw_X = raw_X.progress_apply(lambda row: convert_to_window(row), axis=1).to_numpy()
    X = dataset.drop(columns=['pid', 'window10', 'timestamp', 'time', 'x', 'y', 'z', 'intoxicated'], axis=1).to_numpy()
    y = dataset[['intoxicated']].to_numpy().flatten()

    train_raw_X, test_raw_X, train_X, test_X, train_y, test_y = train_test_split(raw_X, X, y, test_size=0.3, random_state=seed)
    train_length = len(train_raw_X)
    test_length = len(test_raw_X)
    train_set = list(zip(train_raw_X, train_X, train_y))
    print("Number of Windows For Training: %d" % (train_length))
    test_set = list(zip(test_raw_X, test_X, test_y))
    print("Number of Windows For Testing: %d" % (test_length))
    return (train_set, test_set)


def load_combined_data(pids: list):
    print("Loading input data...")
    acc_path = "preprocessed_data/"
    with open("%s/merged.pkl" % acc_path, "rb") as file:
        merged = pickle.load(file)
    
    train_set, test_set = split_dataset(merged)
    
    return (WINDOW, train_set, test_set)


if __name__ == "__main__":
    path = "data/"
    new_path = "preprocessed_data/"
    print("Reading in all data")
    #pre.preprocess_acc(path, new_path)
    
    print("Running feature engineering...")
    full_acc, raw_acc = fe.run_feature_engineering(new_path)
    
    print("Read in TAC...")
    tac = pre.preprocess_tac(path + "clean_tac/")
    tac = tac.drop(columns=['TAC_Reading'], axis=1)
    
    # Join target onto features.
    print("Merging dataframes...")
    merged = fe.reconcile_acc_tac(full_acc, raw_acc, tac)
    merged_path = "merged_data/"
    with open("%s/merged.pkl" % new_path, "wb") as file:
        pickle.dump(merged, file)