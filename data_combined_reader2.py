import warnings
import numpy as np
import pandas as pd
import torch
import pickle
from sklearn.model_selection import train_test_split
import feature_engineering.eda as eda
import feature_engineering.feature_engineering as fe
import feature_engineering.preprocessing as pre

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

def split_dataset(dataset, seed=1):
    """
    Split dataset into train (70%), validation (15%), test (15%).
    """
    X = dataset.drop(columns=['pid', 'window10', 'timestamp', 'intoxicated'], axis=1)
    y = dataset[['intoxicated']]
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=seed)
    valid_X, test_X, valid_y, test_y = train_test_split(test_X, test_y, test_size=0.5, random_state=seed)
    return train_X, valid_X, test_X, train_y, valid_y, test_y


def load_combined_data(pids: list):
    print("Loading input data...")
    acc_path = "preprocessed_data/"
    with open("%s/merged.pkl" % acc_path, "rb") as file:
        merged = pickle.load(file)
    
    train_X, valid_X, test_X, train_y, valid_y, test_y = split_dataset(merged)
    
    train_data_set = list(zip(train_X, train_y))
    test_data_set = list(zip(test_X, test_y))
    
    return (WINDOW, train_data_set, test_data_set)


if __name__ == "__main__":
    path = "data/"
    # new_path = "preprocessed_data/"
    # print("Reading in all data")
    # pre.preprocess_acc(path, new_path)
    
    print("Running feature engineering...")
    acc_path = "preprocessed_data/"
    full_acc, raw_acc = fe.run_feature_engineering(acc_path)
    
    print("Read in TAC...")
    tac = pre.preprocess_tac(path + "clean_tac/")
    tac = tac.drop(columns=['TAC_Reading'], axis=1)
    
    # Join target onto features.
    print("Merging dataframes...")
    merged = fe.reconcile_acc_tac(full_acc, raw_acc, tac)
    
    with open("%s/merged.pkl" % acc_path, "wb") as file:
        pickle.dump(merged, file)