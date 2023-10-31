import torch
import numpy as np
from autogluon.tabular import TabularDataset, TabularPredictor
import torchhd
import random
from data_combined_reader import load_combined_data
from tqdm import tqdm
import csv
import getopt, sys
import pandas as pd
import datetime

# Data windowing settings (this is actually read from PKL data files,
# but we provide a default here)
DEFAULT_WINDOW = 400  # 10 second window: 10 seconds * 40Hz = 400 samples per window

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

def calc_feat_importance():
    label = "TAC"
    ldf = pd.DataFrame(columns=[label])

    print("Unzipping training data")
    _, feat_list, label_list = list(zip(*train_data_set))
    print("Converting feature vector list to Numpy array")
    feat_matrix = np.array(feat_list)
    print("Converting feature numpy array to dataframe")
    df = pd.DataFrame(feat_matrix, columns=range(600))
    df.columns = ['Feat ' + str(i+1) for i in range(len(df.columns))]
    print("Create label dataframe")
    ldf['TAC'] = label_list

    print("Merge two dataframes")
    df = pd.concat([df, ldf], axis=1)


    print("Fitting predictor")
    predictor = TabularPredictor(label=label).fit(train_data=df, presets='best_quality')
    print("Calculating feature importance")
    feat_importance = predictor.feature_importance(data=df)
    print("output to CSV")
    feat_importance.to_csv("data/feat_importance.csv")
    return feat_importance
    
    

if __name__ == "__main__":
    print("Using {} device".format(device))
    torch.set_default_tensor_type(torch.DoubleTensor)

    print("Start time: ", datetime.datetime.now().strftime("%H:%M:%S"))
    imp = None
    try:
        imp = pd.read_csv("data/feat_importance.csv")
    except:
        pass
    if imp is None:
        # Load datasets in windowed format
        load_all_pid_data()
        imp = calc_feat_importance()
    
    feat = imp.iloc[:, 0].values
    feat = [int(x.split()[1])-1 for x in feat]
    wo_mfcc = [x for x in feat if x >= 546]
    top_120_wo_mfcc = wo_mfcc[:120]
    top_120 = imp.iloc[:120, 0].values
    top_120 = [int(x.split()[1])-1 for x in top_120]
    print("Top 120 w MFCC", top_120)
    print("Top 120 w/o MFCC", top_120_wo_mfcc)
    print("Length:", len(top_120))