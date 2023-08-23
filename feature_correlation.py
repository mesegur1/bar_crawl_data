import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import seaborn as sb
import matplotlib.pyplot as mp

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

def calculate_avg_corr(pid : str):
    with open("data/%s_random_train_set.pkl" % pid, "rb") as file:
        train_set = pickle.load(file)
        df = pd.DataFrame()
        for _, f, y in tqdm(train_set):
            #Create row of data frame
            row = {}
            row[0] = y #Label
            for i in range(1, len(f)):
                row[i] = f[i]
            #Append
            df = pd.concat([df, pd.DataFrame(row, index=[0])], ignore_index=True)

        #Find correlation
        corr = df.corr(method='pearson')

        # plotting correlation heatmap
        dataplot = sb.heatmap(corr, cmap="YlGnBu", annot=True)
        
        # displaying heatmap
        mp.show()


if __name__ == "__main__":
    for pid in PIDS:
        print("Correlation matrix: %s" % pid)
        calculate_avg_corr(pid)