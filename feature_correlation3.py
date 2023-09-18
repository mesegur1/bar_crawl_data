import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

MFCC_COV_FEAT_LENGTH = 91
MFCC_COV_NUM = 6
MFCC_FEAT_LENGTH = MFCC_COV_FEAT_LENGTH * MFCC_COV_NUM

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

df = pd.DataFrame()

class Graph:
    # init function to declare class variables
    def __init__(self, V):
        self.V = V
        self.adj = [[] for i in range(V)]
 
    def dfs_util(self, temp, v, visited):
        # Mark the current vertex as visited
        visited[v] = True
        # Store the vertex to list
        temp.append(v)
        # Repeat for all vertices adjacent
        # to this vertex v
        for i in self.adj[v]:
            if visited[i] == False:
                # Update the list
                temp = self.dfs_util(temp, i, visited)
        return temp
 
    # method to add an undirected edge
    def add_edge(self, v, w):
        self.adj[v].append(w)
        self.adj[w].append(v)
 
    # Method to retrieve connected components
    # in an undirected graph
    def connected_components(self):
        visited = []
        cc = []
        for i in range(self.V):
            visited.append(False)
        for v in range(self.V):
            if visited[v] == False:
                temp = []
                cc.append(self.dfs_util(temp, v, visited))
        return cc

def load_pid_data(pid : str):
    global df
    with open("data/%s_random_train_set.pkl" % pid, "rb") as file:
        train_set = pickle.load(file)
        for _, f, y in tqdm(train_set):
            #Create row of data frame
            row = {}
            row[0] = y #Label
            for i in range(0, len(f)):
                row[i+1] = f[i]
            #Append
            df = pd.concat([df, pd.DataFrame(row, index=[0])], ignore_index=True)
        file.close()

def calculate_avg_corr():
    global df
    #Find correlation
    corr = df.corr(method='spearman')

    return corr

def which_feat_set(i : int):
    return (i - MFCC_FEAT_LENGTH - 1) // 3

def feat_set_start_index(i : int):
    return i*3 + MFCC_FEAT_LENGTH

def which_mfcc_feat_set(i : int):
    return (i - 1) // MFCC_COV_FEAT_LENGTH

def mfcc_feat_set_start_index(i : int):
    return i * MFCC_COV_FEAT_LENGTH

def generate_code_stubs(corr : pd.DataFrame):
    print("Correlation matrix shape: ", corr.shape)
    max_tac_corr = np.max([corr.iat[0, f] for f in range(1, len(corr))])
    min_tac_corr = np.min([corr.iat[0, f] for f in range(1, len(corr))])
    max_mfcc_tac_corr = np.max([corr.iat[0, f] for f in range(1, 1 + MFCC_FEAT_LENGTH)])
    print("Max correlation with TAC = %.5f" % max_tac_corr)
    print("Min correlation with TAC = %.5f" % min_tac_corr)
    print("Max correlation of MFCC features with TAC = %.5f" % max_mfcc_tac_corr)

    #Choose what features to keep for consideration
    keep = {}
    for f in range(1, len(corr)):
        c = corr.iat[0, f]
        if c > 0:
            keep[f] = c
    feat = sorted(keep.items(), key=lambda item : item[1], reverse=True)
    feat = sorted(feat)


    print("Generating code stubs")
    with open("data/feature_code_stubs.txt", "w") as file:
        file.write("self.feat_levels = embeddings.Level(200, out_dimension, dtype=torch.float64)")
        file.write("\n\n\n")

        file.write("corr_feat = feat[[")
        for f in feat:
            file.write(("%d, " % f[0]))
        file.write("]]\n")
        s = "feat_hvs = self.feat_levels(corr_feat)\n"
        file.write(s)
        file.write("\n\n\n")


if __name__ == "__main__":
    corr = None
    try:
        with open("data/correlations.pkl", "rb") as file:
            print("Load precalculated correlation matrix")
            corr = pickle.load(file)
    except:
        pass
    if corr is None:
        for pid in PIDS:
            print("Load data for %s" % pid)
            load_pid_data(pid)
        print("Calculate correlation matrix")
        corr = calculate_avg_corr()
        with open("data/correlations.pkl", "wb") as file2:
            pickle.dump(corr, file2)
    generate_code_stubs(corr)
    