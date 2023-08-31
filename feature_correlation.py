import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

MFCC_FEAT_LENGTH = 91 * 6

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

def generate_code_stubs(corr : pd.DataFrame):
    print("Correlation matrix shape: ", corr.shape)
    max_tac_corr = np.max([corr.iat[0, f] for f in range(1, len(corr))])
    min_tac_corr = np.min([corr.iat[0, f] for f in range(1, len(corr))])
    max_mfcc_tac_corr = np.max([corr.iat[0, f] for f in range(1, 1 + MFCC_FEAT_LENGTH)])
    print("Max correlation with TAC = %.5f" % max_tac_corr)
    print("Min correlation with TAC = %.5f" % min_tac_corr)
    print("Max correlation of MFCC features with TAC = %.5f" % max_mfcc_tac_corr)

    keep = []
    for f in range(1 + MFCC_FEAT_LENGTH, len(corr)):
        if corr.iat[0, f] > 0:
            keep.append(f)
    print("keep: ", keep)
    
    #Form graph where edges are correlation above threshold
    threshold = 0.9
    g = Graph(len(corr))
    for f_x in keep:
        for f_y in keep:
            if f_x == f_y:
                continue
            c = corr.iat[f_x, f_y]
            if c > threshold:
                g.add_edge(f_x, f_y)
    #Get lists of binded features
    bind_sets = g.connected_components()
    bind_sets = [s for s in bind_sets if s[0] in keep]

    print("Outputting bind/bundle schema to file")
    with open("data/feature_bind_bundle_schema.txt", "w") as file:
        file.write("Kept features: \n")
        for k in keep:
            file.write("%d, " % k)        
        file.write("\nBundled/bind schema (%d sets): \n" % len(bind_sets))
        for s in bind_sets:
            st = str(s) + "\n"
            file.write(st)

    print("Generating code stubs")
    with open("data/feature_code_stubs.txt", "w") as file:
        file.write("chosen_feat = [")
        for k in keep:
            file.write("%d, " % k)
        file.write("]\n")
        file.write("self.feat_kernels = {}\n")
        file.write("for f in chosen_feat:\n")
        file.write("    self.feat_kernels[f] = embeddings.Sinusoid(1, out_dimension, dtype=torch.float64, device=\"cuda\")")
        file.write("\n\n\n")
        s = "feat_hvs = {}\n"
        s2 = "feat_hvs[%d] = self.feat_kernels[%d](feat[%d].unsqueeze(0))\n"
        file.write(s)
        for k in keep:
            file.write(s2 % (k, k, k-1))
        file.write("\n\n\n")
        file.write("(\n")
        for s in bind_sets:
            s3 = "+ ("
            for e in s:
                s3 += "feat_hvs[%d] * " % e
            s3 = s3.rstrip(" * ")
            s3 += ")\n"
            file.write(s3)
        file.write(")")


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
    