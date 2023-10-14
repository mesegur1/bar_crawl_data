import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

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
    with open("data/%s_random_data_set.pkl" % pid, "rb") as file:
        data_set = pickle.load(file)
        for _, f, y in tqdm(data_set):
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
    #These features are not offset by added TAC column
    important_feat = [558, 582, 554, 552, 93, 555, 580, 571, 574, 578, 566, 287, 556, 550, 14, 551, 64, 581]
    print("Important features used: ", important_feat)
    print("Correlation matrix shape: ", corr.shape)
    max_tac_corr = np.max([corr.iat[0, f+1] for f in important_feat])
    min_tac_corr = np.min([corr.iat[0, f+1] for f in important_feat])
    print("Max correlation with TAC = %.5f" % max_tac_corr)
    print("Min correlation with TAC = %.5f" % min_tac_corr)

    threshold = 0.6
    g = Graph(len(corr)-1) #Subtract added TAC column
    for f_x in important_feat:
        for f_y in important_feat:
            if f_x == f_y:
                continue
            c = corr.iat[f_x + 1, f_y + 1] #Offset by 1 in corr matrix
            if c > threshold:
                g.add_edge(f_x, f_y)
    #Get lists of binded features
    corr_sets = g.connected_components()
    corr_sets = [s for s in corr_sets if s[0] in important_feat]

    print("Outputting correlated feature groups to file")
    with open("data/feature_corr_schema.txt", "w") as file:
        file.write("Kept features: \n")
        for k in important_feat:
            file.write("%d, " % k)        
        file.write("\nCorr schema (%d sets): \n" % len(corr_sets))
        for s in corr_sets:
            st = str(s) + "\n"
            file.write(st)

        file.write("\n\n\n")
        for s in corr_sets:
            s3 = "* ("
            for e in s:
                s3 += "feat_hvs[%d] + " % e
            s3 = s3.rstrip(" + ")
            s3 += ")\n"
            file.write(s3)
        file.write(")\n\n\n")


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