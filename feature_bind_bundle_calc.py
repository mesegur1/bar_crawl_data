import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
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
    corr = df.fillna(0).corr(method='spearman')

    return corr

def calculate_avg_similarity():
    global df
    #Find similarity
    sim = cosine_similarity(df.fillna(0).iloc[:].T)
    print(sim.shape)

    return sim

def which_feat_set(i : int):
    if i < MFCC_FEAT_LENGTH:
        #MFCC set
        s = (i - 1) // MFCC_COV_FEAT_LENGTH
    else:
        #Non-MFCC set
        s = (i - MFCC_FEAT_LENGTH - 1) // 3 + MFCC_COV_NUM
    return s

def feat_set_start_index(i : int):
    if i < MFCC_COV_NUM:
        #MFCC set  (0 - 5)
        idx = i * MFCC_COV_FEAT_LENGTH
    else:
        #Non-MFCC set (6 - 23)
        # 6 -> 546
        idx = (i - MFCC_COV_NUM) * 3 + MFCC_FEAT_LENGTH
    return idx

def feat_set_end_index(i : int):
    idx = feat_set_start_index(i) 
    if i < MFCC_COV_NUM:
        idx += MFCC_COV_FEAT_LENGTH
    else:
        idx += 3
    return idx 

def generate_code_stubs(corr : pd.DataFrame, sim : np.ndarray):
    print("Correlation matrix shape: ", corr.shape)
    max_tac_corr = np.max([corr.iat[0, f] for f in range(1, len(corr))])
    min_tac_corr = np.min([corr.iat[0, f] for f in range(1, len(corr))])
    max_mfcc_tac_corr = np.max([corr.iat[0, f] for f in range(1, 1 + MFCC_FEAT_LENGTH)])
    print("Max correlation with TAC = %.5f" % max_tac_corr)
    print("Min correlation with TAC = %.5f" % min_tac_corr)
    print("Max correlation of MFCC features with TAC = %.5f" % max_mfcc_tac_corr)

    print("Similarity matrix shape: ", corr.shape)
    max_tac_sim = np.max([sim[0, f] for f in range(1, len(sim))])
    min_tac_sim = np.min([sim[0, f] for f in range(1, len(sim))])
    max_mfcc_tac_sim = np.max([sim[0, f] for f in range(1, 1 + MFCC_FEAT_LENGTH)])
    print("Max similarity with TAC = %.5f" % max_tac_sim)
    print("Min similarity with TAC = %.5f" % min_tac_sim)
    print("Max similarity of MFCC features with TAC = %.5f" % max_mfcc_tac_sim)

    #Compute correlations between sets
    num_feat_sets = which_feat_set(len(corr))
    set_corr = np.zeros((num_feat_sets, num_feat_sets))
    m1 = np.zeros((num_feat_sets, num_feat_sets))
    for f_x in range(1 , len(corr)):
        for f_y in range(1, len(corr)):
            f_x_set = which_feat_set(f_x)
            f_y_set = which_feat_set(f_y)
            if f_x_set == f_y_set:
                continue
            set_corr[f_x_set, f_y_set] += corr.iat[f_x, f_y]
            m1[f_x_set, f_y_set] += 1
    m1 = np.where(m1 == 0, 1, m1)
    set_corr = set_corr / m1

    #Compute similarities between sets
    set_sim = np.zeros((num_feat_sets, num_feat_sets))
    m2 = np.zeros((num_feat_sets, num_feat_sets))
    for f_x in range(1, len(corr)):
        for f_y in range(1, len(corr)):
            f_x_set = which_feat_set(f_x)
            f_y_set = which_feat_set(f_y)
            if f_x_set == f_y_set:
                continue
            set_sim[f_x_set, f_y_set] += sim[f_x, f_y]
            m2[f_x_set, f_y_set] += 1
    m2 = np.where(m2 == 0, 1, m2)
    set_sim = set_sim / m2
    

    #Choose what features to keep for consideration
    tac_corr = np.zeros((num_feat_sets, 1))
    m3 = np.zeros((num_feat_sets, 1))
    for f in range(1, len(corr)):
        c = corr.iat[0, f]
        f_set = which_feat_set(f)
        tac_corr[f_set] += c
        m3[f_set] += 1
    m3 = np.where(m3 == 0, 1, m3)
    tac_corr = tac_corr / m3
    feat_sets_keep = [i for i in range(num_feat_sets) if tac_corr[i] > 0]

    #Form graph where edges are correlation above threshold
    c_threshold = 0.8
    s_threshold = 0.5
    g_bind = Graph(num_feat_sets)
    g_bundle = Graph(num_feat_sets)
    for f_x in range(num_feat_sets):
        for f_y in range(num_feat_sets):
            if f_x == f_y:
                continue
            c = set_corr[f_x, f_y]
            s = set_sim[f_x, f_y]
            if f_x in feat_sets_keep and f_y in feat_sets_keep:
                if s > s_threshold:
                    g_bundle.add_edge(f_x, f_y)
                if c > c_threshold:
                    g_bind.add_edge(f_x, f_y)
    #Get lists of binded features
    bind_sets = g_bind.connected_components()
    bind_sets = [s for s in bind_sets if all(item in feat_sets_keep for item in s)]
    bundle_sets = g_bundle.connected_components()
    bundle_sets = [s for s in bundle_sets if all(item in feat_sets_keep for item in s)]
    

    print("Outputting bind/bundle schema to file")
    with open("data/feature_bind_bundle_schema.txt", "w") as file: 
        file.write("Feat set correlation with TAC:\n\n")
        for i in range(tac_corr.shape[0]):
            file.write("%d, %.5f\n" % (i, tac_corr[i]))
        file.write("\n\n\n")

        s1 = "        self.feat_kernels[s] = embeddings.Sinusoid(3, out_dimension, dtype=torch.float64, device=\"cuda\")\n"
        s2 = "        self.feat_kernels[s] = embeddings.Sinusoid(%d, out_dimension, dtype=torch.float64, device=\"cuda\")\n" % MFCC_COV_FEAT_LENGTH
        file.write("self.feat_kernels = {}\n")
        file.write("for s in range(%d):\n" % num_feat_sets)
        file.write("    if s < %d:\n" % MFCC_COV_NUM)
        file.write(s2)
        file.write("    else:\n")
        file.write(s1)
        file.write("\n\n\n")
        s = "feat_hvs = {}\n"
        s2 = "feat_hvs[%d] = self.feat_kernels[%d](feat[%d:%d])\n"
        file.write(s)
        for s in range(num_feat_sets):
            file.write(s2 % (s, s, feat_set_start_index(s), feat_set_end_index(s)))
        file.write("\n\n\n")

        file.write("\nBundled/bind schema\n\n")
        file.write("Bind sets:\n")
        for s in bind_sets:
            st = str(s) + "\n"
            file.write(st)
        file.write("Bundle sets:\n")
        for s in bundle_sets:
            st = str(s) + "\n"
            file.write(st)


if __name__ == "__main__":
    corr = None
    try:
        with open("data/correlations.pkl", "rb") as file:
            print("Load precalculated correlation matrix")
            corr = pickle.load(file)
    except:
        pass
    sim = None
    try:
        with open("data/similarities.pkl", "rb") as file:
            print("Load precalculated similarity matrix")
            sim = pickle.load(file)
    except:
        pass
    if corr is None or sim is None:
        for pid in PIDS:
            print("Load data for %s" % pid)
            load_pid_data(pid)
        if corr is None:
            print("Calculate correlation matrix")
            corr = calculate_avg_corr()
            with open("data/correlations.pkl", "wb") as file:
                pickle.dump(corr, file)
        if sim is None:
            print("Calculate similarity matrix")
            sim = calculate_avg_similarity()
            with open("data/similarities.pkl", "wb") as file:
                pickle.dump(sim, file)
    generate_code_stubs(corr, sim)
    