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

    ###
    # Non-MFCC correlations
    ###

    #Choose what features to keep for consideration
    keep = []
    for f in range(1 + MFCC_FEAT_LENGTH, len(corr)):
        if corr.iat[0, f] > 0:
            keep.append(f)
    print("Non MFCC keep: ", keep)
    #Form graph where edges are correlation above threshold
    num_feat_sets = which_feat_set(len(corr))
    threshold = 0.7
    g = Graph(num_feat_sets)
    for f_x in keep:
        for f_y in keep:
            if f_x == f_y:
                continue
            c = corr.iat[f_x, f_y]
            if c > threshold:
                f_x_set = which_feat_set(f_x)
                f_y_set = which_feat_set(f_y)
                g.add_edge(f_x_set, f_y_set)
    #Get lists of binded features
    bind_sets = g.connected_components()
    #Use only the sets that have keep elements
    #bind_sets = [s for s in bind_sets if which_feat_set(k) == s for k in keep]

    ###
    # MFCC correlations
    ###
    mfcc_keep = []
    for f in range(1, MFCC_FEAT_LENGTH + 1):
        if corr.iat[0, f] > 0:
            mfcc_keep.append(f)
    mfcc_sets = set([which_mfcc_feat_set(f) for f in mfcc_keep])

    mfcc_tac_corr = np.zeros((MFCC_COV_NUM, 1))
    m2 = np.zeros((MFCC_COV_NUM, 1))
    for f in range(1, MFCC_FEAT_LENGTH + 1):
        c = corr.iat[0, f]
        f_set = which_mfcc_feat_set(f)
        mfcc_tac_corr[f_set] += c
        m2[f_set] += 1
    mfcc_tac_corr = mfcc_tac_corr / m2
    
    mfcc_corr = np.zeros((MFCC_COV_NUM, MFCC_COV_NUM))
    m = np.zeros((MFCC_COV_NUM, MFCC_COV_NUM))
    for f_x in range(1, MFCC_FEAT_LENGTH + 1):
        for f_y in range(1, MFCC_FEAT_LENGTH + 1):
            c = corr.iat[f_x, f_y]
            f_x_set = which_mfcc_feat_set(f_x)
            f_y_set = which_mfcc_feat_set(f_y)
            m[f_x_set, f_y_set] += 1
            mfcc_corr[f_x_set, f_y_set] += c
    mfcc_corr = mfcc_corr / m


    # keep = [f for f in range(1, MFCC_FEAT_LENGTH+1)]
    # #Form graph where edges are correlation above threshold
    # threshold = 0.65
    # ratio = 0.8
    # m = np.zeros((MFCC_COV_NUM, MFCC_COV_NUM), dtype=int)
    # for f_x in keep:
    #     for f_y in keep:
    #         if f_x == f_y:
    #             continue
    #         c = corr.iat[f_x, f_y]
    #         if c > threshold:
    #             f_x_set = which_mfcc_feat_set(f_x)
    #             f_y_set = which_mfcc_feat_set(f_y)
    #             m[f_x_set, f_y_set] += 1
    # g = Graph(MFCC_COV_NUM)
    # for x in range(MFCC_COV_NUM):
    #     for y in range(MFCC_COV_NUM):
    #         if x == y:
    #             continue
    #         if m[x, y] > int(ratio*MFCC_COV_FEAT_LENGTH):
    #             g.add_edge(x, y)
    # #Get lists of binded features
    # mfcc_bind_sets = g.connected_components() 
    # 

    mfcc_keep = {}
    for f in range(1, MFCC_FEAT_LENGTH + 1):
        c = corr.iat[0, f]
        if c > 0:
            mfcc_keep[f] = c
    mfcc_feat = sorted(mfcc_keep.items(), key=lambda item : item[1], reverse=True)
    mfcc_feat = sorted(mfcc_feat[0 : 20])
    print(len(mfcc_feat))
    print(mfcc_feat)

    print("mfcc_feat = feat[", end="")
    for f in mfcc_feat:
        print(("%d, " % f[0]), end="")
    print("]")
    

    print("Outputting bind/bundle schema to file")
    with open("data/feature_bind_bundle_schema.txt", "w") as file:
        file.write("Non MFCC kept features: \n")
        for k in keep:
            file.write("%d, " % k)        
        file.write("\nBundled/bind schema (%d sets): \n" % len(bind_sets))
        for s in bind_sets:
            st = str(s) + "\n"
            file.write(st)

    print("Generating code stubs")
    with open("data/feature_code_stubs.txt", "w") as file:
        file.write("NON MFCC FEAT CODE STUBS:\n")

        file.write("self.feat_kernels = {}\n")
        file.write("for s in range(%d):\n" % num_feat_sets)
        file.write("    self.feat_kernels[s] = embeddings.Sinusoid(3, out_dimension, dtype=torch.float64, device=\"cuda\")")
        file.write("\n\n\n")

        s = "feat_hvs = {}\n"
        s2 = "feat_hvs[%d] = self.feat_kernels[%d](feat[%d:%d])\n"
        file.write(s)
        for s in range(num_feat_sets):
            file.write(s2 % (s, s, feat_set_start_index(s), feat_set_start_index(s)+3))
        file.write("\n\n\n")

        file.write("(\n")
        for s in bind_sets:
            s3 = "* ("
            for e in s:
                s3 += "feat_hvs[%d] + " % e
            s3 = s3.rstrip(" + ")
            s3 += ")\n"
            file.write(s3)
        file.write(")\n\n\n")

        file.write("MFCC FEAT CODE STUBS:\n")

        file.write("self.mfcc_feat_kernels = {}\n")
        file.write("for s in range(%d):\n" % MFCC_COV_NUM)
        file.write("    self.mfcc_feat_kernels[s] = embeddings.Sinusoid(MFCC_COV_FEAT_LENGTH, out_dimension, dtype=torch.float64, device=\"cuda\")")
        file.write("\n\n\n")

        s = "mfcc_feat_hvs = {}\n"
        s2 = "mfcc_feat_hvs[%d] = self.mfcc_feat_kernels[%d](feat[%d:%d])\n"
        file.write(s)
        for s in range(MFCC_COV_NUM):
            file.write(s2 % (s, s, mfcc_feat_set_start_index(s), mfcc_feat_set_start_index(s)+MFCC_COV_FEAT_LENGTH))
        file.write("\n\n\n")

        # file.write("(\n")
        # for s in mfcc_bind_sets:
        #     s3 = "* ("
        #     for e in s:
        #         s3 += "mfcc_feat_hvs[%d] + " % e
        #     s3 = s3.rstrip(" + ")
        #     s3 += ")\n"
        #     file.write(s3)
        # file.write(")")
        file.write("Relevant sets:")
        file.write(str(mfcc_sets))
        file.write("\n\n\n")
        file.write("MFCC set correlation with TAC:\n\n")
        file.write(np.array2string(mfcc_tac_corr))
        file.write("\n\n\n")
        file.write("MFCC set correlation matrix:\n\n")
        file.write(np.array2string(mfcc_corr))
        file.write("\n\n\n")

        file.write("mfcc_feat = feat[[")
        for f in mfcc_feat:
            file.write(("%d, " % f[0]))
        file.write("]]\n")




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
    