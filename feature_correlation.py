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
    corr = df.corr(method='pearson')

    max_tac_corr = np.max([corr.iat[0, f] for f in range(1, len(corr))])
    print("Max correlation with TAC = %.5f" % max_tac_corr)

    keep = []
    for f in range(1, len(corr)):
        if corr.iat[0, f] > max_tac_corr*0.1:
            keep.append(f)
    print("keep: ", keep)
    
    bind = []
    bundle = []
    no_connect = []

    for f_x in keep:
        for f_y in keep:
            if f_x == f_y:
                continue
            if corr.iat[f_x, f_y] > 0.8:
                if (f_y, f_x) not in bundle:
                    bundle.append((f_x, f_y))
            elif corr.iat[f_x, f_y] >= 0.5:
                if (f_y, f_x) not in bind:
                    bind.append((f_x, f_y))
            else:
                if (f_y, f_x) not in no_connect:
                    no_connect.append((f_x, f_y))

    print("Outputting bind/bundle schema to file")
    with open("data/feature_bind_bundle_schema.txt", "w") as file:
        file.write("Bundled feature pairs: ")
        for t in bundle:
            file.write("(%d, %d), " % t)
        file.write("\nBind feature pairs: ")
        for t in bind:
            file.write("(%d, %d), " % t)
        file.write("\nNot connected feature pairs: ")
        for t in no_connect:
            file.write("(%d, %d), " % t)
        file.close()

if __name__ == "__main__":
    for pid in PIDS:
        print("Load data for %s" % pid)
        load_pid_data(pid)
    print("Calculate correlation for all data, write to file")
    calculate_avg_corr()
    