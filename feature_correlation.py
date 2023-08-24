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
            for i in range(0, len(f)):
                row[i+1] = f[i]
            #Append
            df = pd.concat([df, pd.DataFrame(row, index=[0])], ignore_index=True)

        #Find correlation
        corr = df.corr(method='pearson')
        print(corr)

        max_tac_corr = np.max([corr.iat[0, f] for f in range(1, len(corr))])
        print("Max correlation with TAC = %.5f" % max_tac_corr)

        keep = []
        for f in range(1, len(corr)):
            if corr.iat[0, f] > max_tac_corr*0.5:
                keep.append(f)
        print(keep)
        
        bind = []
        bundle = []
        no_connect = []

        for f_x in keep:
            for f_y in keep:
                if f_x == f_y:
                    continue
                if corr.iat[f_x, f_y] > 0.8:
                    bundle.append((f_x, f_y))
                elif corr.iat[f_x, f_y] > 0.5:
                    bind.append((f_x, f_y))
                else:
                    no_connect.append((f_x, f_y))

        print("Outputting bind/bundle schema")
        with open("data/%s_feature_bind_bundle_schema.txt" % pid, "w") as file2:
            file2.write("Bundled feature pairs: ")
            for t in bundle:
                file2.write("(%d, %d), " % t)
            file2.write("\nBind feature pairs: ")
            for t in bind:
                file2.write("(%d, %d), " % t)
            file2.write("\nNot connected feature pairs: ")
            for t in no_connect:
                file2.write("(%d, %d), " % t)


            file2.close()
        file.close()

if __name__ == "__main__":
    for pid in PIDS:
        print("Correlation matrix: %s" % pid)
        calculate_avg_corr(pid)