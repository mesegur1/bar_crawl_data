import numpy as np
import pandas as pd
import csv
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import balanced_accuracy_score
import getopt, sys

if __name__ == "__main__":

    # Remove 1st argument from the
    # list of command line arguments
    argumentList = sys.argv[1:]

    # Options
    options = "f:"

    # Long options
    long_options = ["file="]

    try:
        # Parsing argument
        arguments, values = getopt.getopt(argumentList, options, long_options)
        filename = None
        # Checking each argument
        for currentArgument, currentValue in arguments:
            if currentArgument in ("-f", "--file"):
                filename = currentValue

        with open(filename, "r", newline="") as file:
            reader = csv.reader(file)
            y_true = np.array(next(reader), dtype=int)
            preds = np.array(next(reader), dtype=int)
            precision = precision_score(y_true, preds)
            recall = recall_score(y_true, preds)
            b_acc = balanced_accuracy_score(y_true, preds)

            print("Precision = %f" % precision)
            print("Recall = %f" % recall)
            print("Balanced Accuracy = %f" % b_acc)
    except Exception as err:
        print(str(err))

    
