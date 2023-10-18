import numpy as np
import pandas as pd
import csv
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
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
            y_preds = np.array(next(reader), dtype=int)
            preds = []
            for i in range(len(y_true)):
                preds.append(np.array(next(reader)))
            preds = np.array(preds)

            c_matrix = confusion_matrix(y_true, y_preds)
            t_pos = c_matrix[1][1]
            t_neg = c_matrix[0][0]
            f_pos = c_matrix[0][1]
            f_neg = c_matrix[1][0]

            precision = precision_score(y_true, y_preds)
            recall = recall_score(y_true, y_preds)
            b_acc = balanced_accuracy_score(y_true, y_preds)

            sober_data = [d for d in list(zip(y_true, y_preds)) if d[0] == 0]
            sober_true = np.array([d[0] for d in sober_data])
            sober_pred = np.array([d[1] for d in sober_data])
            sober_acc = accuracy_score(sober_true, sober_pred)

            drunk_data = [d for d in list(zip(y_true, y_preds)) if d[0] == 1]
            drunk_true = np.array([d[0] for d in drunk_data])
            drunk_pred = np.array([d[1] for d in drunk_data])
            drunk_acc = accuracy_score(drunk_true, drunk_pred)


            print("Precision = %f" % precision)
            print("Recall = %f" % recall)
            print("Balanced Accuracy = %f" % b_acc)
            print("Sober Accuracy = %f" % sober_acc)
            print("Drunk Accuracy = %f" % drunk_acc)
            print("False negatives = %d" % f_neg)
            print("False postives = %d" % f_pos)
            print("True negatives = %d" % t_neg)
            print("True positives = %d" % t_pos)
    except Exception as err:
        print(str(err))

    
