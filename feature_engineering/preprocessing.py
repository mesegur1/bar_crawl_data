import os
import numpy as np
import pandas as pd
from tqdm import tqdm


def missing_data_imputation(df):
    """
    Standardize: generate max_timestamp (ms) - min_timestamp (ms) 
    number of rows for a given df, so that there is a row for 
    every millisecond. 

    Impute: Fill in missing accelerometer readings with readings from the 
    previous millisecond timestamp.

    Returns: array of timestamps and array of accelerometer readings.
    """
    min_timestamp = df['time'].min()
    max_timestamp = df['time'].max()
    min_timeinterval = 1

    print(min_timestamp, max_timestamp)

    array_size = int((max_timestamp-min_timestamp)/min_timeinterval) + 1
    # Initialize empty array of size array_size.
    accelerometer_readings = [None] * array_size

    # Add data to the arrays based on the readings.
    first_accelerometer_reading = None
    for i in range(0, len(df)):
        if(first_accelerometer_reading == None):
            first_accelerometer_reading = [
                df.loc[i, 'x'], df.loc[i, 'y'], df.loc[i, 'z']]
        index = int((df.loc[i, 'time'] - min_timestamp)/min_timeinterval)
        try:
            accelerometer_readings[index] = [
                df.loc[i, 'x'], df.loc[i, 'y'], df.loc[i, 'z']]
        except:  # If sensor readings are empty. -- Erroroneous.
            pass

    prev_accelerometer = None
    for i in range(0, array_size):
        # If missing, add reading from previous timestamp.
        if(accelerometer_readings[i] == None):
            if(prev_accelerometer != None):
                accelerometer_readings[i] = prev_accelerometer
            else:
                accelerometer_readings[i] = first_accelerometer_reading
        # If not missing, skip row and do not override it.
        elif (accelerometer_readings[i] != None):
            prev_accelerometer = accelerometer_readings[i]

    return list(range(min_timestamp, max_timestamp+1)), accelerometer_readings


def preprocess_acc(path, new_path):
    """
    Given a path load "all_accelerometer_data_pids_13.csv" 
    and drop if missing time or zero accelerometer data.

    For each pid, standardize sampling frequency to every millisecond, 
    impute missing data and save a pickle file for each pid to new_path.

    This function returns None.
    """
    # Create folder for new_path if does not exist.
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    # Load file.
    df = pd.read_csv(path + "all_accelerometer_data_pids_13.csv")
    # Drop if missing timestamp.
    df = df.loc[df.time!=0]
    # Keep non-zero accelerometer data only.
    df = df.loc[(df.x!=0) & (df.y!=0) & (df.z!=0)]
    # For each pid, standardize sampling frequency to 1 millisecond.
    for current_pid in tqdm(list(df.pid.unique())):
        print(f"Preprocessing: {current_pid}")
        # Filter on pid.
        temp = df.loc[df.pid == current_pid].sort_values(
            'time', ascending=True).reset_index(drop=True)
        print(f"Original shape: {temp.shape}")
        timestamps, readings = missing_data_imputation(temp)
         # Create df with timestamps and readings.
        new_df = pd.DataFrame(readings, columns=["x", "y", "z"]).astype('float32')
        new_df['time'] = timestamps
        new_df['pid'] = current_pid
	    # Print new df shape.
        print(f"New shape: {new_df.shape}")
	    # Export preproccessed data as a pickle file.
        new_df.to_pickle(new_path + current_pid + 
	    				"_preprocessed_acc.pkl")
    print("Preprocessing complete and files exported.")
    return None


def preprocess_tac(path):
    """
    Given a path to the "clean_tac" folder,
    append all tac files in directory and create a pid variable.
    
    Convert "TAC_Reading" into binary "intoxicated" variable:
    intoxicated = 1 if TAC_Reading > 0.08,
    intoxicated = 0 if TAC_Reading <= 0.08.

	Returns concatenated dataframe with all pids.
    """
    appended_data = []
    directory = os.fsencode(path)
    for file in tqdm(os.listdir(directory)):
        filename = os.fsdecode(file)
        df = pd.read_csv(path + filename)
        df['pid'] = filename.split("_")[0]
        appended_data.append(df)
    df = pd.concat(appended_data).sort_values(['timestamp'], ascending=True).reset_index(drop=True)
    # Create binary flag.
    df.loc[df.TAC_Reading > 0.08, "intoxicated"] = 1
    df.loc[df.TAC_Reading <= 0.08, "intoxicated"] = 0
    return df
