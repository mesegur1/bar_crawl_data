import sys
import struct
import numpy as np
import sklearn
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.signal import find_peaks

TAC_LEVEL_0 = 0  # < 0.080 g/dl
TAC_LEVEL_1 = 1  # >= 0.080 g/dl

MS_PER_SEC = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


# Convert TAC measurement to a class
def tac_to_class(tac: float):
    if tac < 0:
        tac = 0
    tac = round(tac, 3) * 1000
    if tac < 80:
        return TAC_LEVEL_0
    else:
        return TAC_LEVEL_1


accel_data_full = pd.DataFrame([])  # = []


# Load in accelerometer data into memory
def load_accel_data_full():
    global accel_data_full
    print("Read in accelerometer data")
    accel_data_full = pd.read_csv("data/all_accelerometer_data_pids_13.csv")
    accel_data_full["time"] = accel_data_full["time"].astype("datetime64[ms]")
    accel_data_full["pid"] = accel_data_full["pid"].astype(str)
    accel_data_full["x"] = accel_data_full["x"].astype("float64")
    accel_data_full["y"] = accel_data_full["y"].astype("float64")
    accel_data_full["z"] = accel_data_full["z"].astype("float64")
    accel_data_full = accel_data_full.set_index("time")


# Load data from CSVs
def load_pid_data(
    pid: str,
    limit: int,
    offset: int,
    sample_rate: int = 20,
):
    global accel_data_full
    print("Reading in Data for person %s" % (pid))
    tac_data = pd.read_csv("data/clean_tac/%s_clean_TAC.csv" % pid)
    tac_data["timestamp"] = tac_data["timestamp"].astype("datetime64[s]")
    tac_data["TAC_Reading"] = tac_data["TAC_Reading"].astype(float)
    # Get formatted TAC data
    tac_data["TAC_Reading"] = (
        tac_data["TAC_Reading"].map(lambda tac: tac_to_class(tac)).astype("int64")
    )
    tac_data = tac_data.rename(columns={"timestamp": "time"})
    tac_data = tac_data.set_index("time")

    # Get specific accelerometer data
    accel_data_specific = accel_data_full.query("pid == @pid")
    if pid == "JB3156" or pid == "CC6740":
        # skip first row (dummy data)
        accel_data_specific = accel_data_specific.iloc[1:-1]

    # Down sample accelerometer data
    accel_data = accel_data_specific.resample("%dL" % (MS_PER_SEC / sample_rate)).last()

    if limit > len(accel_data_specific.index):
        limit = len(accel_data_specific.index)
    accel_data_specific = accel_data_specific.iloc[offset:limit]

    # Combine Data Frames to perform interpolation and backfilling
    input_data = accel_data.join(tac_data, how="outer")
    input_data = input_data.apply(pd.Series.interpolate, args=("time",))
    input_data = input_data.fillna(method="backfill")
    input_data["time"] = input_data.index
    input_data["time"] = input_data["time"].astype("int64")
    input_data["TAC_Reading"] = (
        input_data["TAC_Reading"].to_numpy().round().astype("int64")
    )

    return input_data


# Load all datafor all PIDs into common dataframe
def create_raw_train_test_dataframes(
    train_pids: list,
    test_pids: list,
    window: int,
    window_step: int,
    sample_rate: int = 20,
):
    train_data_frame = pd.DataFrame([])
    test_data_frame = pd.DataFrame([])

    # Load accel data
    load_accel_data_full()

    # Use merge accel data with PID data in test and train sets
    for pid in train_pids:
        train_data_frame = pd.concat(
            [train_data_frame, load_pid_data(pid, -1, 0, sample_rate)]
        )
    for pid in test_pids:
        test_data_frame = pd.concat(
            [test_data_frame, load_pid_data(pid, -1, 0, sample_rate)]
        )

    # Create windowed data sets of features
    print("Creating training set feature data")
    train_feature_data_frame, train_labels = create_features_data_frame(
        train_data_frame, window, window_step
    )
    print("Num of features = %d" % train_feature_data_frame.shape[1])
    print("Creating test set feature data")
    test_feature_data_frame, test_labels = create_features_data_frame(
        test_data_frame, window, window_step
    )

    # Standardize data
    scaler = StandardScaler()
    scaler.fit(train_feature_data_frame)
    s_train_data = scaler.transform(train_feature_data_frame)
    s_test_data = scaler.transform(test_feature_data_frame)

    return (s_train_data, train_labels, s_test_data, test_labels)


# Create windowed feature data from given data frame
def create_features_data_frame(df: pd.DataFrame, window: int, window_step: int):
    x_list = []
    y_list = []
    z_list = []
    t_list = []
    labels = []

    # Creating overlaping windows
    print("Create overlaping windows")
    for i in range(0, df.shape[0] - window, window_step):
        xs = df["x"].to_numpy()[i : i + 100]
        ys = df["y"].to_numpy()[i : i + 100]
        zs = df["z"].to_numpy()[i : i + 100]
        ts = df["time"].to_numpy()[i : i + 100]
        label = stats.mode(df["TAC_Reading"][i : i + 100], keepdims=True)[0][0]

        x_list.append(xs)
        y_list.append(ys)
        z_list.append(zs)
        t_list.append(ts)
        labels.append(label)

    # Frame to contain windowed feature data
    feature_frames = pd.DataFrame([])
    x_series = pd.Series(x_list)
    y_series = pd.Series(y_list)
    z_series = pd.Series(z_list)

    #####
    # Time Domain Features
    #####
    print("Extracting Time Domain Features (few minutes)")

    # raw readings
    feature_frames["x"] = x_series
    feature_frames["y"] = y_series
    feature_frames["z"] = z_series
    feature_frames["time"] = pd.Series(t_list)
    # mean
    feature_frames["x_mean"] = x_series.apply(lambda x: x.mean())
    feature_frames["y_mean"] = y_series.apply(lambda x: x.mean())
    feature_frames["z_mean"] = z_series.apply(lambda x: x.mean())
    # std dev
    feature_frames["x_std"] = x_series.apply(lambda x: x.std())
    feature_frames["y_std"] = y_series.apply(lambda x: x.std())
    feature_frames["z_std"] = z_series.apply(lambda x: x.std())
    # avg absolute diff
    feature_frames["x_aad"] = x_series.apply(
        lambda x: np.mean(np.absolute(x - np.mean(x)))
    )
    feature_frames["y_aad"] = y_series.apply(
        lambda x: np.mean(np.absolute(x - np.mean(x)))
    )
    feature_frames["z_aad"] = z_series.apply(
        lambda x: np.mean(np.absolute(x - np.mean(x)))
    )
    # min
    feature_frames["x_min"] = x_series.apply(lambda x: x.min())
    feature_frames["y_min"] = y_series.apply(lambda x: x.min())
    feature_frames["z_min"] = z_series.apply(lambda x: x.min())
    # max
    feature_frames["x_max"] = x_series.apply(lambda x: x.max())
    feature_frames["y_max"] = y_series.apply(lambda x: x.max())
    feature_frames["z_max"] = z_series.apply(lambda x: x.max())
    # max-min diff
    feature_frames["x_maxmin_diff"] = feature_frames["x_max"] - feature_frames["x_min"]
    feature_frames["y_maxmin_diff"] = feature_frames["y_max"] - feature_frames["y_min"]
    feature_frames["z_maxmin_diff"] = feature_frames["z_max"] - feature_frames["z_min"]
    # median
    feature_frames["x_median"] = x_series.apply(lambda x: np.median(x))
    feature_frames["y_median"] = y_series.apply(lambda x: np.median(x))
    feature_frames["z_median"] = z_series.apply(lambda x: np.median(x))
    # median abs dev
    feature_frames["x_mad"] = x_series.apply(
        lambda x: np.median(np.absolute(x - np.median(x)))
    )
    feature_frames["y_mad"] = y_series.apply(
        lambda x: np.median(np.absolute(x - np.median(x)))
    )
    feature_frames["z_mad"] = z_series.apply(
        lambda x: np.median(np.absolute(x - np.median(x)))
    )
    # interquartile range
    feature_frames["x_IQR"] = x_series.apply(
        lambda x: np.percentile(x, 75) - np.percentile(x, 25)
    )
    feature_frames["y_IQR"] = y_series.apply(
        lambda x: np.percentile(x, 75) - np.percentile(x, 25)
    )
    feature_frames["z_IQR"] = z_series.apply(
        lambda x: np.percentile(x, 75) - np.percentile(x, 25)
    )
    # negtive count
    feature_frames["x_neg_count"] = x_series.apply(lambda x: np.sum(x < 0))
    feature_frames["y_neg_count"] = y_series.apply(lambda x: np.sum(x < 0))
    feature_frames["z_neg_count"] = z_series.apply(lambda x: np.sum(x < 0))
    # positive count
    feature_frames["x_pos_count"] = x_series.apply(lambda x: np.sum(x > 0))
    feature_frames["y_pos_count"] = y_series.apply(lambda x: np.sum(x > 0))
    feature_frames["z_pos_count"] = z_series.apply(lambda x: np.sum(x > 0))
    # values above mean
    feature_frames["x_above_mean"] = x_series.apply(lambda x: np.sum(x > x.mean()))
    feature_frames["y_above_mean"] = y_series.apply(lambda x: np.sum(x > x.mean()))
    feature_frames["z_above_mean"] = z_series.apply(lambda x: np.sum(x > x.mean()))
    # number of peaks
    feature_frames["x_peak_count"] = x_series.apply(lambda x: len(find_peaks(x)[0]))
    feature_frames["y_peak_count"] = y_series.apply(lambda x: len(find_peaks(x)[0]))
    feature_frames["z_peak_count"] = z_series.apply(lambda x: len(find_peaks(x)[0]))
    # skewness
    # feature_frames["x_skewness"] = x_series.apply(lambda x: stats.skew(x))
    # feature_frames["y_skewness"] = y_series.apply(lambda x: stats.skew(x))
    # feature_frames["z_skewness"] = z_series.apply(lambda x: stats.skew(x))
    # kurtosis
    # feature_frames["x_kurtosis"] = x_series.apply(lambda x: stats.kurtosis(x))
    # feature_frames["y_kurtosis"] = y_series.apply(lambda x: stats.kurtosis(x))
    # feature_frames["z_kurtosis"] = z_series.apply(lambda x: stats.kurtosis(x))
    # energy
    feature_frames["x_energy"] = x_series.apply(lambda x: np.sum(x**2) / 100)
    feature_frames["y_energy"] = y_series.apply(lambda x: np.sum(x**2) / 100)
    feature_frames["z_energy"] = z_series.apply(lambda x: np.sum(x**2 / 100))
    # avg resultant
    feature_frames["avg_result_accl"] = [
        i.mean() for i in ((x_series**2 + y_series**2 + z_series**2) ** 0.5)
    ]
    # signal magnitude area
    feature_frames["sma"] = (
        x_series.apply(lambda x: np.sum(abs(x) / 100))
        + y_series.apply(lambda x: np.sum(abs(x) / 100))
        + z_series.apply(lambda x: np.sum(abs(x) / 100))
    )

    #####
    # Frequency Domain Features
    #####
    print("Extracting Frequency Domain Features (few minutes)")

    # converting the signals from time domain to frequency domain using FFT
    x_list_fft = x_series.apply(lambda x: np.abs(np.fft.fft(x))[1:51])
    y_list_fft = y_series.apply(lambda x: np.abs(np.fft.fft(x))[1:51])
    z_list_fft = z_series.apply(lambda x: np.abs(np.fft.fft(x))[1:51])
    # Statistical Features on raw x, y and z in frequency domain
    # FFT mean
    feature_frames["x_mean_fft"] = x_list_fft.apply(lambda x: x.mean())
    feature_frames["y_mean_fft"] = y_list_fft.apply(lambda x: x.mean())
    feature_frames["z_mean_fft"] = z_list_fft.apply(lambda x: x.mean())
    # FFT std dev
    feature_frames["x_std_fft"] = x_list_fft.apply(lambda x: x.std())
    feature_frames["y_std_fft"] = y_list_fft.apply(lambda x: x.std())
    feature_frames["z_std_fft"] = z_list_fft.apply(lambda x: x.std())
    # FFT avg absolute diff
    feature_frames["x_aad_fft"] = x_list_fft.apply(
        lambda x: np.mean(np.absolute(x - np.mean(x)))
    )
    feature_frames["y_aad_fft"] = y_list_fft.apply(
        lambda x: np.mean(np.absolute(x - np.mean(x)))
    )
    feature_frames["z_aad_fft"] = z_list_fft.apply(
        lambda x: np.mean(np.absolute(x - np.mean(x)))
    )
    # FFT min
    feature_frames["x_min_fft"] = x_list_fft.apply(lambda x: x.min())
    feature_frames["y_min_fft"] = y_list_fft.apply(lambda x: x.min())
    feature_frames["z_min_fft"] = z_list_fft.apply(lambda x: x.min())
    # FFT max
    feature_frames["x_max_fft"] = x_list_fft.apply(lambda x: x.max())
    feature_frames["y_max_fft"] = y_list_fft.apply(lambda x: x.max())
    feature_frames["z_max_fft"] = z_list_fft.apply(lambda x: x.max())
    # FFT max-min diff
    feature_frames["x_maxmin_diff_fft"] = (
        feature_frames["x_max_fft"] - feature_frames["x_min_fft"]
    )
    feature_frames["y_maxmin_diff_fft"] = (
        feature_frames["y_max_fft"] - feature_frames["y_min_fft"]
    )
    feature_frames["z_maxmin_diff_fft"] = (
        feature_frames["z_max_fft"] - feature_frames["z_min_fft"]
    )
    # FFT median
    feature_frames["x_median_fft"] = x_list_fft.apply(lambda x: np.median(x))
    feature_frames["y_median_fft"] = y_list_fft.apply(lambda x: np.median(x))
    feature_frames["z_median_fft"] = z_list_fft.apply(lambda x: np.median(x))
    # FFT median abs dev
    feature_frames["x_mad_fft"] = x_list_fft.apply(
        lambda x: np.median(np.absolute(x - np.median(x)))
    )
    feature_frames["y_mad_fft"] = y_list_fft.apply(
        lambda x: np.median(np.absolute(x - np.median(x)))
    )
    feature_frames["z_mad_fft"] = z_list_fft.apply(
        lambda x: np.median(np.absolute(x - np.median(x)))
    )
    # FFT Interquartile range
    feature_frames["x_IQR_fft"] = x_list_fft.apply(
        lambda x: np.percentile(x, 75) - np.percentile(x, 25)
    )
    feature_frames["y_IQR_fft"] = y_list_fft.apply(
        lambda x: np.percentile(x, 75) - np.percentile(x, 25)
    )
    feature_frames["z_IQR_fft"] = z_list_fft.apply(
        lambda x: np.percentile(x, 75) - np.percentile(x, 25)
    )
    # FFT values above mean
    feature_frames["x_above_mean_fft"] = x_list_fft.apply(
        lambda x: np.sum(x > x.mean())
    )
    feature_frames["y_above_mean_fft"] = y_list_fft.apply(
        lambda x: np.sum(x > x.mean())
    )
    feature_frames["z_above_mean_fft"] = z_list_fft.apply(
        lambda x: np.sum(x > x.mean())
    )
    # FFT number of peaks
    feature_frames["x_peak_count_fft"] = x_list_fft.apply(
        lambda x: len(find_peaks(x)[0])
    )
    feature_frames["y_peak_count_fft"] = y_list_fft.apply(
        lambda x: len(find_peaks(x)[0])
    )
    feature_frames["z_peak_count_fft"] = z_list_fft.apply(
        lambda x: len(find_peaks(x)[0])
    )
    # FFT skewness
    # feature_frames["x_skewness_fft"] = x_list_fft.apply(lambda x: stats.skew(x))
    # feature_frames["y_skewness_fft"] = y_list_fft.apply(lambda x: stats.skew(x))
    # feature_frames["z_skewness_fft"] = z_list_fft.apply(lambda x: stats.skew(x))
    # FFT kurtosis
    # feature_frames["x_kurtosis_fft"] = x_list_fft.apply(lambda x: stats.kurtosis(x))
    # feature_frames["y_kurtosis_fft"] = y_list_fft.apply(lambda x: stats.kurtosis(x))
    # feature_frames["z_kurtosis_fft"] = z_list_fft.apply(lambda x: stats.kurtosis(x))
    # FFT energy
    feature_frames["x_energy_fft"] = x_list_fft.apply(lambda x: np.sum(x**2) / 50)
    feature_frames["y_energy_fft"] = y_list_fft.apply(lambda x: np.sum(x**2) / 50)
    feature_frames["z_energy_fft"] = z_list_fft.apply(lambda x: np.sum(x**2 / 50))
    # FFT avg resultant
    feature_frames["avg_result_accl_fft"] = [
        i.mean() for i in ((x_list_fft**2 + y_list_fft**2 + z_list_fft**2) ** 0.5)
    ]
    # FFT Signal magnitude area
    feature_frames["sma_fft"] = (
        x_list_fft.apply(lambda x: np.sum(abs(x) / 50))
        + y_list_fft.apply(lambda x: np.sum(abs(x) / 50))
        + z_list_fft.apply(lambda x: np.sum(abs(x) / 50))
    )

    #####
    # Index features
    #####
    print("Extracting Index Features (few minutes)")

    # index of max value in time domain
    feature_frames["x_argmax"] = x_series.apply(lambda x: np.argmax(x))
    feature_frames["y_argmax"] = y_series.apply(lambda x: np.argmax(x))
    feature_frames["z_argmax"] = z_series.apply(lambda x: np.argmax(x))
    # index of min value in time domain
    feature_frames["x_argmin"] = x_series.apply(lambda x: np.argmin(x))
    feature_frames["y_argmin"] = y_series.apply(lambda x: np.argmin(x))
    feature_frames["z_argmin"] = z_series.apply(lambda x: np.argmin(x))
    # absolute difference between above indices
    feature_frames["x_arg_diff"] = abs(
        feature_frames["x_argmax"] - feature_frames["x_argmin"]
    )
    feature_frames["y_arg_diff"] = abs(
        feature_frames["y_argmax"] - feature_frames["y_argmin"]
    )
    feature_frames["z_arg_diff"] = abs(
        feature_frames["z_argmax"] - feature_frames["z_argmin"]
    )
    # absolute difference between above indices
    feature_frames["x_arg_diff_fft"] = abs(
        feature_frames["x_argmax_fft"] - feature_frames["x_argmin_fft"]
    )
    feature_frames["y_arg_diff_fft"] = abs(
        feature_frames["y_argmax_fft"] - feature_frames["y_argmin_fft"]
    )
    feature_frames["z_arg_diff_fft"] = abs(
        feature_frames["z_argmax_fft"] - feature_frames["z_argmin_fft"]
    )

    return (feature_frames, labels)
