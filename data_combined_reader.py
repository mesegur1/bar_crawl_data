import warnings
import numpy as np
import pandas as pd
import torch
import random
import pickle
from tqdm import tqdm
from scipy import stats
from scipy.signal import welch
from collections import OrderedDict
import librosa
from sklearn.model_selection import train_test_split

TAC_LEVEL_0 = 0  # < 0.080 g/dl
TAC_LEVEL_1 = 1  # >= 0.080 g/dl

MS_PER_SEC = 1000

# Data windowing settings
WINDOW = 400  # 10 second window: 10 seconds * 40Hz = 400 samples per window
WINDOW_STEP = 360  # 8 second step: 9 seconds * 40Hz = 360 samples per step
START_OFFSET = 0
END_INDEX = np.inf
TRAINING_EPOCHS = 1
SAMPLE_RATE = 40  # Hz
TEST_RATIO = 0.25
MOTION_EPSILON = 0.0001

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
    accel_data_full["x"] = accel_data_full["x"].astype(float)
    accel_data_full["y"] = accel_data_full["y"].astype(float)
    accel_data_full["z"] = accel_data_full["z"].astype(float)
    accel_data_full = accel_data_full.set_index("time")


def load_combined_data(pids: list):
    train_data_set = []
    test_data_set = []
    print("Reading in all data")
    for pid in pids:
        # Load from PKLs
        with open("data/%s_random_train_set.pkl" % pid, "rb") as file:
            train_set = pickle.load(file)
        with open("data/%s_random_test_set.pkl" % pid, "rb") as file:
            test_set = pickle.load(file)
        for d in train_set:
            train_data_set.append(d)
        for d in test_set:
            test_data_set.append(d)
    print("Randomly shuffle windows")
    random.shuffle(train_data_set)
    random.shuffle(test_data_set)

    window = WINDOW
    with open("data/window_size.pkl", "rb") as file:
        window = pickle.load(file)

    return (window, train_data_set, test_data_set)


# Load data from CSVs into PKL files
def load_data(
    pid: str,
    limit: int,
    offset: int,
    window: int,
    window_step: int,
    sample_rate: int = 20,
    test_ratio: float = 0.5,
):
    global accel_data_full
    print("Reading in Data for person %s" % (pid))
    tac_data = pd.read_csv("data/clean_tac/%s_clean_TAC.csv" % pid)
    tac_data["timestamp"] = tac_data["timestamp"].astype("datetime64[s]")
    tac_data["TAC_Reading"] = tac_data["TAC_Reading"].astype(float)
    tac_data = tac_data.rename(columns={"timestamp": "time"})
    tac_data = tac_data.set_index("time")

    # Get specific accelerometer data
    accel_data_specific = accel_data_full.query("pid == @pid")
    if pid == "JB3156" or pid == "CC6740":
        # skip first row (dummy data)
        accel_data_specific = accel_data_specific.iloc[1:-1]

    # Down sample accelerometer data
    accel_data = accel_data_specific.resample("%dL" % (MS_PER_SEC / sample_rate)).last()

    # Combine Data Frames to perform interpolation and backfilling
    input_data = accel_data.join(tac_data, how="outer")
    input_data = input_data.apply(pd.Series.interpolate, args=("time",))
    input_data = input_data.fillna(method="backfill")
    input_data["time"] = input_data.index
    input_data["time"] = input_data["time"].astype("int64")

    if limit > len(input_data.index):
        limit = len(input_data.index)
    input_data = input_data.iloc[offset:limit]

    print("Total Data length: %d" % (len(input_data.index)))

    # Get formatted TAC data
    input_data["TAC_Reading"] = (
        input_data["TAC_Reading"].map(lambda tac: tac_to_class(tac)).astype("int64")
    )

    # Split data back into two parts for train/test set creation
    accel_data = input_data[["time", "x", "y", "z"]].to_numpy()
    tac_data_labels = input_data["TAC_Reading"].to_numpy()

    # Change training data to be windowed
    data_accel_w = [
        accel_data[base : base + window]
        for base in range(0, len(accel_data), window_step)
    ]
    data_feat_w = feature_extraction(accel_data, sample_rate, window, window_step)
    data_tac_w = [
        stats.mode(tac_data_labels[base : base + window], keepdims=True)[0][0]
        for base in range(0, len(tac_data_labels), window_step)
    ]

    # Removing zeroed windows
    print("Removing zeroed accel data windows")
    zero_windows_i = []
    for i in range(0, len(data_accel_w)):
        if is_greater_than(data_accel_w[i], MOTION_EPSILON) == False:
            zero_windows_i.append(i)
    data_accel_w = [
        w for w in data_accel_w if w not in [data_accel_w[i] for i in zero_windows_i]
    ]
    data_feat_w = [
        w for w in data_feat_w if w not in [data_feat_w[i] for i in zero_windows_i]
    ]
    data_tac_w = [
        w for w in data_tac_w if w not in [data_tac_w[i] for i in zero_windows_i]
    ]

    print("Creating data sets")
    # Split data into two parts
    (
        train_data_accel,
        test_data_accel,
        train_data_feat,
        test_data_feat,
        train_data_tac,
        test_data_tac,
    ) = train_test_split(
        data_accel_w,
        data_feat_w,
        data_tac_w,
        test_size=test_ratio,
        shuffle=True,
    )
    train_length = len(train_data_accel)
    test_length = len(test_data_accel)

    train_set = list(zip(train_data_accel, train_data_feat, train_data_tac))
    print("Number of Windows For Training: %d" % (train_length))
    test_set = list(zip(test_data_accel, test_data_feat, test_data_tac))
    print("Number of Windows For Testing: %d" % (test_length))

    with open("data/%s_random_train_set.pkl" % pid, "wb") as file:
        pickle.dump(train_set, file)

    with open("data/%s_random_test_set.pkl" % pid, "wb") as file:
        pickle.dump(test_set, file)


def is_greater_than(x: torch.Tensor, eps: float):
    count = (x > eps).sum()
    if count > 0:
        return True
    return False

def feature_extraction(accel_data : np.ndarray, sample_rate : float, window : int, window_step : int):
    window_data = []
    for base in range(0, len(accel_data), window_step):
        xyz = accel_data[base : base + window, 1:]
        
        #Create dictionary of features
        feature_dict = OrderedDict()
        #Root mean square
        feature_dict["rms"] = accel_rms(xyz)
        #MFCC covariance
        feature_dict["mfcc_cov"] = accel_mfcc_cov(xyz, sample_rate)
        #FFT mean
        feature_dict["fft_mean"] = accel_fft_mean(xyz)
        #FFT max
        feature_dict["fft_max"] = accel_fft_max(xyz)
        #FFT variance
        feature_dict["fft_var"] = accel_fft_var(xyz)
        #Mean
        feature_dict["mean"] = accel_mean(xyz)
        #Max
        feature_dict["max"] = accel_max(xyz)
        #Variance
        feature_dict["var"] = accel_var(xyz)
        #Spectral centroid
        feature_dict["spectral_centroid"] = spectral_centroid(xyz, sample_rate)

        #Flatten dictionary and save
        features = np.concatenate([feature_dict[column].flatten() for column in feature_dict])
        window_data.append(features)
    return window_data


def accel_rms(xyz: np.ndarray):
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    frame_length = xyz.shape[0]
    warnings.filterwarnings("ignore")  # There is a harmless padding warning
    rms_x = librosa.feature.rms(y=x, frame_length=frame_length, center=False)
    rms_y = librosa.feature.rms(y=y, frame_length=frame_length, center=False)
    rms_z = librosa.feature.rms(y=z, frame_length=frame_length, center=False)
    return np.array([rms_x.item(), rms_y.item(), rms_z.item()]).flatten()


def accel_mfcc_cov(xyz: np.ndarray, sample_rate: float):
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    frame_length = xyz.shape[0]
    warnings.filterwarnings("ignore")  # There is a harmless padding warning
    mfcc_x = librosa.feature.mfcc(
        y=x, sr=sample_rate, n_mfcc=13, n_fft=frame_length, lifter=22, window=frame_length
    )
    mfcc_y = librosa.feature.mfcc(
        y=y, sr=sample_rate, n_mfcc=13, n_fft=frame_length, lifter=22, window=frame_length
    )
    mfcc_z = librosa.feature.mfcc(
        y=z, sr=sample_rate, n_mfcc=13, n_fft=frame_length, lifter=22, window=frame_length
    )

    mfcc_cov_x = mfcc_x.T @ mfcc_x
    mfcc_cov_y = mfcc_y.T @ mfcc_y
    mfcc_cov_z = mfcc_z.T @ mfcc_z
    mfcc_cov_xy = mfcc_x.T @ mfcc_y
    mfcc_cov_xz = mfcc_x.T @ mfcc_z
    mfcc_cov_yz = mfcc_y.T @ mfcc_z

    mfcc_cov = np.concatenate(
        (mfcc_cov_x, mfcc_cov_y, mfcc_cov_z, mfcc_cov_xy, mfcc_cov_xz, mfcc_cov_yz),
        axis=None,
    )
    return mfcc_cov.flatten()


def accel_fft_mean(xyz: np.ndarray):
    x_ffts = np.abs(np.fft.fft(xyz[:, 0]))
    x_ffts = x_ffts[0:int(xyz.shape[0]/2)]
    x_ffts = x_ffts / len(x_ffts)
    y_ffts = np.abs(np.fft.fft(xyz[:, 1]))
    y_ffts = y_ffts[0:int(xyz.shape[0]/2)]
    y_ffts = y_ffts / len(y_ffts)
    z_ffts = np.abs(np.fft.fft(xyz[:, 2]))
    z_ffts = z_ffts[0:int(xyz.shape[0]/2)]
    z_ffts = z_ffts / len(z_ffts)

    x_fft_mean = x_ffts.mean()
    y_fft_mean = y_ffts.mean()
    z_fft_mean = z_ffts.mean()

    return np.array([x_fft_mean, y_fft_mean, z_fft_mean])


def accel_fft_max(xyz: np.ndarray):
    x_ffts = np.abs(np.fft.fft(xyz[:, 0]))
    x_ffts = x_ffts[0:int(xyz.shape[0]/2)]
    x_ffts = x_ffts / len(x_ffts)
    y_ffts = np.abs(np.fft.fft(xyz[:, 1]))
    y_ffts = y_ffts[0:int(xyz.shape[0]/2)]
    y_ffts = y_ffts / len(y_ffts)
    z_ffts = np.abs(np.fft.fft(xyz[:, 2]))
    z_ffts = z_ffts[0:int(xyz.shape[0]/2)]
    z_ffts = z_ffts / len(z_ffts)

    x_fft_max = x_ffts.max()
    y_fft_max = y_ffts.max()
    z_fft_max = z_ffts.max()

    return np.array([x_fft_max, y_fft_max, z_fft_max])


def accel_fft_var(xyz: np.ndarray):
    x_ffts = np.abs(np.fft.fft(xyz[:, 0]))
    x_ffts = x_ffts[0:int(xyz.shape[0]/2)]
    x_ffts = x_ffts / len(x_ffts)
    y_ffts = np.abs(np.fft.fft(xyz[:, 1]))
    y_ffts = y_ffts[0:int(xyz.shape[0]/2)]
    y_ffts = y_ffts / len(y_ffts)
    z_ffts = np.abs(np.fft.fft(xyz[:, 2]))
    z_ffts = z_ffts[0:int(xyz.shape[0]/2)]
    z_ffts = z_ffts / len(z_ffts)

    x_fft_var = x_ffts.var()
    y_fft_var = y_ffts.var()
    z_fft_var = z_ffts.var()

    return np.array([x_fft_var, y_fft_var, z_fft_var])


def accel_mean(xyz: np.ndarray):
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    x_mean = x.mean()
    y_mean = y.mean()
    z_mean = z.mean()

    return np.array([x_mean, y_mean, z_mean])

def accel_median(xyz: np.ndarray):
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    x_median = np.median(x)
    y_median = np.median(y)
    z_median = np.median(z)

    return np.array([x_median, y_median, z_median])

def accel_std(xyz: np.ndarray):
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    x_std = x.std()
    y_std = y.std()
    z_std = z.std()

    return np.array([x_std, y_std, z_std])

def accel_max(xyz: np.ndarray):
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    x_max = x.max()
    y_max = y.max()
    z_max = z.max()

    return np.array([x_max, y_max, z_max])

def accel_min(xyz: np.ndarray):
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    x_min = x.min()
    y_min = y.min()
    z_min = z.min()

    return np.array([x_min, y_min, z_min])

def accel_abs_max(xyz: np.ndarray):
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    x_max = np.abs(x).max()
    y_max = np.abs(y).max()
    z_max = np.abs(z).max()

    return np.array([x_max, y_max, z_max])

def accel_abs_min(xyz: np.ndarray):
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    x_min = np.abs(x).min()
    y_min = np.abs(y).min()
    z_min = np.abs(z).min()

    return np.array([x_min, y_min, z_min])

def accel_var(xyz: np.ndarray):
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    x_var = x.var()
    y_var = y.var()
    z_var = z.var()

    return np.array([x_var, y_var, z_var])

def zero_crossing_rate(xyz: np.ndarray):
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    frame_length = xyz.shape[0]
    warnings.filterwarnings("ignore")  # There is a harmless padding warning
    zero_x = librosa.feature.zero_crossing_rate(y=x, frame_length=frame_length, center=False)
    zero_y = librosa.feature.zero_crossing_rate(y=y, frame_length=frame_length, center=False)
    zero_z = librosa.feature.zero_crossing_rate(y=z, frame_length=frame_length, center=False)
    return np.array([zero_x.item(), zero_y.item(), zero_z.item()])

def spectral_entropy_s(signal: np.ndarray, n_short_blocks=10):
    eps = 0.00000001

    # number of frame samples
    num_frames = len(signal)

    # total spectral energy
    total_energy = np.sum(signal ** 2)

    # length of sub-frame
    sub_win_len = int(np.floor(num_frames / n_short_blocks))
    if num_frames != sub_win_len * n_short_blocks:
        signal = signal[0:sub_win_len * n_short_blocks]

    # define sub-frames (using matrix reshape)
    sub_wins = signal.reshape(sub_win_len, n_short_blocks, order='F').copy()

    # compute spectral sub-energies
    s = np.sum(sub_wins ** 2, axis=0) / (total_energy + eps)

    # compute spectral entropy
    entropy = -np.sum(s * np.log2(s + eps))

    return entropy

def spectral_entropy(xyz: np.ndarray):
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    ent_x = spectral_entropy_s(x)
    ent_y = spectral_entropy_s(y)
    ent_z = spectral_entropy_s(z)

    return np.array([ent_x, ent_y, ent_z])

def spectral_entropy_fft(xyz: np.ndarray):
    x_ffts = np.abs(np.fft.fft(xyz[:, 0]))
    x_ffts = x_ffts[0:int(xyz.shape[0]/2)]
    x_ffts = x_ffts / len(x_ffts)
    y_ffts = np.abs(np.fft.fft(xyz[:, 1]))
    y_ffts = y_ffts[0:int(xyz.shape[0]/2)]
    y_ffts = y_ffts / len(y_ffts)
    z_ffts = np.abs(np.fft.fft(xyz[:, 2]))
    z_ffts = z_ffts[0:int(xyz.shape[0]/2)]
    z_ffts = z_ffts / len(z_ffts)

    ent_x = spectral_entropy_s(x_ffts)
    ent_y = spectral_entropy_s(y_ffts)
    ent_z = spectral_entropy_s(z_ffts)

    return np.array([ent_x, ent_y, ent_z])

def spectral_centroid(xyz: np.ndarray, sample_rate: float):
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    frame_length = xyz.shape[0]
    warnings.filterwarnings("ignore")  # There is a harmless padding warning
    cent_x = librosa.feature.spectral_centroid(y=x, sr=sample_rate, n_fft=frame_length)
    cent_y = librosa.feature.spectral_centroid(y=y, sr=sample_rate, n_fft=frame_length)
    cent_z = librosa.feature.spectral_centroid(y=z, sr=sample_rate, n_fft=frame_length)
    return np.array([cent_x.item(), cent_y.item(), cent_z.item()])


def spectral_spread_s(fft_magnitude: np.ndarray, sampling_rate: float):
    eps = 0.00000001
    ind = (np.arange(1, len(fft_magnitude) + 1)) * (sampling_rate / (2.0 * len(fft_magnitude)))

    Xt = fft_magnitude.copy()
    Xt = Xt / Xt.max()
    NUM = np.sum(ind * Xt)
    DEN = np.sum(Xt) + eps

    # Spread:
    spread = np.sqrt(np.sum(((ind - (NUM / DEN)) ** 2) * Xt) / DEN)

    # Normalize:
    spread = spread / (sampling_rate / 2.0)

    return spread

def spectral_spread(xyz: np.ndarray, sample_rate: float):
    x_ffts = np.abs(np.fft.fft(xyz[:, 0]))
    x_ffts = x_ffts[0:int(xyz.shape[0]/2)]
    x_ffts = x_ffts / len(x_ffts)
    y_ffts = np.abs(np.fft.fft(xyz[:, 1]))
    y_ffts = y_ffts[0:int(xyz.shape[0]/2)]
    y_ffts = y_ffts / len(y_ffts)
    z_ffts = np.abs(np.fft.fft(xyz[:, 2]))
    z_ffts = z_ffts[0:int(xyz.shape[0]/2)]
    z_ffts = z_ffts / len(z_ffts)

    spread_x = spectral_spread_s(x_ffts, sample_rate)
    spread_y = spectral_spread_s(y_ffts, sample_rate)
    spread_z = spectral_spread_s(z_ffts, sample_rate)

    return np.array([spread_x, spread_y, spread_z])

def spectral_flux_s(fft_magnitude, previous_fft_magnitude):
    eps = 0.00000001
    # compute the spectral flux as the sum of square distances:
    fft_sum = np.sum(fft_magnitude + eps)
    previous_fft_sum = np.sum(previous_fft_magnitude + eps)
    sp_flux = np.sum((fft_magnitude / fft_sum - previous_fft_magnitude / previous_fft_sum) ** 2)

    return sp_flux

def spectral_flux(xyz: np.ndarray, prev_xyz: np.ndarray):
    x_ffts = np.abs(np.fft.fft(xyz[:, 0]))
    x_ffts = x_ffts[0:int(xyz.shape[0]/2)]
    x_ffts = x_ffts / len(x_ffts)
    y_ffts = np.abs(np.fft.fft(xyz[:, 1]))
    y_ffts = y_ffts[0:int(xyz.shape[0]/2)]
    y_ffts = y_ffts / len(y_ffts)
    z_ffts = np.abs(np.fft.fft(xyz[:, 2]))
    z_ffts = z_ffts[0:int(xyz.shape[0]/2)]
    z_ffts = z_ffts / len(z_ffts)

    prev_x_ffts = np.abs(np.fft.fft(prev_xyz[:, 0]))
    prev_x_ffts = prev_x_ffts[0:int(prev_xyz.shape[0]/2)]
    prev_x_ffts = prev_x_ffts / len(prev_x_ffts)
    prev_y_ffts = np.abs(np.fft.fft(prev_xyz[:, 1]))
    prev_y_ffts = prev_y_ffts[0:int(prev_xyz.shape[0]/2)]
    prev_y_ffts = prev_y_ffts / len(prev_y_ffts)
    prev_z_ffts = np.abs(np.fft.fft(prev_xyz[:, 2]))
    prev_z_ffts = prev_z_ffts[0:int(prev_xyz.shape[0]/2)]
    prev_z_ffts = prev_z_ffts / len(prev_z_ffts)

    flux_x = spectral_flux_s(x_ffts, prev_x_ffts)
    flux_y = spectral_flux_s(y_ffts, prev_y_ffts)
    flux_z = spectral_flux_s(z_ffts, prev_z_ffts)

    return np.array([flux_x, flux_y, flux_z])

def spectral_rolloff_s(signal: np.ndarray, c=0.90):
    eps = 0.00000001
    energy = np.sum(signal ** 2)
    fft_length = len(signal)
    threshold = c * energy
    # Find the spectral rolloff as the frequency position where the respective spectral energy is equal to c*totalEnergy
    cumulative_sum = np.cumsum(signal ** 2) + eps
    a = np.nonzero(cumulative_sum > threshold)[0]
    sp_rolloff = 0.0
    if len(a) > 0: sp_rolloff = np.float64(a[0]) / (float(fft_length))

    return sp_rolloff

def spectral_rolloff(xyz: np.ndarray):
    x_ffts = np.abs(np.fft.fft(xyz[:, 0]))
    x_ffts = x_ffts[0:int(xyz.shape[0]/2)]
    x_ffts = x_ffts / len(x_ffts)
    y_ffts = np.abs(np.fft.fft(xyz[:, 1]))
    y_ffts = y_ffts[0:int(xyz.shape[0]/2)]
    y_ffts = y_ffts / len(y_ffts)
    z_ffts = np.abs(np.fft.fft(xyz[:, 2]))
    z_ffts = z_ffts[0:int(xyz.shape[0]/2)]
    z_ffts = z_ffts / len(z_ffts)

    ro_x = spectral_rolloff_s(x_ffts)
    ro_y = spectral_rolloff_s(y_ffts)
    ro_z = spectral_rolloff_s(z_ffts)

    return np.array([ro_x, ro_y, ro_z])

def skewness(xyz: np.ndarray):
    skew = stats.skew(xyz, axis=0)
    return skew.flatten()

def kurtosis(xyz: np.ndarray):
    k = stats.kurtosis(xyz, axis=0)
    return k.flatten()

def avg_power(xyz: np.ndarray, sample_rate: float):
    _, power = welch(xyz, sample_rate, axis=sample_rate)
    return np.mean(power, axis=0).flatten()

def avg_stft_per_frame_s(x: np.ndarray):
    frame_length = x.shape[0]
    warnings.filterwarnings("ignore")  # There is a harmless padding warning
    stft = librosa.amplitude_to_db(np.abs(librosa.stft(x, n_fft=frame_length, hop_length=frame_length, center=False)), ref=np.max)
    return stft.mean(axis=1)

def avg_stft_per_frame(xyz: np.ndarray):
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    stft_x = avg_stft_per_frame_s(x)
    stft_y = avg_stft_per_frame_s(y)
    stft_z = avg_stft_per_frame_s(z)
    
    return np.array([stft_x, stft_y, stft_z])

if __name__ == "__main__":
    load_accel_data_full()
    print("Reading in all data")
    for pid in PIDS:
        # Load data from CSVs
        load_data(
            pid, END_INDEX, START_OFFSET, WINDOW, WINDOW_STEP, SAMPLE_RATE, TEST_RATIO
        )

    with open("data/window_size.pkl", "wb") as file:
        pickle.dump(WINDOW, file)
