import torch
import numpy as np
import torchhd

# from torchhd import embeddings
from torchhd_custom import embeddings

SIGNAL_MIN = -5
SIGNAL_MAX = 5

NUM_RMS = 3
NUM_MFCC = 6
NUM_FFT_MEAN = 3
NUM_FFT_MAX = 3
NUM_FFT_VAR = 3
NUM_MEAN = 3
NUM_MAX = 3
NUM_VAR = 3
NUM_SPECTRAL_CENTROID = 3

RMS_START = 0
MFCC_START = RMS_START + NUM_RMS
FFT_MEAN_START = MFCC_START + NUM_MFCC
FFT_MAX_START = FFT_MEAN_START + NUM_FFT_MEAN
FFT_VAR_START = FFT_MAX_START + NUM_FFT_MAX
MEAN_START = FFT_VAR_START + NUM_FFT_VAR
MAX_START = MEAN_START + NUM_MEAN
VAR_START = MAX_START + NUM_MAX
SP_CNTD_START = VAR_START + NUM_VAR


# HDC Encoder for Bar Crawl Data
class HdcLevelEncoder(torch.nn.Module):
    def __init__(self, levels: int, timestamps: int, out_dimension: int):
        super(HdcLevelEncoder, self).__init__()

        self.keys = embeddings.Random(timestamps, out_dimension, dtype=torch.float64)
        self.embed = embeddings.Level(levels, out_dimension, dtype=torch.float64, low=SIGNAL_MIN,
            high=SIGNAL_MAX)
        self.timestamps = embeddings.Level(
            timestamps, out_dimension, dtype=torch.float64, low=0, high=timestamps
        )

        self.feat_rms_kernel = embeddings.Sinusoid(
            NUM_RMS, out_dimension, dtype=torch.float64
        )
        self.feat_mfcc_kernel = embeddings.Sinusoid(
            NUM_MFCC, out_dimension, dtype=torch.float64
        )
        self.feat_fft_mean_kernel = embeddings.Sinusoid(
            NUM_FFT_MEAN, out_dimension, dtype=torch.float64
        )
        self.feat_fft_max_kernel = embeddings.Sinusoid(
            NUM_FFT_MAX, out_dimension, dtype=torch.float64
        )
        self.feat_fft_var_kernel = embeddings.Sinusoid(
            NUM_FFT_VAR, out_dimension, dtype=torch.float64
        )
        self.feat_mean_kernel = embeddings.Sinusoid(
            NUM_MEAN, out_dimension, dtype=torch.float64
        )
        self.feat_max_kernel = embeddings.Sinusoid(
            NUM_MAX, out_dimension, dtype=torch.float64
        )
        self.feat_var_kernel = embeddings.Sinusoid(
            NUM_VAR, out_dimension, dtype=torch.float64
        )
        self.feat_spectral_centroid_kernel = embeddings.Sinusoid(
            NUM_SPECTRAL_CENTROID, out_dimension, dtype=torch.float64
        )

    # Encode window of raw vectors (x,y,z) and feature vectors (f,)
    def forward(self, input: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        # Get hypervectors for x, y, z samples
        xyz = torch.clamp(input[:, 1:], min=SIGNAL_MIN, max=SIGNAL_MAX)
        xyz_levels = self.embed(xyz)
        xyz_level = torchhd.multiset(xyz_levels)
        key_weights = self.keys.weight[:input.shape[0]]
        sample_hvs = torchhd.bind(xyz_level, key_weights)
        # Get time hypervectors
        times = self.timestamps(input[:, 0])
        # Bind time sequence for x, y, z samples
        sample_hvs = sample_hvs * times
        sample_hv = torchhd.multiset(sample_hvs)
        # Encode calculated features
        sample_f1_hv = self.feat_rms_kernel(feat[RMS_START : RMS_START + NUM_RMS])
        sample_f2_hv = self.feat_mfcc_kernel(feat[MFCC_START : MFCC_START + NUM_MFCC])
        sample_f3_hv = self.feat_fft_mean_kernel(
            feat[FFT_MEAN_START : FFT_MEAN_START + NUM_FFT_MEAN]
        )
        sample_f4_hv = self.feat_fft_max_kernel(
            feat[FFT_MAX_START : FFT_MAX_START + NUM_FFT_MAX]
        )
        sample_f5_hv = self.feat_fft_var_kernel(
            feat[FFT_VAR_START : FFT_VAR_START + NUM_FFT_VAR]
        )
        sample_f6_hv = self.feat_mean_kernel(feat[MEAN_START : MEAN_START + NUM_MEAN])
        sample_f7_hv = self.feat_max_kernel(feat[MAX_START : MAX_START + NUM_MAX])
        sample_f8_hv = self.feat_var_kernel(feat[VAR_START : VAR_START + NUM_VAR])
        sample_f9_hv = self.feat_spectral_centroid_kernel(
            feat[SP_CNTD_START : SP_CNTD_START + NUM_SPECTRAL_CENTROID]
        )

        sample_hv = (
            sample_hv
            * sample_f1_hv  # Misc features
            * (sample_f2_hv + sample_f9_hv)  # Spectral features
            * (sample_f3_hv + sample_f4_hv + sample_f5_hv)  # Frequency domain features
            * (sample_f6_hv + sample_f7_hv + sample_f8_hv)  # Time domain features
        )
        # Apply activation function
        sample_hv = torchhd.hard_quantize(sample_hv)  # torch.sin(sample_hv)
        return sample_hv.flatten()
