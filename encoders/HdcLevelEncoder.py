import torch
import numpy as np
import torchhd

# from torchhd import embeddings
from torchhd_custom import embeddings

SIGNAL_X_MIN = -5
SIGNAL_X_MAX = 5
SIGNAL_Y_MIN = -5
SIGNAL_Y_MAX = 5
SIGNAL_Z_MIN = -5
SIGNAL_Z_MAX = 5
NUM_RMS = 3
NUM_MFCC = 3
NUM_FFT_MEAN = 3
NUM_FFT_MAX = 3
RMS_START = 0
MFCC_START = RMS_START + NUM_RMS
FFT_MEAN_START = MFCC_START + NUM_MFCC
FFT_MAX_START = FFT_MEAN_START + NUM_FFT_MEAN


# HDC Encoder for Bar Crawl Data
class HdcLevelEncoder(torch.nn.Module):
    def __init__(self, levels: int, timestamps: int, out_dimension: int):
        super(HdcLevelEncoder, self).__init__()

        self.signal_level_x = embeddings.Level(
            levels,
            out_dimension,
            dtype=torch.float64,
            low=SIGNAL_X_MIN,
            high=SIGNAL_X_MAX,
        )
        self.signal_level_y = embeddings.Level(
            levels,
            out_dimension,
            dtype=torch.float64,
            low=SIGNAL_Y_MIN,
            high=SIGNAL_Y_MAX,
        )
        self.signal_level_z = embeddings.Level(
            levels,
            out_dimension,
            dtype=torch.float64,
            low=SIGNAL_Z_MIN,
            high=SIGNAL_Z_MAX,
        )
        self.timestamps = embeddings.Level(
            timestamps, out_dimension, dtype=torch.float64, low=0, high=timestamps
        )

        self.feat_rms_kernel = embeddings.Sinusoid(NUM_RMS, out_dimension, dtype=torch.float64)
        self.feat_mfcc_kernel = embeddings.Sinusoid(NUM_MFCC, out_dimension, dtype=torch.float64)
        self.feat_fft_mean_kernel = embeddings.Sinusoid(NUM_FFT_MEAN, out_dimension, dtype=torch.float64)
        self.feat_fft_max_kernel = embeddings.Sinusoid(NUM_FFT_MAX, out_dimension, dtype=torch.float64)

    # Encode window of raw vectors (x,y,z) and feature vectors (f,)
    def forward(self, input: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        # Get level hypervectors for x, y, z samples
        x_signal = torch.clamp(input[:, 1], min=SIGNAL_X_MIN, max=SIGNAL_X_MAX)
        y_signal = torch.clamp(input[:, 2], min=SIGNAL_Y_MIN, max=SIGNAL_Y_MAX)
        z_signal = torch.clamp(input[:, 3], min=SIGNAL_Z_MIN, max=SIGNAL_Z_MAX)
        x_levels = self.signal_level_x(x_signal)
        y_levels = self.signal_level_y(y_signal)
        z_levels = self.signal_level_z(z_signal)
        # Get time hypervectors
        times = self.timestamps(input[:, 0])
        # Bind time sequence for x, y, z samples
        sample_hvs = (x_levels + y_levels + z_levels) * times
        sample_hv = torchhd.multibind(sample_hvs)
        # Encode calculated features
        sample_f1_hv = self.feat_rms_kernel(feat[RMS_START : RMS_START + NUM_RMS])
        sample_f2_hv = self.feat_mfcc_kernel(feat[MFCC_START : MFCC_START + NUM_MFCC])
        sample_f3_hv = self.feat_fft_mean_kernel(feat[FFT_MEAN_START : FFT_MEAN_START + NUM_FFT_MEAN])
        sample_f4_hv = self.feat_fft_max_kernel(feat[FFT_MAX_START : FFT_MAX_START + NUM_FFT_MAX])
        sample_hv = sample_hv * sample_f1_hv * sample_f2_hv * sample_f3_hv * sample_f4_hv
        # Apply activation function
        sample_hv = torchhd.hard_quantize(sample_hv)  # torch.sin(sample_hv)
        return sample_hv.flatten()
