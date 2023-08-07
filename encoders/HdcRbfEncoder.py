import torch
import torch.nn.functional as F
import numpy as np
import torchhd
from torchhd_custom import embeddings

NUM_CHANNEL = 3
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
class HdcRbfEncoder(torch.nn.Module):
    def __init__(
        self, timestamps: int, out_dimension: int, use_back_prop: bool = False
    ):
        super(HdcRbfEncoder, self).__init__()

        self.timestamps = timestamps
        # RBF Kernel Trick
        if use_back_prop == True:
            self.kernel = embeddings.HyperTangent(
                timestamps * NUM_CHANNEL, out_dimension, dtype=torch.float64
            )
        else:
            self.kernel = embeddings.Sinusoid(
                timestamps * NUM_CHANNEL, out_dimension, dtype=torch.float64
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

    # Encode window of feature vectors (x,y,z) and feature vectors (f,)
    def forward(self, input: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        # Get features from x, y, z samples
        window = input.size(0)  # sample count
        if window < self.timestamps:
            # Pad the inputs to required length
            padding = self.timestamps - window
            x_signal = F.pad(
                input=input[:, 1], pad=(0, padding), mode="constant", value=0
            )
            y_signal = F.pad(
                input=input[:, 2], pad=(0, padding), mode="constant", value=0
            )
            z_signal = F.pad(
                input=input[:, 3], pad=(0, padding), mode="constant", value=0
            )
        else:
            x_signal = input[:, 1]
            y_signal = input[:, 2]
            z_signal = input[:, 3]
        accel_features = torch.cat((x_signal, y_signal, z_signal))
        # Use kernel encoder
        sample_hv = self.kernel(accel_features)
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
        return sample_hv.squeeze(0)
