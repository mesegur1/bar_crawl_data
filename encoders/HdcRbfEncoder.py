import torch
import torch.nn.functional as F
import numpy as np
import torchhd
from torchhd_custom import embeddings

NUM_CHANNEL = 3
NUM_RMS = 3
NUM_MFCC = 3
NUM_FFT_MEAN = 3
NUM_FFT_MAX = 3
RMS_START = 0
MFCC_START = RMS_START + NUM_RMS
FFT_MEAN_START = MFCC_START + NUM_MFCC
FFT_MAX_START = FFT_MEAN_START + NUM_FFT_MEAN

# HDC Encoder for Bar Crawl Data
class HdcRbfEncoder(torch.nn.Module):
    def __init__(self, timestamps: int, out_dimension: int, use_tanh: bool = True):
        super(HdcRbfEncoder, self).__init__()

        self.timestamps = timestamps
        # RBF Kernel Trick
        if use_tanh == True:
            self.kernel = embeddings.HyperTangent(
                timestamps * NUM_CHANNEL, out_dimension, dtype=torch.float64
            )
        else:
            self.kernel = embeddings.Sinusoid(
                timestamps * NUM_CHANNEL, out_dimension, dtype=torch.float64
            )
        self.feat_rms_kernel = embeddings.Sinusoid(NUM_RMS, out_dimension, dtype=torch.float64)
        self.feat_mfcc_kernel = embeddings.Sinusoid(NUM_MFCC, out_dimension, dtype=torch.float64)
        self.feat_fft_mean_kernel = embeddings.Sinusoid(NUM_FFT_MEAN, out_dimension, dtype=torch.float64)
        self.feat_fft_max_kernel = embeddings.Sinusoid(NUM_FFT_MAX, out_dimension, dtype=torch.float64)

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
        sample_f3_hv = self.feat_fft_mean_kernel(feat[FFT_MEAN_START : FFT_MEAN_START + NUM_FFT_MEAN])
        sample_f4_hv = self.feat_fft_max_kernel(feat[FFT_MAX_START : FFT_MAX_START + NUM_FFT_MAX])
        sample_hv = sample_hv * sample_f1_hv * sample_f2_hv * sample_f3_hv * sample_f4_hv
        # Apply activation function
        sample_hv = torchhd.hard_quantize(sample_hv)  # torch.sin(sample_hv)
        return sample_hv.squeeze(0)
