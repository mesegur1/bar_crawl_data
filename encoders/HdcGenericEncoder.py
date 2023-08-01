import torch
import torch.nn.functional as F
import numpy as np
import torchhd
from torchhd_custom import embeddings
from torchhd_custom import functional

NUM_CHANNEL = 4
NGRAM_SIZE = 3
NUM_RMS = 3
NUM_MFCC = 3
NUM_FFT_MEAN = 3
NUM_FFT_MAX = 3
NUM_FFT_VAR = 3
RMS_START = 0
MFCC_START = RMS_START + NUM_RMS
FFT_MEAN_START = MFCC_START + NUM_MFCC
FFT_MAX_START = FFT_MEAN_START + NUM_FFT_MEAN
FFT_VAR_START = FFT_MAX_START + NUM_FFT_MAX


# HDC Encoder for Bar Crawl Data
class HdcGenericEncoder(torch.nn.Module):
    def __init__(self, levels: int, out_dimension: int):
        super(HdcGenericEncoder, self).__init__()

        self.keys = embeddings.Random(NUM_CHANNEL, out_dimension, dtype=torch.float64)
        self.embed = embeddings.Level(levels, out_dimension, dtype=torch.float64)
        self.feat_rms_kernel = embeddings.Sinusoid(NUM_RMS, out_dimension, dtype=torch.float64)
        self.feat_mfcc_kernel = embeddings.Sinusoid(NUM_MFCC, out_dimension, dtype=torch.float64)
        self.feat_fft_mean_kernel = embeddings.Sinusoid(NUM_FFT_MEAN, out_dimension, dtype=torch.float64)
        self.feat_fft_max_kernel = embeddings.Sinusoid(NUM_FFT_MAX, out_dimension, dtype=torch.float64)
        self.feat_fft_var_kernel = embeddings.Sinusoid(NUM_FFT_VAR, out_dimension, dtype=torch.float64)

    # Encode window of feature vectors (x,y,z) and feature vectors (f,)
    def forward(self, input: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        # Get features from t, x, y, z samples
        signals = input
        # Use generic encoder
        sample_hvs = functional.generic(
            self.keys.weight, self.embed(signals), NGRAM_SIZE
        )
        sample_hv = torchhd.multiset(sample_hvs)
        # Encode calculated features
        sample_f1_hv = self.feat_rms_kernel(feat[RMS_START : RMS_START + NUM_RMS])
        sample_f2_hv = self.feat_mfcc_kernel(feat[MFCC_START : MFCC_START + NUM_MFCC])
        sample_f3_hv = self.feat_fft_mean_kernel(feat[FFT_MEAN_START : FFT_MEAN_START + NUM_FFT_MEAN])
        sample_f4_hv = self.feat_fft_max_kernel(feat[FFT_MAX_START : FFT_MAX_START + NUM_FFT_MAX])
        sample_f5_hv = self.feat_fft_var_kernel(feat[FFT_VAR_START : FFT_VAR_START + NUM_FFT_VAR])
        sample_hv = (
            sample_hv
            * sample_f1_hv
            * sample_f2_hv
            * (sample_f3_hv + sample_f4_hv + sample_f5_hv)
        )
        # Apply activation function
        sample_hv = torchhd.hard_quantize(sample_hv)
        return sample_hv.flatten()
