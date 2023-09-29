import torch
import torch.nn.functional as F
import numpy as np
import torchhd
from torchhd_custom import embeddings

NUM_CHANNEL = 4
NGRAM_SIZE = 3

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
class HdcSinusoidNgramEncoder(torch.nn.Module):
    def __init__(self, out_dimension: int):
        super(HdcSinusoidNgramEncoder, self).__init__()

        self.kernel = embeddings.Sinusoid(
            NUM_CHANNEL, out_dimension, dtype=torch.float64
        )

        #Embeddings for extracted feature data
        self.feat_encode = embeddings.DensityFlocet(135, out_dimension, dtype=torch.float64)

    # Encode window of feature vectors (x,y,z) and feature vectors (f,)
    def forward(self, input: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        # Get features from t, x, y, z samples
        signals = input
        # Use kernel encoder
        sample_hv = self.kernel(signals)
        # Perform ngram statistics
        sample_hv = torchhd.hard_quantize(torchhd.ngrams(sample_hv, NGRAM_SIZE))

        # Encode calculated features
        feat_hv = torchhd.hard_quantize(self.feat_encode(feat))

        combined_hv = sample_hv * feat_hv

        # Apply activation function
        #combined_hv = torchhd.hard_quantize(combined_hv)
        return combined_hv.flatten()
