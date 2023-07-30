import torch
import torch.nn.functional as F
import numpy as np
import torchhd
from torchhd_custom import embeddings
from torchhd_custom import functional

NUM_CHANNEL = 4
NGRAM_SIZE = 4


# HDC Encoder for Bar Crawl Data
class HdcGenericEncoder(torch.nn.Module):
    def __init__(self, levels: int, out_dimension: int):
        super(HdcGenericEncoder, self).__init__()

        self.keys = embeddings.Random(NUM_CHANNEL, out_dimension, dtype=torch.float64)
        self.embed = embeddings.Level(levels, out_dimension, dtype=torch.float64)
        self.feat_kernel = embeddings.Sinusoid(6, out_dimension, dtype=torch.float64)

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
        sample_f_hv = self.feat_kernel(feat)
        sample_hv = sample_hv * sample_f_hv
        # Apply activation function
        sample_hv = torchhd.hard_quantize(sample_hv)
        return sample_hv.flatten()
