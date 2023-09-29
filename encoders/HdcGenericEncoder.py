import torch
import torch.nn.functional as F
import numpy as np
import torchhd
from torchhd_custom import embeddings
from torchhd_custom import functional

NUM_CHANNEL = 3
NGRAM_SIZE = 3

class HashTableEncoder(torch.nn.Module):
    def __init__(self, channels : int, levels : int, out_dimension : int):
        super(HashTableEncoder, self).__init__()

        self.keys = embeddings.Random(channels, out_dimension, dtype=torch.float64)
        self.embed = embeddings.Level(levels, out_dimension, dtype=torch.float64)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        sample_hv = torchhd.hash_table(self.keys.weight, self.embed(x))
        return sample_hv

# HDC Encoder for Bar Crawl Data
class HdcGenericEncoder(torch.nn.Module):
    def __init__(self, levels: int, out_dimension: int):
        super(HdcGenericEncoder, self).__init__()

        #Embeddings for raw data
        self.keys = embeddings.Random(NUM_CHANNEL, out_dimension, dtype=torch.float64)
        self.embed = embeddings.Level(10000, out_dimension, dtype=torch.float64)
        self.feat_encode = embeddings.DensityFlocet(135, out_dimension, dtype=torch.float64)

    # Encode window of raw features (t,x,y,z) and extracted feature vectors (f,)
    def forward(self, signals: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        # Use generic encoder
        sample_hvs = functional.generic(
            self.keys.weight, self.embed(signals[:, 1:]*10), NGRAM_SIZE
        )
        sample_hv = torchhd.hard_quantize(torchhd.multiset(sample_hvs))
        feat_hv = torchhd.hard_quantize(self.feat_encode(feat))

        combined_hv = sample_hv * feat_hv

        # Apply activation function
        #combined_hv = torchhd.hard_quantize(combined_hv)
        return combined_hv.flatten()
