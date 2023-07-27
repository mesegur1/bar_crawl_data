import torch
import torch.nn.functional as F
import numpy as np
import torchhd
from torchhd_custom import embeddings

NUM_CHANNEL = 4
NGRAM_SIZE = 4


# HDC Encoder for Bar Crawl Data
class HdcSinusoidNgramEncoder(torch.nn.Module):
    def __init__(self, out_dimension: int):
        super(HdcSinusoidNgramEncoder, self).__init__()

        self.kernel = embeddings.Sinusoid(
            NUM_CHANNEL, out_dimension, dtype=torch.float64
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Get features from t, x, y, z samples
        signals = input
        # Use kernel encoder
        sample_hv = self.kernel(signals)
        # Perform ngram statistics
        sample_hv = torchhd.ngrams(sample_hv, NGRAM_SIZE)
        # Apply activation function
        sample_hv = torch.sin(sample_hv)
        return sample_hv.squeeze(0)
