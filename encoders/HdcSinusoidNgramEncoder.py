import torch
import torch.nn.functional as F
import numpy as np
import torchhd
from torchhd_custom import embeddings

NUM_CHANNEL = 3
NGRAM_SIZE = 3


# HDC Encoder for Bar Crawl Data
class HdcSinusoidNgramEncoder(torch.nn.Module):
    def __init__(self, out_dimension: int):
        super(HdcSinusoidNgramEncoder, self).__init__()

        self.kernel = embeddings.Sinusoid(
            NUM_CHANNEL, out_dimension, dtype=torch.float64
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Get features from x, y, z samples
        signals = input[:, 1:]
        # Use kernel encoder
        sample_hv = self.kernel(signals)
        # Perform ngram statistics
        sample_hv = torchhd.ngrams(sample_hv, NGRAM_SIZE)
        return sample_hv.squeeze(0)
