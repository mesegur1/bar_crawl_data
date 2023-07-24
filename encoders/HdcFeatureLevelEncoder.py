import torch
import numpy as np
import torchhd
from torchhd import embeddings

# HDC Feature Encoder for Bar Crawl Data
class HdcFeatureLevelEncoder(torch.nn.Module):
    def __init__(self, levels: int, out_dimension: int):
        super(HdcFeatureLevelEncoder, self).__init__()

        self.levels = embeddings.Level(
            levels, out_dimension, dtype=torch.float64,
        )

    # Encode feature vectors
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        feature_hvs = self.levels(input)
        sample_hv = torchhd.multiset(feature_hvs)
        # Apply activation function
        sample_hv = torch.tanh(sample_hv)
        return sample_hv
