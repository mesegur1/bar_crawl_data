import torch
import numpy as np
import torchhd
from torchhd_custom import embeddings


#Key value pair encoder for encoding a feature
class KeyValuePairEncoder(torch.nn.Module):
    def __init__(self, num_channels: int, levels: int, out_dimension: int):
        super(KeyValuePairEncoder, self).__init__()
        self.keys = embeddings.Random(num_channels, out_dimension, dtype=torch.float64)
        self.embed = embeddings.Level(levels, out_dimension, dtype=torch.float64)

    # Encode window of feature vectors (x,y,z)
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        sample_hv = torchhd.bind(self.keys.weight, self.embed(input))
        sample_hv = torchhd.multiset(sample_hv)
        return torchhd.hard_quantize(sample_hv)
