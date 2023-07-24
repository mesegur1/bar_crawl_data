import torch
import torch.nn.functional as F
import numpy as np
import torchhd
from torchhd_custom import embeddings


# HDC Encoder for Bar Crawl Data
class HdcFeatureRbfEncoder(torch.nn.Module):
    def __init__(self, num_features: int, out_dimension: int, use_tanh: bool = True):
        super(HdcFeatureRbfEncoder, self).__init__()

        self.num_features = num_features
        # RBF Kernel Trick
        if use_tanh == True:
            self.kernel = embeddings.HyperTangent(
                num_features, out_dimension, dtype=torch.float64
            )
        else:
            self.kernel = embeddings.Sinusoid(
                num_features, out_dimension, dtype=torch.float64
            )

    # Encode window of feature vectors (x,y,z)
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Use kernel encoder
        sample_hv = self.kernel(input)
        sample_hv = torch.tanh(sample_hv)
        return sample_hv.squeeze(0)