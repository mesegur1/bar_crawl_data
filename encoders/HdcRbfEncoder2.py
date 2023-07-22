import torch
import torch.nn.functional as F
import numpy as np
import torchhd
from torchhd_custom import embeddings

NUM_CHANNEL = 9


# HDC Encoder for Bar Crawl Data
class HdcRbfEncoder2(torch.nn.Module):
    def __init__(
        self, features: int, timestamps: int, out_dimension: int, use_tanh: bool = True
    ):
        super(HdcRbfEncoder2, self).__init__()

        self.timestamps = timestamps
        # RBF Kernel Trick
        if use_tanh == True:
            self.kernel1 = embeddings.HyperTangent(
                timestamps * features, out_dimension, dtype=torch.float64
            )
        else:
            self.kernel1 = embeddings.Sinusoid(
                timestamps * features, out_dimension, dtype=torch.float64
            )

    # Encode window of feature vectors
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Flatten in column-major order
        signals = input.transpose(1, 0).flatten()
        window = input.size(0)  # sample count
        if window < self.timestamps:
            # Pad the inputs to required length
            padding = self.timestamps - window
            signals = F.pad(input=signals, pad=(padding, 0), mode="constant", value=0)
        # Use kernel encoder
        sample_hv = self.kernel1(signals)
        sample_hv = torch.tanh(sample_hv)
        return sample_hv.squeeze(0)
