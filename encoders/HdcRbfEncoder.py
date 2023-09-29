import torch
import torch.nn.functional as F
import numpy as np
import torchhd
from torchhd_custom import embeddings

NUM_CHANNEL = 3


# HDC Encoder for Bar Crawl Data
class HdcRbfEncoder(torch.nn.Module):
    def __init__(
        self, timestamps: int, out_dimension: int, use_back_prop: bool = False
    ):
        super(HdcRbfEncoder, self).__init__()

        self.timestamps = timestamps
        # RBF Kernel Trick
        if use_back_prop == True:
            self.kernel = embeddings.HyperTangent(
                timestamps * NUM_CHANNEL, out_dimension, dtype=torch.float64
            )
        else:
            self.kernel = embeddings.Sinusoid(
                timestamps * NUM_CHANNEL, out_dimension, dtype=torch.float64
            )
        self.feat_encode = embeddings.DensityFlocet(135, out_dimension, dtype=torch.float64)

    # Encode window of raw features (t,x,y,z) and extracted feature vectors (f,)
    def forward(self, signals: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        # Get features from x, y, z samples
        window = signals.size(0)  # sample count
        if window < self.timestamps:
            # Pad the inputs to required length
            padding = self.timestamps - window
            x_signal = F.pad(
                input=signals[:, 1], pad=(0, padding), mode="constant", value=0
            )
            y_signal = F.pad(
                input=signals[:, 2], pad=(0, padding), mode="constant", value=0
            )
            z_signal = F.pad(
                input=signals[:, 3], pad=(0, padding), mode="constant", value=0
            )
        else:
            x_signal = signals[:, 1]
            y_signal = signals[:, 2]
            z_signal = signals[:, 3]
        accel_features = torch.cat((x_signal, y_signal, z_signal))
        # Use kernel encoder
        sample_hv = torchhd.hard_quantize(self.kernel(accel_features))
        feat_hv = torchhd.hard_quantize(self.feat_encode(feat))

        combined_hv = sample_hv * feat_hv

        # Apply activation function
        #combined_hv = torchhd.hard_quantize(combined_hv)
        return combined_hv.flatten()
