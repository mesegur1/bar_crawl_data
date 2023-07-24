import torch
import numpy as np
import torchhd
from torchhd import embeddings


# HDC Encoder for Bar Crawl Data
class HdcLevelEncoder(torch.nn.Module):
    def __init__(self, levels: int, timestamps: int, out_dimension: int):
        super(HdcLevelEncoder, self).__init__()

        self.timestamps = embeddings.Level(
            timestamps, out_dimension, dtype=torch.float64, low=0, high=timestamps
        )

        # Time domain features
        self.signal_level_x = embeddings.Level(
            levels, out_dimension, dtype=torch.float64
        )
        self.signal_level_y = embeddings.Level(
            levels, out_dimension, dtype=torch.float64
        )
        self.signal_level_z = embeddings.Level(
            levels, out_dimension, dtype=torch.float64
        )

    # Encode window of feature vectors (x,y,z)
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Adjust time
        input[:, 0] = input[:, 0] - input[0, 0]
        # Get level hypervectors for x, y, z samples
        x_signal = input[:, 1]
        y_signal = input[:, 2]
        z_signal = input[:, 3]
        x_levels = self.signal_level_x(x_signal)
        y_levels = self.signal_level_y(y_signal)
        z_levels = self.signal_level_z(z_signal)
        # Get time hypervectors
        times = self.timestamps(input[:, 0])
        # Bind time sequence for x, y, z samples
        sample_hvs = (x_levels * y_levels * z_levels ) * times
        sample_hv = torchhd.multiset(sample_hvs)
        # Apply activation function
        sample_hv = torch.tanh(sample_hv)
        return sample_hv
