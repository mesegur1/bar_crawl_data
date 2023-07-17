import torch
import numpy as np
import torchhd
from torchhd import embeddings

SIGNAL_X_MIN = -3
SIGNAL_X_MAX = 3
SIGNAL_Y_MIN = -3
SIGNAL_Y_MAX = 3
SIGNAL_Z_MIN = -3
SIGNAL_Z_MAX = 3

MAG_SIGNAL_MIN = -3
MAG_SIGNAL_MAX = 3


# HDC Encoder for Bar Crawl Data
class HdcLevelEncoder(torch.nn.Module):
    def __init__(self, levels: int, timestamps: int, out_dimension: int):
        super(HdcLevelEncoder, self).__init__()

        self.signal_level_x = embeddings.Level(
            levels,
            out_dimension,
            dtype=torch.float64,
            low=SIGNAL_X_MIN,
            high=SIGNAL_X_MAX,
        )
        self.signal_level_y = embeddings.Level(
            levels,
            out_dimension,
            dtype=torch.float64,
            low=SIGNAL_Y_MIN,
            high=SIGNAL_Y_MAX,
        )
        self.signal_level_z = embeddings.Level(
            levels,
            out_dimension,
            dtype=torch.float64,
            low=SIGNAL_Z_MIN,
            high=SIGNAL_Z_MAX,
        )
        self.timestamps = embeddings.Level(
            timestamps, out_dimension, dtype=torch.float64, low=0, high=timestamps
        )

        self.signal_level_mag = embeddings.Level(
            levels,
            out_dimension,
            dtype=torch.float64,
            low=MAG_SIGNAL_MIN,
            high=MAG_SIGNAL_MAX,
        )

        self.signal_level_x_jerk = embeddings.Level(
            levels,
            out_dimension,
            dtype=torch.float64,
            low=SIGNAL_X_MIN,
            high=SIGNAL_X_MAX,
        )
        self.signal_level_y_jerk = embeddings.Level(
            levels,
            out_dimension,
            dtype=torch.float64,
            low=SIGNAL_Y_MIN,
            high=SIGNAL_Y_MAX,
        )
        self.signal_level_z_jerk = embeddings.Level(
            levels,
            out_dimension,
            dtype=torch.float64,
            low=SIGNAL_Z_MIN,
            high=SIGNAL_Z_MAX,
        )

        self.signal_level_mag_jerk = embeddings.Level(
            levels,
            out_dimension,
            dtype=torch.float64,
            low=MAG_SIGNAL_MIN,
            high=MAG_SIGNAL_MAX,
        )

    # Calculate magnitudes of signals
    def calc_mags(self, xyz: torch.Tensor):
        sq = torch.square(xyz)
        # Sum of squares of each component
        mags = torch.sqrt(torch.sum(sq, dim=1))
        return mags  # Magnitude signal samples

    # Calculate jerk of signals
    def calc_jerk(self, txyz: torch.Tensor):
        n = txyz.shape[0]
        jerks = torch.zeros((1, 3), device="cuda")
        for i in range(n - 1):
            t0 = txyz[i, 0]
            t1 = txyz[i + 1, 0]
            signals0 = txyz[i, 1:]
            signals1 = txyz[i + 1, 1:]
            jerk = (signals1 - signals0) / (t1 - t0).item()
            jerk = jerk.unsqueeze(0)
            jerks = torch.cat((jerks, jerk), dim=0)
        return jerks

    # Encode window of feature vectors (x,y,z)
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Adjust time
        input[:, 0] = input[:, 0] - input[0, 0]
        # Get level hypervectors for x, y, z samples
        x_signal = torch.clamp(input[:, 1], min=SIGNAL_X_MIN, max=SIGNAL_X_MAX)
        y_signal = torch.clamp(input[:, 2], min=SIGNAL_Y_MIN, max=SIGNAL_Y_MAX)
        z_signal = torch.clamp(input[:, 3], min=SIGNAL_Z_MIN, max=SIGNAL_Z_MAX)
        x_levels = self.signal_level_x(x_signal)
        y_levels = self.signal_level_y(y_signal)
        z_levels = self.signal_level_z(z_signal)
        mag_levels = self.signal_level_mag(self.calc_mags(input[:, 1:]))
        # Get level hypervectors for jerk of x, y, z samples
        jerk = self.calc_jerk(input)
        x_jerk_signal = torch.clamp(jerk[:, 0], min=SIGNAL_X_MIN, max=SIGNAL_X_MAX)
        y_jerk_signal = torch.clamp(jerk[:, 1], min=SIGNAL_X_MIN, max=SIGNAL_X_MAX)
        z_jerk_signal = torch.clamp(jerk[:, 2], min=SIGNAL_X_MIN, max=SIGNAL_X_MAX)
        x_jerk_levels = self.signal_level_x_jerk(x_jerk_signal)
        y_jerk_levels = self.signal_level_y_jerk(y_jerk_signal)
        z_jerk_levels = self.signal_level_z_jerk(z_jerk_signal)
        jerk_mag_levels = self.signal_level_mag_jerk(self.calc_mags(jerk))

        # Get time hypervectors
        times = self.timestamps(input[:, 0])
        # Bind time sequence for x, y, z samples
        # sample_hvs = x_levels * y_levels * z_levels * times
        sample_hvs = (
            (x_levels + y_levels + z_levels) * mag_levels
            + (x_jerk_levels + y_jerk_levels + z_jerk_levels) * jerk_mag_levels
        ) * times
        sample_hv = torchhd.multiset(sample_hvs)
        # Apply activation function
        sample_hv = torch.tanh(sample_hv)
        return sample_hv
