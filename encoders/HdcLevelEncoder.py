import torch
import numpy as np
import torchhd
from torchhd import embeddings

SIGNAL_X_MIN = -5
SIGNAL_X_MAX = 5
SIGNAL_Y_MIN = -5
SIGNAL_Y_MAX = 5
SIGNAL_Z_MIN = -5
SIGNAL_Z_MAX = 5

MAG_SIGNAL_MIN = -10
MAG_SIGNAL_MAX = 10

ENERGY_SIGNAL_MIN = -10
ENERGY_SIGNAL_MAX = 10


# HDC Encoder for Bar Crawl Data
class HdcLevelEncoder(torch.nn.Module):
    def __init__(self, levels: int, timestamps: int, out_dimension: int):
        super(HdcLevelEncoder, self).__init__()

        # Raw Signal Level Bases
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

        # Raw Signal Magnitude Bases
        self.signal_level_mag = embeddings.Level(
            levels,
            out_dimension,
            dtype=torch.float64,
            low=MAG_SIGNAL_MIN,
            high=MAG_SIGNAL_MAX,
        )

        # Jerk Level HV Bases
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

        # Jerk Signal Magnitude Bases
        self.signal_level_mag_jerk = embeddings.Level(
            levels,
            out_dimension,
            dtype=torch.float64,
            low=MAG_SIGNAL_MIN,
            high=MAG_SIGNAL_MAX,
        )

        # Energy Level Bases
        self.signal_level_energy_x = embeddings.Level(
            levels,
            out_dimension,
            dtype=torch.float64,
            low=ENERGY_SIGNAL_MIN,
            high=ENERGY_SIGNAL_MAX,
        )
        self.signal_level_energy_y = embeddings.Level(
            levels,
            out_dimension,
            dtype=torch.float64,
            low=ENERGY_SIGNAL_MIN,
            high=ENERGY_SIGNAL_MAX,
        )
        self.signal_level_energy_z = embeddings.Level(
            levels,
            out_dimension,
            dtype=torch.float64,
            low=ENERGY_SIGNAL_MIN,
            high=ENERGY_SIGNAL_MAX,
        )

        # Time Basis
        self.timestamps = embeddings.Level(
            timestamps, out_dimension, dtype=torch.float64, low=0, high=timestamps
        )

    # Calculate magnitudes of signals
    def calc_mags(self, xyz: torch.Tensor):
        sq = torch.square(xyz)
        # Sum of squares of each component
        mags = torch.sqrt(torch.sum(sq, dim=1))
        return mags  # Magnitude signal samples

    # Calculate energy of signals
    def calc_energy(self, xyz: torch.Tensor):
        n = xyz.shape[0]
        sq = torch.square(xyz)
        energy = torch.sum(sq, dim=0) / max(n, 1)
        return energy

    # Calculate jerk of signals
    def calc_jerk(self, txyz: torch.Tensor):
        n = txyz.shape[0]
        jerks = torch.zeros((1, 3), device="cuda", dtype=torch.float64)
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
        mags = self.calc_mags(input[:, 1:])
        mag_levels = self.signal_level_mag(mags)
        # Get level hypervectors for jerk of x, y, z samples
        jerk = self.calc_jerk(input)
        x_jerk_signal = torch.clamp(jerk[:, 0], min=SIGNAL_X_MIN, max=SIGNAL_X_MAX)
        y_jerk_signal = torch.clamp(jerk[:, 1], min=SIGNAL_X_MIN, max=SIGNAL_X_MAX)
        z_jerk_signal = torch.clamp(jerk[:, 2], min=SIGNAL_X_MIN, max=SIGNAL_X_MAX)
        x_jerk_levels = self.signal_level_x_jerk(x_jerk_signal)
        y_jerk_levels = self.signal_level_y_jerk(y_jerk_signal)
        z_jerk_levels = self.signal_level_z_jerk(z_jerk_signal)
        jerk_mags = self.calc_mags(jerk)
        jerk_mag_levels = self.signal_level_mag_jerk(jerk_mags)
        # Get energy
        energy = self.calc_energy(input[:, 1:])
        x_energy_level = self.signal_level_energy_x(energy[0])
        y_energy_level = self.signal_level_energy_y(energy[1])
        z_energy_level = self.signal_level_energy_z(energy[2])
        # Get time hypervectors
        times = self.timestamps(input[:, 0])
        # Bind time sequence for x, y, z samples
        sample_hvs = (
            x_levels
            * y_levels
            * z_levels
            * x_jerk_levels
            * y_jerk_levels
            * z_jerk_levels
            * mag_levels
            * jerk_mag_levels
        ) * times
        sample_hv = torchhd.multiset(sample_hvs)
        # Bind non-timewindowed features
        sample_hv *= x_energy_level * y_energy_level * z_energy_level
        # Apply activation function
        sample_hv = torch.tanh(sample_hv)
        return sample_hv
