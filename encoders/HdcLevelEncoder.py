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
        self.signal_level_mag = embeddings.Level(
            levels, out_dimension, dtype=torch.float64
        )
        self.signal_level_energy = embeddings.Level(
            levels, out_dimension, dtype=torch.float64
        )

        # Frequency domain features
        self.signal_level_x_fft = embeddings.Level(
            levels, out_dimension, dtype=torch.float64
        )
        self.signal_level_y_fft = embeddings.Level(
            levels, out_dimension, dtype=torch.float64
        )
        self.signal_level_z_fft = embeddings.Level(
            levels, out_dimension, dtype=torch.float64
        )
        self.signal_level_mag_fft = embeddings.Level(
            levels, out_dimension, dtype=torch.float64
        )
        self.signal_level_energy_fft = embeddings.Level(
            levels, out_dimension, dtype=torch.float64
        )
        self.signal_level_x_fft_i = embeddings.Level(
            levels, out_dimension, dtype=torch.float64
        )
        self.signal_level_y_fft_i = embeddings.Level(
            levels, out_dimension, dtype=torch.float64
        )
        self.signal_level_z_fft_i = embeddings.Level(
            levels, out_dimension, dtype=torch.float64
        )
        self.signal_level_mag_fft_i = embeddings.Level(
            levels, out_dimension, dtype=torch.float64
        )
        self.signal_level_energy_fft_i = embeddings.Level(
            levels, out_dimension, dtype=torch.float64
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
        energy = torch.sum(sq, dim=1) / max(n, 1)
        return energy

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
        mag_levels = self.signal_level_mag(self.calc_mags(input[:, 1:]))
        energy_levels = self.signal_level_energy(self.calc_energy(input[:, 1:]))
        # Get level hypervectors for FFT x, y, z samples
        fft_signals = torch.fft.fft(input[:, 1:], dim=0)
        x_fft_signal = fft_signals[:, 0].real
        y_fft_signal = fft_signals[:, 1].real
        z_fft_signal = fft_signals[:, 2].real
        x_fft_i_signal = fft_signals[:, 0].imag
        y_fft_i_signal = fft_signals[:, 1].imag
        z_fft_i_signal = fft_signals[:, 2].imag
        x_fft_levels = self.signal_level_x_fft(x_fft_signal)
        y_fft_levels = self.signal_level_y_fft(y_fft_signal)
        z_fft_levels = self.signal_level_z_fft(z_fft_signal)
        x_fft_i_levels = self.signal_level_x_fft_i(x_fft_i_signal)
        y_fft_i_levels = self.signal_level_y_fft_i(y_fft_i_signal)
        z_fft_i_levels = self.signal_level_z_fft_i(z_fft_i_signal)
        fft_mag_levels = self.signal_level_mag_fft(self.calc_mags(fft_signals.real))
        fft_mag_i_levels = self.signal_level_mag_fft_i(self.calc_mags(fft_signals.imag))
        energy_fft_levels = self.signal_level_energy_fft(
            self.calc_energy(fft_signals.real)
        )
        energy_fft_i_levels = self.signal_level_energy_fft_i(
            self.calc_energy(fft_signals.imag)
        )
        # Get time hypervectors
        times = self.timestamps(input[:, 0])
        # Bind time sequence for x, y, z samples
        sample_hvs = (
            (x_levels * y_levels * z_levels + mag_levels + energy_levels)
            + (
                x_fft_levels
                * y_fft_levels
                * z_fft_levels
                * x_fft_i_levels
                * y_fft_i_levels
                * z_fft_i_levels
                + fft_mag_levels * fft_mag_i_levels
                + energy_fft_levels * energy_fft_i_levels
            )
        ) * times
        sample_hv = torchhd.multiset(sample_hvs)
        # Apply activation function
        sample_hv = torch.tanh(sample_hv)
        return sample_hv
