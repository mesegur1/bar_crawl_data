import torch
import torch.nn.functional as F
import numpy as np
import torchhd
from torchhd_custom import embeddings

NUM_CHANNEL = 9


# HDC Encoder for Bar Crawl Data
class HdcRbfEncoder(torch.nn.Module):
    def __init__(self, timestamps: int, out_dimension: int, use_tanh: bool = True):
        super(HdcRbfEncoder, self).__init__()

        self.timestamps = timestamps
        # RBF Kernel Trick
        if use_tanh == True:
            self.kernel1 = embeddings.HyperTangent(
                timestamps * 3, out_dimension, dtype=torch.float64
            )
            self.kernel2 = embeddings.HyperTangent(
                timestamps * 3, out_dimension, dtype=torch.float64
            )
            self.kernel3 = embeddings.HyperTangent(
                timestamps * 3, out_dimension, dtype=torch.float64
            )
        else:
            self.kernel1 = embeddings.Sinusoid(
                timestamps * 3, out_dimension, dtype=torch.float64
            )
            self.kernel2 = embeddings.Sinusoid(
                timestamps * 3, out_dimension, dtype=torch.float64
            )
            self.kernel3 = embeddings.Sinusoid(
                timestamps * 3, out_dimension, dtype=torch.float64
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
        # Get FFT Signals
        fft_signals = torch.fft.fft(input[:, 1:], dim=0)
        # Get features from x, y, z samples
        window = input.size(0)  # sample count
        if window < self.timestamps:
            # Pad the inputs to required length
            padding = self.timestamps - window
            x_signal = F.pad(
                input=input[:, 1], pad=(0, padding), mode="constant", value=0
            )
            y_signal = F.pad(
                input=input[:, 2], pad=(0, padding), mode="constant", value=0
            )
            z_signal = F.pad(
                input=input[:, 3], pad=(0, padding), mode="constant", value=0
            )
            x_fft_signal = F.pad(
                input=fft_signals[:, 0].real, pad=(0, padding), mode="constant", value=0
            )
            y_fft_signal = F.pad(
                input=fft_signals[:, 1].real, pad=(0, padding), mode="constant", value=0
            )
            z_fft_signal = F.pad(
                input=fft_signals[:, 2].real, pad=(0, padding), mode="constant", value=0
            )
            x_fft_i_signal = F.pad(
                input=fft_signals[:, 0].imag, pad=(0, padding), mode="constant", value=0
            )
            y_fft_i_signal = F.pad(
                input=fft_signals[:, 1].imag, pad=(0, padding), mode="constant", value=0
            )
            z_fft_i_signal = F.pad(
                input=fft_signals[:, 2].imag, pad=(0, padding), mode="constant", value=0
            )
        else:
            x_signal = input[:, 1]
            y_signal = input[:, 2]
            z_signal = input[:, 3]
            x_fft_signal = fft_signals[:, 0].real
            y_fft_signal = fft_signals[:, 1].real
            z_fft_signal = fft_signals[:, 2].real
            x_fft_i_signal = fft_signals[:, 0].imag
            y_fft_i_signal = fft_signals[:, 1].imag
            z_fft_i_signal = fft_signals[:, 2].imag
        features1 = torch.cat((x_signal, y_signal, z_signal))
        features2 = torch.cat((x_fft_signal, y_fft_signal, z_fft_signal))
        features3 = torch.cat((x_fft_i_signal, y_fft_i_signal, z_fft_i_signal))
        # Use kernel encoder
        sample_hv1 = self.kernel1(features1)
        sample_hv2 = self.kernel2(features2)
        sample_hv3 = self.kernel3(features3)
        sample_hv = sample_hv1 + sample_hv2 * sample_hv3
        sample_hv = torch.tanh(sample_hv)
        return sample_hv.squeeze(0)
