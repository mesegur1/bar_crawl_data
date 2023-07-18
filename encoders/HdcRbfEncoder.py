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
                timestamps * 2, out_dimension, dtype=torch.float64
            )
            self.kernel4 = embeddings.HyperTangent(
                3, out_dimension, dtype=torch.float64
            )
        else:
            self.kernel1 = embeddings.Sinusoid(
                timestamps * 3, out_dimension, dtype=torch.float64
            )
            self.kernel2 = embeddings.Sinusoid(
                timestamps * 3, out_dimension, dtype=torch.float64
            )
            self.kernel3 = embeddings.Sinusoid(
                timestamps * 2, out_dimension, dtype=torch.float64
            )
            self.kernel4 = embeddings.Sinusoid(3, out_dimension, dtype=torch.float64)

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
        # Get features from x, y, z samples
        mags = self.calc_mags(input[:, 1:])
        energy = self.calc_energy(input[:, 1:])
        jerks = self.calc_jerk(input)
        jerk_mags = self.calc_mags(jerks)
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
            x_jerk = F.pad(
                input=jerks[:, 0], pad=(0, padding), mode="constant", value=0
            )
            y_jerk = F.pad(
                input=jerks[:, 1], pad=(0, padding), mode="constant", value=0
            )
            z_jerk = F.pad(
                input=jerks[:, 2], pad=(0, padding), mode="constant", value=0
            )
            mags = F.pad(input=mags, pad=(0, padding), mode="constant", value=0)
            jerk_mags = F.pad(
                input=jerk_mags, pad=(0, padding), mode="constant", value=0
            )
        else:
            x_signal = input[:, 1]
            y_signal = input[:, 2]
            z_signal = input[:, 3]
            x_jerk = jerks[:, 0]
            y_jerk = jerks[:, 1]
            z_jerk = jerks[:, 2]
        features1 = torch.cat((x_signal, y_signal, z_signal))
        features2 = torch.cat((x_jerk, y_jerk, z_jerk))
        features3 = torch.cat((mags, jerk_mags))
        # Use kernel encoder
        sample_hv1 = self.kernel1(features1)
        sample_hv2 = self.kernel2(features2)
        sample_hv3 = self.kernel3(features3)
        sample_hv4 = self.kernel4(energy)
        sample_hv = sample_hv1 * sample_hv4 + sample_hv2 + sample_hv3
        sample_hv = torch.tanh(sample_hv)
        return sample_hv.squeeze(0)
