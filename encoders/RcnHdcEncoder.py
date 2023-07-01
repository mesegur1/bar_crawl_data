import torch
import numpy as np
from rctorch_mod import *
import torchhd

NUM_CHANNELS = 3
NUM_RCN_NODES = 200
NUM_TAC_LEVELS = 2
LEARNING_RATE = 0.05
RCN_CONNECTIVITY = 0.4
RCN_SPECTRAL_RADIUS = 1.2
RCN_REGULARIZATION = 1.7
RCN_LEAKING_RATE = 0.01
RCN_BIAS = 0.49


# RCN-HDC Encoder for Bar Crawl Data
class RcnHdcEncoder(torch.nn.Module):
    def __init__(self, out_dimension: int):
        super(RcnHdcEncoder, self).__init__()
        self.hps = {
            "n_nodes": NUM_RCN_NODES,
            "n_inputs": NUM_CHANNELS,
            "n_outputs": NUM_CHANNELS,
            "connectivity": RCN_CONNECTIVITY,
            "spectral_radius": RCN_SPECTRAL_RADIUS,
            "regularization": RCN_REGULARIZATION,
            "leaking_rate": RCN_LEAKING_RATE,
            "bias": RCN_BIAS,
        }
        self.my_device = torch_device("cuda" if torch.cuda.is_available() else "cpu")
        self.rcn = RcNetwork(**self.hps, feedback=True)
        self.x_basis = self.generate_basis(NUM_RCN_NODES + NUM_CHANNELS, out_dimension)
        self.y_basis = self.generate_basis(NUM_RCN_NODES + NUM_CHANNELS, out_dimension)
        self.z_basis = self.generate_basis(NUM_RCN_NODES + NUM_CHANNELS, out_dimension)

    # Generate n x d matrix with orthogonal rows
    def generate_basis(self, features: int, dimension: int):
        # Generate random projection n x d matrix M using chosen probability distribution
        # Hyperdimensionality causes quasi-orthogonality
        M = np.random.normal(0, 1, (features, dimension))
        # return n x d matrix as a tensor
        return torch.tensor(M, device=self.my_device)

    # Encode window of feature vectors (x,y,z)
    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        # Feature extraction from x, y, z samples
        self.rcn.fit(X=signals[:-1, 1:], y=signals[1:, 1:])
        x_hypervector = torch.matmul(
            self.rcn.LinOut.weight.data[0].float(), self.x_basis.float()
        )
        y_hypervector = torch.matmul(
            self.rcn.LinOut.weight.data[1].float(), self.y_basis.float()
        )
        z_hypervector = torch.matmul(
            self.rcn.LinOut.weight.data[2].float(), self.z_basis.float()
        )
        sample_hvs = torch.stack((x_hypervector, y_hypervector, z_hypervector))
        # Data fusion of channels
        sample_hv = torchhd.multiset(sample_hvs) + torchhd.multibind(sample_hvs)
        # Apply activation function
        sample_hv = torch.tanh(sample_hv)
        sample_hv = torchhd.hard_quantize(sample_hv)
        return sample_hv
