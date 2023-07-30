import torch
import numpy as np
from rctorch_mod import *
import torchhd
from torchhd import embeddings

NUM_CHANNELS = 3
NUM_RCN_NODES = 100
RCN_CONNECTIVITY = 0.1
RCN_SPECTRAL_RADIUS = 1.0
RCN_REGULARIZATION = 1.3
RCN_LEAKING_RATE = 0.12
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
        self.rcn = RcNetwork(**self.hps, feedback=False)
        self.x_basis = self.generate_basis(NUM_RCN_NODES + NUM_CHANNELS, out_dimension)
        self.y_basis = self.generate_basis(NUM_RCN_NODES + NUM_CHANNELS, out_dimension)
        self.z_basis = self.generate_basis(NUM_RCN_NODES + NUM_CHANNELS, out_dimension)
        self.feat_kernel = embeddings.Sinusoid(6, out_dimension, dtype=torch.float32)

    # Generate n x d matrix with orthogonal rows
    def generate_basis(self, features: int, dimension: int):
        # Generate random projection n x d matrix M using chosen probability distribution
        # Hyperdimensionality causes quasi-orthogonality
        M = np.random.normal(0, 1, (features, dimension))
        # return n x d matrix as a tensor
        return torch.tensor(M, dtype=torch.float32, device=self.my_device)

    # Encode window of feature vectors (x,y,z) and feature vectors (f,)
    def forward(self, signals: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        if (
            signals.dim() == 2
            and signals.size(dim=0) > 2
            and signals.size(dim=1) == NUM_CHANNELS + 1
        ):
            # Feature extraction from x, y, z samples
            signals = signals.float()  # RCN class works with float32
            # Fit RCN to timeseries to get weights
            self.rcn.fit(X=signals[:-1, 1:], y=signals[1:, 1:])
            # Convert weights to hypervectors
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
            sample_hv = torchhd.multiset(sample_hvs).sign()
            # Apply activation function
            sample_hv = torch.sin(sample_hv)
        else:
            sample_hv = torch.zeros_like(self.x_basis[0])
        # Encode calculated features
        sample_f_hv = self.feat_kernel(feat.float())
        sample_hv = sample_hv * sample_f_hv
        return sample_hv.double().flatten()
