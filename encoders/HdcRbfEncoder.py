import torch
import torch.nn.functional as F
import numpy as np
import torchhd
from torchhd_custom import embeddings

NUM_CHANNEL = 3


# HDC Encoder for Bar Crawl Data
class HdcRbfEncoder(torch.nn.Module):
    def __init__(
        self, timestamps: int, out_dimension: int, use_back_prop: bool = False, device : str = "cpu"
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
        self.device = device

        #Embeddings for extracted feature data
        #self.feat_emb = embeddings.DensityFlocet(18, out_dimension, dtype=torch.float64)
        self.feat_emb = {}
        for i in range(18):
            self.feat_emb[i] = embeddings.Sinusoid(1, out_dimension, dtype=torch.float64, device=self.device)

    # Encode window of feature vectors (x,y,z) and feature vectors (f,)
    def forward(self, input: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
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
        else:
            x_signal = input[:, 1]
            y_signal = input[:, 2]
            z_signal = input[:, 3]
        accel_features = torch.cat((x_signal, y_signal, z_signal))
        # Use kernel encoder
        sample_hv = self.kernel(accel_features)

        feat_hvs = {}
        feat_hvs[558] = self.feat_emb[0](feat[558].unsqueeze(0))
        feat_hvs[582] = self.feat_emb[1](feat[582].unsqueeze(0))
        feat_hvs[554] = self.feat_emb[2](feat[554].unsqueeze(0))
        feat_hvs[552] = self.feat_emb[3](feat[552].unsqueeze(0))
        feat_hvs[93] = self.feat_emb[4](feat[93].unsqueeze(0))
        feat_hvs[555] = self.feat_emb[5](feat[555].unsqueeze(0))
        feat_hvs[580] = self.feat_emb[6](feat[580].unsqueeze(0))
        feat_hvs[571] = self.feat_emb[7](feat[571].unsqueeze(0))
        feat_hvs[574] = self.feat_emb[8](feat[574].unsqueeze(0))
        feat_hvs[578] = self.feat_emb[9](feat[578].unsqueeze(0))
        feat_hvs[566] = self.feat_emb[10](feat[566].unsqueeze(0))
        feat_hvs[287] = self.feat_emb[11](feat[287].unsqueeze(0))
        feat_hvs[556] = self.feat_emb[12](feat[556].unsqueeze(0))
        feat_hvs[550] = self.feat_emb[13](feat[550].unsqueeze(0))
        feat_hvs[14] = self.feat_emb[14](feat[14].unsqueeze(0))
        feat_hvs[551] = self.feat_emb[15](feat[551].unsqueeze(0))
        feat_hvs[64] = self.feat_emb[16](feat[64].unsqueeze(0))
        feat_hvs[581] = self.feat_emb[17](feat[581].unsqueeze(0))

        feat_hv = ((feat_hvs[14] + feat_hvs[287])
                    * (feat_hvs[64])
                    * (feat_hvs[93] + feat_hvs[574] + feat_hvs[580] + feat_hvs[582] + feat_hvs[555] + feat_hvs[556] + feat_hvs[581])
                    * (feat_hvs[550])
                    * (feat_hvs[551] + feat_hvs[554])
                    * (feat_hvs[552])
                    * (feat_hvs[558])
                    * (feat_hvs[566])
                    * (feat_hvs[571])
                    * (feat_hvs[578])
        )

        combined_hv = sample_hv + feat_hv

        # Apply activation function
        combined_hv = torchhd.hard_quantize(combined_hv)
        return combined_hv.flatten()
