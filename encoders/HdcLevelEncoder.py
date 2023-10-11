import torch
import numpy as np
import torchhd

# from torchhd import embeddings
from torchhd_custom import embeddings

SIGNAL_X_MIN = -5
SIGNAL_X_MAX = 5
SIGNAL_Y_MIN = -5
SIGNAL_Y_MAX = 5
SIGNAL_Z_MIN = -5
SIGNAL_Z_MAX = 5


# HDC Encoder for Bar Crawl Data
class HdcLevelEncoder(torch.nn.Module):
    def __init__(self, levels: int, timestamps: int, out_dimension: int, device : str = "cpu"):
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
        self.device = device

        #Embeddings for extracted feature data
        #self.feat_emb = embeddings.DensityFlocet(18, out_dimension, dtype=torch.float64)
        self.feat_emb = {}
        for i in range(18):
            self.feat_emb[i] = embeddings.Sinusoid(1, out_dimension, dtype=torch.float64, device=self.device)

    # Encode window of raw vectors (x,y,z) and feature vectors (f,)
    def forward(self, input: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        # Get level hypervectors for x, y, z samples
        x_signal = torch.clamp(input[:, 1], min=SIGNAL_X_MIN, max=SIGNAL_X_MAX)
        y_signal = torch.clamp(input[:, 2], min=SIGNAL_Y_MIN, max=SIGNAL_Y_MAX)
        z_signal = torch.clamp(input[:, 3], min=SIGNAL_Z_MIN, max=SIGNAL_Z_MAX)
        x_levels = self.signal_level_x(x_signal)
        y_levels = self.signal_level_y(y_signal)
        z_levels = self.signal_level_z(z_signal)
        # Get time hypervectors
        times = self.timestamps(input[:, 0])
        # Bind time sequence for x, y, z samples
        sample_hvs = (x_levels + y_levels + z_levels) * times
        sample_hv = torchhd.multibind(sample_hvs)

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
