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
        self.time_index = torch.tensor(range(timestamps), dtype=torch.float64, device=device)
        self.device = device
        self.out_dimension = out_dimension

        #Embeddings for extracted feature data
        self.feat_emb = embeddings.DensityFlocet(120, out_dimension, dtype=torch.float64)

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
        window = input.size(0)
        times = self.timestamps(self.time_index[:window])
        # Bind time sequence for x, y, z samples
        sample_hvs = (x_levels + y_levels + z_levels) * times
        sample_hv = torchhd.multibind(sample_hvs)

        # 20% of features (most important)
        feat_hv = self.feat_emb(feat[[557, 581, 553, 551, 92, 554, 579, 570, 573, 577, 565, 286, 
                                      555, 549, 13, 550, 63, 580, 556, 564, 0, 576, 567, 552, 578, 
                                      588, 597, 566, 571, 44, 572, 574, 14, 582, 381, 594, 4, 593, 
                                      218, 25, 84, 592, 3, 591, 547, 561, 562, 548, 319, 596, 558, 
                                      563, 87, 65, 599, 17, 88, 2, 49, 309, 6, 81, 15, 590, 589, 
                                      43, 273, 420, 546, 568, 400, 277, 202, 287, 434, 435, 423, 
                                      431, 301, 417, 412, 205, 179, 327, 176, 442, 172, 450, 391, 
                                      163, 154, 480, 485, 490, 491, 498, 503, 507, 509, 452, 239, 
                                      388, 219, 303, 292, 310, 316, 320, 322, 324, 326, 330, 336, 
                                      263, 262, 339, 340, 256, 345, 347]])

        combined_hv = sample_hv + feat_hv + sample_hv * feat_hv

        # Apply activation function
        combined_hv = torchhd.hard_quantize(combined_hv)
        return combined_hv.flatten()
