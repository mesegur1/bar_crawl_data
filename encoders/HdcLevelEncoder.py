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

        #Embeddings for extracted feature data
        self.feat_kernels = {}
        for s in range(24):
            if s < 6:
                self.feat_kernels[s] = embeddings.Sinusoid(91, out_dimension, dtype=torch.float64, device="cuda")
            else:
                self.feat_kernels[s] = embeddings.Sinusoid(3, out_dimension, dtype=torch.float64, device="cuda")

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

        # Encode calculated features
        feat_hvs = {}
        feat_hvs[0] = self.feat_kernels[0](feat[0:91])
        feat_hvs[1] = self.feat_kernels[1](feat[91:182])
        feat_hvs[2] = self.feat_kernels[2](feat[182:273])
        feat_hvs[3] = self.feat_kernels[3](feat[273:364])
        feat_hvs[4] = self.feat_kernels[4](feat[364:455])
        feat_hvs[5] = self.feat_kernels[5](feat[455:546])
        
        feat_hvs[6] = self.feat_kernels[6](feat[546:549])
        feat_hvs[7] = self.feat_kernels[7](feat[549:552])
        feat_hvs[8] = self.feat_kernels[8](feat[552:555])
        feat_hvs[9] = self.feat_kernels[9](feat[555:558])
        feat_hvs[10] = self.feat_kernels[10](feat[558:561])
        feat_hvs[11] = self.feat_kernels[11](feat[561:564])
        feat_hvs[12] = self.feat_kernels[12](feat[564:567])
        feat_hvs[13] = self.feat_kernels[13](feat[567:570])
        feat_hvs[14] = self.feat_kernels[14](feat[570:573])
        feat_hvs[15] = self.feat_kernels[15](feat[573:576])
        feat_hvs[16] = self.feat_kernels[16](feat[576:579])
        feat_hvs[17] = self.feat_kernels[17](feat[579:582])
        feat_hvs[18] = self.feat_kernels[18](feat[582:585])
        feat_hvs[19] = self.feat_kernels[19](feat[585:588])
        feat_hvs[20] = self.feat_kernels[20](feat[588:591])
        feat_hvs[21] = self.feat_kernels[21](feat[591:594])
        feat_hvs[22] = self.feat_kernels[22](feat[594:597])
        feat_hvs[23] = self.feat_kernels[23](feat[597:600])

        # 6 RMS_START = MFCC_START + NUM_FEAT_ITEMS_MFCC
        # 7 MEAN_START = RMS_START + NUM_FEAT_ITEMS_GENERAL
        # 8 MEDIAN_START = MEAN_START + NUM_FEAT_ITEMS_GENERAL
        # 9 STD_START = MEDIAN_START + NUM_FEAT_ITEMS_GENERAL
        # 10 ABS_MAX_START = STD_START + NUM_FEAT_ITEMS_GENERAL
        # 11 ABS_MIN_START = ABS_MAX_START + NUM_FEAT_ITEMS_GENERAL
        # 12 FFT_MAX_START = ABS_MIN_START + NUM_FEAT_ITEMS_GENERAL
        # 13 ZERO_CROSS_RATE_START = FFT_MAX_START + NUM_FEAT_ITEMS_GENERAL
        # 14 SPECTRAL_ENTROPY_START = ZERO_CROSS_RATE_START + NUM_FEAT_ITEMS_GENERAL
        # 15 SPECTRAL_ENTROPY_FFT_START = SPECTRAL_ENTROPY_START + NUM_FEAT_ITEMS_GENERAL
        # 16 SPECTRAL_CENTROID_START = SPECTRAL_ENTROPY_FFT_START + NUM_FEAT_ITEMS_GENERAL
        # 17 SPECTRAL_SPREAD_START = SPECTRAL_CENTROID_START + NUM_FEAT_ITEMS_GENERAL
        # 18 SPECTRAL_FLUX_START = SPECTRAL_SPREAD_START + NUM_FEAT_ITEMS_GENERAL
        # 19 SPECTRAL_ROLLOFF_START = SPECTRAL_FLUX_START + NUM_FEAT_ITEMS_GENERAL
        # 20 SPECTRAL_PEAK_RATIO_START = SPECTRAL_ROLLOFF_START + NUM_FEAT_ITEMS_GENERAL
        # 21 SKEWNESS_START = SPECTRAL_PEAK_RATIO_START + NUM_FEAT_ITEMS_GENERAL
        # 22 KURTOSIS_START = SKEWNESS_START + NUM_FEAT_ITEMS_GENERAL
        # 23 AVG_POWER_START = KURTOSIS_START + NUM_FEAT_ITEMS_GENERAL

        combined_hv = (sample_hv
            * (feat_hvs[6] + feat_hvs[21] + feat_hvs[23])
            * (feat_hvs[9] + feat_hvs[10])
            * (feat_hvs[11])
            * (feat_hvs[12])
            * (feat_hvs[17])
            * (feat_hvs[18])
            + sample_hv * (feat_hvs[6] + feat_hvs[10] + feat_hvs[11] + feat_hvs[12])
            * (feat_hvs[0] * feat_hvs[1] * feat_hvs[2] * feat_hvs[3] * feat_hvs[4] * feat_hvs[5])
        )

        # Apply activation function
        combined_hv = torchhd.hard_quantize(combined_hv)
        return combined_hv.flatten()
