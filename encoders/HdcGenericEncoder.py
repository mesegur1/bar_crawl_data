import torch
import torch.nn.functional as F
import numpy as np
import torchhd
from torchhd_custom import embeddings
from torchhd_custom import functional

NUM_CHANNEL = 4
NGRAM_SIZE = 3
MFCC_COV_FEAT_LENGTH = 91
MFCC_COV_NUM = 6


# HDC Encoder for Bar Crawl Data
class HdcGenericEncoder(torch.nn.Module):
    def __init__(self, levels: int, out_dimension: int):
        super(HdcGenericEncoder, self).__init__()

        #Embeddings for raw data
        self.keys = embeddings.Random(NUM_CHANNEL, out_dimension, dtype=torch.float64)
        self.embed = embeddings.Level(levels, out_dimension, dtype=torch.float64)

        #Embeddings for extracted feature data
        self.feat_kernels = {}
        for s in range(18):
            self.feat_kernels[s] = embeddings.Sinusoid(3, out_dimension, dtype=torch.float64, device="cuda")
        # self.mfcc_feat_kernels = {}
        # for s in range(6):
        #     self.mfcc_feat_kernels[s] = embeddings.Sinusoid(MFCC_COV_FEAT_LENGTH, out_dimension, dtype=torch.float64, device="cuda")
        # self.mfcc_feat_kernel = embeddings.Sinusoid(20, out_dimension, dtype=torch.float64, device="cuda")
        self.mfcc_feat_level = embeddings.Level(1000, out_dimension, dtype=torch.float64)

    # Encode window of raw features (t,x,y,z) and extracted feature vectors (f,)
    def forward(self, signals: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        # Use generic encoder
        sample_hvs = functional.generic(
            self.keys.weight, self.embed(signals), NGRAM_SIZE
        )
        sample_hv = torchhd.multiset(sample_hvs)
        # Encode calculated features
        feat_hvs = {}
        feat_hvs[0] = self.feat_kernels[0](feat[546:549])
        feat_hvs[1] = self.feat_kernels[1](feat[549:552])
        feat_hvs[2] = self.feat_kernels[2](feat[552:555])
        feat_hvs[3] = self.feat_kernels[3](feat[555:558])
        feat_hvs[4] = self.feat_kernels[4](feat[558:561])
        feat_hvs[5] = self.feat_kernels[5](feat[561:564])
        feat_hvs[6] = self.feat_kernels[6](feat[564:567])
        feat_hvs[7] = self.feat_kernels[7](feat[567:570])
        feat_hvs[8] = self.feat_kernels[8](feat[570:573])
        feat_hvs[9] = self.feat_kernels[9](feat[573:576])
        feat_hvs[10] = self.feat_kernels[10](feat[576:579])
        feat_hvs[11] = self.feat_kernels[11](feat[579:582])
        feat_hvs[12] = self.feat_kernels[12](feat[582:585])
        feat_hvs[13] = self.feat_kernels[13](feat[585:588])
        feat_hvs[14] = self.feat_kernels[14](feat[588:591])
        feat_hvs[15] = self.feat_kernels[15](feat[591:594])
        feat_hvs[16] = self.feat_kernels[16](feat[594:597])
        feat_hvs[17] = self.feat_kernels[17](feat[597:600])

        # #MFCC features to use
        mfcc_feat = feat[[14, 15, 16, 17, 18, 20, 22, 24, 26, 27, 28, 29, 31, 37, 38, 39, 105, 106, 107, 
                          108, 109, 111, 113, 115, 117, 118, 120, 128, 129, 130, 196, 197, 198, 199, 200, 
                          201, 202, 203, 204, 206, 208, 209, 210, 211, 212, 213, 219, 220, 221, 285, 287, 
                          288, 289, 290, 291, 293, 295, 297, 299, 300, 301, 302, 310, 311, 312, 374, 376, 
                          378, 379, 380, 381, 382, 384, 386, 388, 390, 391, 392, 393, 395, 401, 402, 403, 
                          467, 469, 470, 471, 472, 473, 474, 475, 477, 479, 481, 482, 483, 484, 486, 492, 
                          494, ]]
        mfcc_feat = torch.nn.functional.normalize(mfcc_feat, dim=0)
        mfcc_feat_hvs = torchhd.soft_quantize(self.mfcc_feat_level(mfcc_feat))
        mfcc_feat_hv = torchhd.multibind(mfcc_feat_hvs)

        # mfcc_feat_hvs = []
        # for i in range(MFCC_COV_NUM):
        #     mfcc_feat_hvs.append(self.mfcc_feat_kernels[i](feat[i*MFCC_COV_FEAT_LENGTH:(i+1)*MFCC_COV_FEAT_LENGTH]))

        # (feat_hvs[0] + feat_hvs[7] + feat_hvs[15] + feat_hvs[16] + feat_hvs[17])
        # * (feat_hvs[1] + feat_hvs[2] + feat_hvs[3] + feat_hvs[4] + feat_hvs[5])
        # * (feat_hvs[6])
        # * (feat_hvs[8] + feat_hvs[9] + feat_hvs[10] + feat_hvs[11] + feat_hvs[12] + feat_hvs[13] + feat_hvs[14])

        # 0 RMS_START = MFCC_START + NUM_FEAT_ITEMS_MFCC
        # 1 MEAN_START = RMS_START + NUM_FEAT_ITEMS_GENERAL
        # 2 MEDIAN_START = MEAN_START + NUM_FEAT_ITEMS_GENERAL
        # 3 STD_START = MEDIAN_START + NUM_FEAT_ITEMS_GENERAL
        # 4 ABS_MAX_START = STD_START + NUM_FEAT_ITEMS_GENERAL
        # 5 ABS_MIN_START = ABS_MAX_START + NUM_FEAT_ITEMS_GENERAL
        # 6 FFT_MAX_START = ABS_MIN_START + NUM_FEAT_ITEMS_GENERAL
        # 7 ZERO_CROSS_RATE_START = FFT_MAX_START + NUM_FEAT_ITEMS_GENERAL
        # 8 SPECTRAL_ENTROPY_START = ZERO_CROSS_RATE_START + NUM_FEAT_ITEMS_GENERAL
        # 9 SPECTRAL_ENTROPY_FFT_START = SPECTRAL_ENTROPY_START + NUM_FEAT_ITEMS_GENERAL
        # 10 SPECTRAL_CENTROID_START = SPECTRAL_ENTROPY_FFT_START + NUM_FEAT_ITEMS_GENERAL
        # 11 SPECTRAL_SPREAD_START = SPECTRAL_CENTROID_START + NUM_FEAT_ITEMS_GENERAL
        # 12 SPECTRAL_FLUX_START = SPECTRAL_SPREAD_START + NUM_FEAT_ITEMS_GENERAL
        # 13 SPECTRAL_ROLLOFF_START = SPECTRAL_FLUX_START + NUM_FEAT_ITEMS_GENERAL
        # 14 SPECTRAL_PEAK_RATIO_START = SPECTRAL_ROLLOFF_START + NUM_FEAT_ITEMS_GENERAL
        # 15 SKEWNESS_START = SPECTRAL_PEAK_RATIO_START + NUM_FEAT_ITEMS_GENERAL
        # 16 KURTOSIS_START = SKEWNESS_START + NUM_FEAT_ITEMS_GENERAL
        # 17 AVG_POWER_START = KURTOSIS_START + NUM_FEAT_ITEMS_GENERAL


        #Combine hypervectors
        sample_hv = (
            sample_hv
            * (
                (feat_hvs[0] + feat_hvs[15] + feat_hvs[17])
                * (feat_hvs[3] + feat_hvs[4] + feat_hvs[5])
                * (feat_hvs[6])
                * (feat_hvs[11] + feat_hvs[12])
            ) * (sample_hv + mfcc_feat_hv)
        )

        # Apply activation function
        sample_hv = torchhd.hard_quantize(sample_hv)
        return sample_hv.flatten()
