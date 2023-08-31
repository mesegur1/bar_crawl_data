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
        chosen_feat = [547, 550, 551, 553, 554, 556, 559, 562, 565, 570, 576, 579, 583, 584, 585, 588, 593, 594, 595, 598, ]
        self.feat_kernels = {}
        for f in chosen_feat:
            self.feat_kernels[f] = embeddings.Sinusoid(1, out_dimension, dtype=torch.float64, device="cuda")
        self.mfcc_feat_kernels = []
        for _ in range(6):
            self.mfcc_feat_kernels.append(embeddings.Sinusoid(MFCC_COV_FEAT_LENGTH, out_dimension, dtype=torch.float64, device="cuda"))

    # Encode window of raw features (t,x,y,z) and extracted feature vectors (f,)
    def forward(self, signals: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        # Use generic encoder
        sample_hvs = functional.generic(
            self.keys.weight, self.embed(signals), NGRAM_SIZE
        )
        sample_hv = torchhd.multiset(sample_hvs)
        # Encode calculated features
        feat_hvs = {}
        feat_hvs[547] = self.feat_kernels[547](feat[546].unsqueeze(0))
        feat_hvs[550] = self.feat_kernels[550](feat[549].unsqueeze(0))
        feat_hvs[551] = self.feat_kernels[551](feat[550].unsqueeze(0))
        feat_hvs[553] = self.feat_kernels[553](feat[552].unsqueeze(0))
        feat_hvs[554] = self.feat_kernels[554](feat[553].unsqueeze(0))
        feat_hvs[556] = self.feat_kernels[556](feat[555].unsqueeze(0))
        feat_hvs[559] = self.feat_kernels[559](feat[558].unsqueeze(0))
        feat_hvs[562] = self.feat_kernels[562](feat[561].unsqueeze(0))
        feat_hvs[565] = self.feat_kernels[565](feat[564].unsqueeze(0))
        feat_hvs[570] = self.feat_kernels[570](feat[569].unsqueeze(0))
        feat_hvs[576] = self.feat_kernels[576](feat[575].unsqueeze(0))
        feat_hvs[579] = self.feat_kernels[579](feat[578].unsqueeze(0))
        feat_hvs[583] = self.feat_kernels[583](feat[582].unsqueeze(0))
        feat_hvs[584] = self.feat_kernels[584](feat[583].unsqueeze(0))
        feat_hvs[585] = self.feat_kernels[585](feat[584].unsqueeze(0))
        feat_hvs[588] = self.feat_kernels[588](feat[587].unsqueeze(0))
        feat_hvs[593] = self.feat_kernels[593](feat[592].unsqueeze(0))
        feat_hvs[594] = self.feat_kernels[594](feat[593].unsqueeze(0))
        feat_hvs[595] = self.feat_kernels[595](feat[594].unsqueeze(0))
        feat_hvs[598] = self.feat_kernels[598](feat[597].unsqueeze(0))
        mfcc_feat_hvs = []
        for i in range(MFCC_COV_NUM):
            mfcc_feat_hvs.append(self.mfcc_feat_kernels[i](feat[i*MFCC_COV_FEAT_LENGTH:(i+1)*MFCC_COV_FEAT_LENGTH]))

        #Combine hypervectors
        mfcc_hv = torchhd.multibind(torch.concat(mfcc_feat_hvs, dim=0))
        sample_hv = (
            sample_hv
            * (
                + (feat_hvs[547] * feat_hvs[565] * feat_hvs[562])
                + (feat_hvs[550] * feat_hvs[553])
                + (feat_hvs[551] * feat_hvs[554])
                + (feat_hvs[556])
                + (feat_hvs[559])
                + (feat_hvs[570] * feat_hvs[588] * feat_hvs[576] * feat_hvs[579])
                + (feat_hvs[583])
                + (feat_hvs[584])
                + (feat_hvs[585])
                + (feat_hvs[593])
                + (feat_hvs[594])
                + (feat_hvs[595])
                + (feat_hvs[598])
                + mfcc_hv
            )
        )
        #sample_hv = sample_hv * torchhd.multibind(torch.concat(mfcc_feat_hvs, dim=0))

        # Apply activation function
        sample_hv = torchhd.hard_quantize(sample_hv)
        return sample_hv.flatten()
