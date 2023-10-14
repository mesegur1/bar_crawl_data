import torch
import torch.nn.functional as F
import numpy as np
import torchhd
from torchhd_custom import embeddings
from torchhd_custom import functional

NUM_CHANNEL = 3
NGRAM_SIZE = 3

# HDC Encoder for Bar Crawl Data
class HdcGenericEncoder(torch.nn.Module):
    def __init__(self, levels: int, out_dimension: int, device : str = "cpu"):
        super(HdcGenericEncoder, self).__init__()

        #Embeddings for raw data
        self.keys = embeddings.Random(NUM_CHANNEL, out_dimension, dtype=torch.float64)
        self.embed = embeddings.Level(levels, out_dimension, dtype=torch.float64)
        self.device = device

        #Embeddings for extracted feature data
        self.feat_emb = embeddings.DensityFlocet(18, out_dimension, dtype=torch.float64)
        # self.feat_emb = {}
        # for i in range(18):
        #     self.feat_emb[i] = embeddings.Sinusoid(1, out_dimension, dtype=torch.float64, device=self.device)

    # Encode window of raw features (t,x,y,z) and extracted feature vectors (f,)
    def forward(self, signals: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        # Use generic encoder
        sample_hvs = functional.generic(
            self.keys.weight, self.embed(signals[:, 1:]), NGRAM_SIZE
        )
        sample_hv = torchhd.multiset(sample_hvs)

        feat_hv = self.feat_emb(feat[[558, 582, 554, 552, 93, 555, 580, 571, 574, 578, 566, 287, 556, 550, 14, 551, 64, 581]])

        # feat_hvs = {}
        # feat_hvs[558] = self.feat_emb[0](feat[558].unsqueeze(0))
        # feat_hvs[582] = self.feat_emb[1](feat[582].unsqueeze(0))
        # feat_hvs[554] = self.feat_emb[2](feat[554].unsqueeze(0))
        # feat_hvs[552] = self.feat_emb[3](feat[552].unsqueeze(0))
        # feat_hvs[93] = self.feat_emb[4](feat[93].unsqueeze(0))
        # feat_hvs[555] = self.feat_emb[5](feat[555].unsqueeze(0))
        # feat_hvs[580] = self.feat_emb[6](feat[580].unsqueeze(0))
        # feat_hvs[571] = self.feat_emb[7](feat[571].unsqueeze(0))
        # feat_hvs[574] = self.feat_emb[8](feat[574].unsqueeze(0))
        # feat_hvs[578] = self.feat_emb[9](feat[578].unsqueeze(0))
        # feat_hvs[566] = self.feat_emb[10](feat[566].unsqueeze(0))
        # feat_hvs[287] = self.feat_emb[11](feat[287].unsqueeze(0))
        # feat_hvs[556] = self.feat_emb[12](feat[556].unsqueeze(0))
        # feat_hvs[550] = self.feat_emb[13](feat[550].unsqueeze(0))
        # feat_hvs[14] = self.feat_emb[14](feat[14].unsqueeze(0))
        # feat_hvs[551] = self.feat_emb[15](feat[551].unsqueeze(0))
        # feat_hvs[64] = self.feat_emb[16](feat[64].unsqueeze(0))
        # feat_hvs[581] = self.feat_emb[17](feat[581].unsqueeze(0))

        # feat_hv = ((feat_hvs[14] + feat_hvs[287])
        #             * (feat_hvs[64])
        #             * (feat_hvs[93] + feat_hvs[574] + feat_hvs[580] + feat_hvs[582] + feat_hvs[555] + feat_hvs[556] + feat_hvs[581])
        #             * (feat_hvs[550])
        #             * (feat_hvs[551] + feat_hvs[554])
        #             * (feat_hvs[552])
        #             * (feat_hvs[558])
        #             * (feat_hvs[566])
        #             * (feat_hvs[571])
        #             * (feat_hvs[578])
        # )

        combined_hv = sample_hv + feat_hv + sample_hv * feat_hv

        # Apply activation function
        combined_hv = torchhd.hard_quantize(combined_hv)
        return combined_hv.flatten()
