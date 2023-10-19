import torch
import torch.nn.functional as F
import numpy as np
import torchhd
from torchhd_custom import embeddings

NUM_CHANNEL = 3
NGRAM_SIZE = 6

# HDC Encoder for Bar Crawl Data
class HdcSinusoidNgramEncoder(torch.nn.Module):
    def __init__(self, out_dimension: int, device : str = "cpu"):
        super(HdcSinusoidNgramEncoder, self).__init__()

        self.kernel = embeddings.Sinusoid(
            NUM_CHANNEL, out_dimension, dtype=torch.float64
        )
        self.device = device

        #Embeddings for extracted feature data
        self.feat_emb = embeddings.DensityFlocet(120, out_dimension, dtype=torch.float64)

    # Encode window of feature vectors (x,y,z) and feature vectors (f,)
    def forward(self, input: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        # Get features from x, y, z samples
        signals = input[:, 1:]
        # Use kernel encoder
        sample_hv = self.kernel(signals)
        # Perform ngram statistics
        sample_hv = torchhd.ngrams(sample_hv, NGRAM_SIZE)

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
