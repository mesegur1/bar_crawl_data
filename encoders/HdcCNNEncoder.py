import torch
import torch.nn.functional as F
import numpy as np
import torchhd
from torchhd_custom import embeddings
from torchhd_custom import functional

NUM_CHANNEL = 4

# HDC Encoder for Bar Crawl Data
class HdcCNNEncoder(torch.nn.Module):
    def __init__(self, out_dimension: int, device : str = "cpu"):
        super(HdcCNNEncoder, self).__init__()

        #CNN for raw data
        self.conv1 = torch.nn.Conv1d(NUM_CHANNEL, 64, kernel_size=10)
        self.conv2 = torch.nn.Conv1d(64, 128, kernel_size=10)
        self.actv1 = torch.nn.ReLU()
        self.actv2 = torch.nn.ReLU()
        self.actv3 = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.5)
        self.maxpool = torch.nn.MaxPool1d(2)
        self.hdc_kernel = embeddings.Sinusoid(128, out_dimension, dtype=torch.float64)
        self.device = device

        #Embeddings for extracted feature data
        self.feat_emb = embeddings.DensityFlocet(120, out_dimension, dtype=torch.float64)

    # Encode window of raw features (t,x,y,z) and extracted feature vectors (f,)
    def forward(self, signals: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        # Apply CNN
        signals = signals.transpose(0, 1).contiguous()
        x = self.conv1(signals[None, ...])
        x = self.actv1(x)
        x = self.conv2(x)
        x = self.actv2(x)
        x = self.dropout(x)
        x = self.maxpool(x)
        #Fully connected layer
        x = torch.flatten(x, 1)
        x = torch.nn.Linear(torch.numel(x), 128, device=self.device)(x)
        x = self.actv3(x)
        #Convert extracted features into hypervector
        sample_hv = self.hdc_kernel(x)

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