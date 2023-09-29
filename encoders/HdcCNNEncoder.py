import torch
import torch.nn.functional as F
import numpy as np
import torchhd
from torchhd_custom import embeddings
from torchhd_custom import functional

NUM_CHANNEL = 4

# HDC Encoder for Bar Crawl Data
class HdcCNNEncoder(torch.nn.Module):
    def __init__(self, out_dimension: int):
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

        #Embeddings for extracted feature data
        self.feat_encode = embeddings.DensityFlocet(135, out_dimension, dtype=torch.float64)

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
        x = torch.nn.Linear(torch.numel(x), 128, device="cuda")(x)
        x = self.actv3(x)
        #Convert extracted features into hypervector
        sample_hv = torchhd.hard_quantize(self.hdc_kernel(x))

        # Encode calculated features
        feat_hv = torchhd.hard_quantize(self.feat_encode(feat))

        combined_hv = sample_hv * feat_hv

        # Apply activation function
        #combined_hv = torchhd.hard_quantize(combined_hv)
        return combined_hv.flatten()