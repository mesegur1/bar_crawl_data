import torch
import torch.nn.functional as F
import numpy as np
import torchhd
from torchhd_custom import embeddings
from torchhd_custom import functional

NUM_CHANNEL = 4
NGRAM_SIZE = 3

NUM_FEAT_ITEMS_MFCC = 6
NUM_FEAT_ITEMS_GENERAL = 3
NUM_FEAT_GENERAL = 18

MFCC_START = 0
RMS_START = MFCC_START + NUM_FEAT_ITEMS_MFCC
MEAN_START = RMS_START + NUM_FEAT_ITEMS_GENERAL
MEDIAN_START = MEAN_START + NUM_FEAT_ITEMS_GENERAL
STD_START = MEDIAN_START + NUM_FEAT_ITEMS_GENERAL
ABS_MAX_START = STD_START + NUM_FEAT_ITEMS_GENERAL
ABS_MIN_START = ABS_MAX_START + NUM_FEAT_ITEMS_GENERAL
FFT_MAX_START = ABS_MIN_START + NUM_FEAT_ITEMS_GENERAL
ZERO_CROSS_RATE_START = FFT_MAX_START + NUM_FEAT_ITEMS_GENERAL
SPECTRAL_ENTROPY_START = ZERO_CROSS_RATE_START + NUM_FEAT_ITEMS_GENERAL
SPECTRAL_ENTROPY_FFT_START = SPECTRAL_ENTROPY_START + NUM_FEAT_ITEMS_GENERAL
SPECTRAL_CENTROID_START = SPECTRAL_ENTROPY_FFT_START + NUM_FEAT_ITEMS_GENERAL
SPECTRAL_SPREAD_START = SPECTRAL_CENTROID_START + NUM_FEAT_ITEMS_GENERAL
SPECTRAL_FLUX_START = SPECTRAL_SPREAD_START + NUM_FEAT_ITEMS_GENERAL
SPECTRAL_ROLLOFF_START = SPECTRAL_FLUX_START + NUM_FEAT_ITEMS_GENERAL
SPECTRAL_PEAK_RATIO_START = SPECTRAL_ROLLOFF_START + NUM_FEAT_ITEMS_GENERAL
SKEWNESS_START = SPECTRAL_PEAK_RATIO_START + NUM_FEAT_ITEMS_GENERAL
KURTOSIS_START = SKEWNESS_START + NUM_FEAT_ITEMS_GENERAL
AVG_POWER_START = KURTOSIS_START + NUM_FEAT_ITEMS_GENERAL


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
        self.feat_mfcc_kernel = embeddings.Sinusoid(NUM_FEAT_ITEMS_MFCC, out_dimension, dtype=torch.float64)
        self.feat_rms_kernel = embeddings.Sinusoid(NUM_FEAT_ITEMS_GENERAL, out_dimension, dtype=torch.float64)
        self.feat_mean_kernel = embeddings.Sinusoid(NUM_FEAT_ITEMS_GENERAL, out_dimension, dtype=torch.float64)
        self.feat_median_kernel = embeddings.Sinusoid(NUM_FEAT_ITEMS_GENERAL, out_dimension, dtype=torch.float64)
        self.feat_std_kernel = embeddings.Sinusoid(NUM_FEAT_ITEMS_GENERAL, out_dimension, dtype=torch.float64)
        self.feat_abs_max_kernel = embeddings.Sinusoid(NUM_FEAT_ITEMS_GENERAL, out_dimension, dtype=torch.float64)
        self.feat_abs_min_kernel = embeddings.Sinusoid(NUM_FEAT_ITEMS_GENERAL, out_dimension, dtype=torch.float64)
        self.feat_fft_max_kernel = embeddings.Sinusoid(NUM_FEAT_ITEMS_GENERAL, out_dimension, dtype=torch.float64)
        self.feat_zero_cross_kernel = embeddings.Sinusoid(NUM_FEAT_ITEMS_GENERAL, out_dimension, dtype=torch.float64)
        self.feat_spectral_entropy_kernel = embeddings.Sinusoid(NUM_FEAT_ITEMS_GENERAL, out_dimension, dtype=torch.float64)
        self.feat_spectral_entropy_fft_kernel = embeddings.Sinusoid(NUM_FEAT_ITEMS_GENERAL, out_dimension, dtype=torch.float64)
        self.feat_spectral_centroid_kernel = embeddings.Sinusoid(NUM_FEAT_ITEMS_GENERAL, out_dimension, dtype=torch.float64)
        self.feat_spectral_spread_kernel = embeddings.Sinusoid(NUM_FEAT_ITEMS_GENERAL, out_dimension, dtype=torch.float64)
        self.feat_spectral_flux_kernel = embeddings.Sinusoid(NUM_FEAT_ITEMS_GENERAL, out_dimension, dtype=torch.float64)
        self.feat_spectral_rolloff_kernel = embeddings.Sinusoid(NUM_FEAT_ITEMS_GENERAL, out_dimension, dtype=torch.float64)
        self.feat_spectral_peak_ratio_kernel = embeddings.Sinusoid(NUM_FEAT_ITEMS_GENERAL, out_dimension, dtype=torch.float64)
        self.feat_skewness_kernel = embeddings.Sinusoid(NUM_FEAT_ITEMS_GENERAL, out_dimension, dtype=torch.float64)
        self.feat_kurtosis_kernel = embeddings.Sinusoid(NUM_FEAT_ITEMS_GENERAL, out_dimension, dtype=torch.float64)
        self.feat_avg_power_kernel = embeddings.Sinusoid(NUM_FEAT_ITEMS_GENERAL, out_dimension, dtype=torch.float64)

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
        sample_hv = self.hdc_kernel(x)

        # Encode calculated features
        feat_mffc_hv = self.feat_mfcc_kernel(feat[MFCC_START:MFCC_START+NUM_FEAT_ITEMS_MFCC])
        feat_rms_hv = self.feat_rms_kernel(feat[RMS_START:RMS_START+NUM_FEAT_ITEMS_GENERAL])
        feat_mean_hv = self.feat_mean_kernel(feat[MEAN_START:MEAN_START+NUM_FEAT_ITEMS_GENERAL])
        feat_median_hv = self.feat_median_kernel(feat[MEDIAN_START:MEDIAN_START+NUM_FEAT_ITEMS_GENERAL])
        feat_std_hv = self.feat_std_kernel(feat[STD_START:STD_START+NUM_FEAT_ITEMS_GENERAL])
        feat_abs_max_hv = self.feat_abs_max_kernel(feat[ABS_MAX_START:ABS_MAX_START+NUM_FEAT_ITEMS_GENERAL])
        feat_abs_min_hv = self.feat_abs_min_kernel(feat[ABS_MIN_START:ABS_MIN_START+NUM_FEAT_ITEMS_GENERAL])
        feat_fft_max_hv = self.feat_fft_max_kernel(feat[FFT_MAX_START:FFT_MAX_START+NUM_FEAT_ITEMS_GENERAL])
        feat_zero_cross_hv = self.feat_zero_cross_kernel(feat[ZERO_CROSS_RATE_START:ZERO_CROSS_RATE_START+NUM_FEAT_ITEMS_GENERAL])
        feat_spctl_ent_hv = self.feat_spectral_entropy_kernel(feat[SPECTRAL_ENTROPY_START:SPECTRAL_ENTROPY_START+NUM_FEAT_ITEMS_GENERAL])
        feat_spctl_ent_fft_hv = self.feat_spectral_entropy_fft_kernel(feat[SPECTRAL_ENTROPY_FFT_START:SPECTRAL_ENTROPY_FFT_START+NUM_FEAT_ITEMS_GENERAL])
        feat_spctl_cntd_hv = self.feat_spectral_centroid_kernel(feat[SPECTRAL_CENTROID_START:SPECTRAL_CENTROID_START+NUM_FEAT_ITEMS_GENERAL])
        feat_spctl_sprd_hv = self.feat_spectral_spread_kernel(feat[SPECTRAL_SPREAD_START:SPECTRAL_SPREAD_START+NUM_FEAT_ITEMS_GENERAL])
        feat_spctl_flux_hv = self.feat_spectral_flux_kernel(feat[SPECTRAL_FLUX_START:SPECTRAL_FLUX_START+NUM_FEAT_ITEMS_GENERAL])
        feat_spctl_rolloff_hv = self.feat_spectral_rolloff_kernel(feat[SPECTRAL_ROLLOFF_START:SPECTRAL_ROLLOFF_START+NUM_FEAT_ITEMS_GENERAL])
        feat_spctl_peak_r_hv = self.feat_spectral_peak_ratio_kernel(feat[SPECTRAL_PEAK_RATIO_START:SPECTRAL_PEAK_RATIO_START+NUM_FEAT_ITEMS_GENERAL])
        feat_skew_hv = self.feat_skewness_kernel(feat[SKEWNESS_START:SKEWNESS_START+NUM_FEAT_ITEMS_GENERAL])
        feat_kurt_hv = self.feat_kurtosis_kernel(feat[KURTOSIS_START:KURTOSIS_START+NUM_FEAT_ITEMS_GENERAL])
        feat_avg_pwr_hv = self.feat_avg_power_kernel(feat[AVG_POWER_START:AVG_POWER_START+NUM_FEAT_ITEMS_GENERAL])

        sample_hv = (
            sample_hv
            * (feat_rms_hv + feat_zero_cross_hv + feat_skew_hv + feat_kurt_hv + feat_avg_pwr_hv)
            * (feat_mean_hv + feat_median_hv + feat_std_hv + feat_abs_max_hv + feat_abs_min_hv)
            * (feat_fft_max_hv)
            * (feat_mffc_hv + feat_spctl_ent_hv + feat_spctl_ent_fft_hv + feat_spctl_cntd_hv + 
               feat_spctl_sprd_hv + feat_spctl_flux_hv + feat_spctl_rolloff_hv + feat_spctl_peak_r_hv)
        )

        # Apply activation function
        sample_hv = torchhd.hard_quantize(sample_hv)
        return sample_hv.flatten()