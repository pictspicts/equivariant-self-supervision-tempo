# -*- coding: utf-8 -*-
"""
@author: Elio Quinton
python3.8

Pre-processing front-end for all neural networks.
"""
import torch
import torchaudio as ta
import torch.nn as nn
import torch.nn.functional as F
import math

from sst.augmentations import TimeStretchFixedSize, Vol, PolarityInversion, GaussianNoise

EPS = 10e-8


class AugParams:
    pass


class FrontEndAug(nn.Module):
    '''Main module class'''
    def __init__(self, config, tempo_range=(0, 300)):
        super(FrontEndAug, self).__init__()
        self.config = config #This is only the "frontend" config - i.e. a subset of the full reference config.
        self.tempo_range = tempo_range
        self.freq_mask_param = self.config.n_fft//(1/self.config.aug_params.freq_masking.mask_ratio_max)

        # Time domain transforms
        self.vol = Vol(gain_type='db')
        self.pol_inv = PolarityInversion()
        self.gauss_noise = GaussianNoise()

        # Complex Spectrogram Pre-processing
        self.spectrogram_cx = ta.transforms.Spectrogram(
            n_fft=self.config.n_fft, 
            hop_length=self.config.hop_length, 
            window_fn=torch.hann_window,
            power=None
            )
        self.melscale = ta.transforms.MelScale(
            sample_rate=self.config.sr, 
            n_stft=self.config.n_fft//2+1, 
            f_min=self.config.f_min, 
            f_max=self.config.f_max, 
            n_mels=self.config.n_mels
            )

        # Frequency domain transforms
        self.timestretch = TimeStretchFixedSize(
            hop_length=self.config.hop_length, 
            n_freq=self.config.n_fft//2+1, 
            tempo_range=self.tempo_range
            )
        self.freq_masking = ta.transforms.FrequencyMasking(freq_mask_param=self.freq_mask_param)

    def draw_timestretch_rate(self, rate_min=0.8, rate_max=1.2):
        rate_tensor = (rate_min - rate_max)*torch.rand(1) + rate_max
        return rate_tensor.item()

    def draw_random_float_in_range(self, value_min: float, value_max: float):
        '''Draw a random float value in range [value_min, value_max]'''
        value_tensor = (value_min - value_max)*torch.rand(1) + value_max
        return value_tensor.item()

    def forward(self, x, y):
        # Waveform Augmentations
        if 'volume' in self.config.augmentations:
            vol_gain = self.draw_random_float_in_range(
                self.config.aug_params.volume.gain_min, 
                self.config.aug_params.volume.gain_max
                )
            x = self.vol(x, vol_gain)
        if 'polarity_inversion' in self.config.augmentations:
            p = self.draw_random_float_in_range(0, 1)
            if p <= self.config.aug_params.polarity_inversion.prob:
                x = self.pol_inv(x)
        if 'gaussian_noise' in self.config.augmentations:
            std = self.draw_random_float_in_range(
                self.config.aug_params.gaussian_noise.std_min, 
                self.config.aug_params.gaussian_noise.std_min
                )
            x = self.gauss_noise(x, std)
        # Complex spectrogram
        x = self.spectrogram_cx(x)
        
        # Decide Time-Stretch rate and update target labels (but don't use the phase vocoder here!)
        ts_rate = -1.0
        if 'timestretch' in self.config.augmentations:
            ts_rate = self.draw_random_float_in_range(
                self.config.aug_params.timestretch.rate_min, 
                self.config.aug_params.timestretch.rate_max
                )
            y = y * ts_rate
            y[y < self.tempo_range[0]] = 0.0
            y[y > self.tempo_range[1]] = 0.0
            
        if self.config.power is not None:
            if self.config.power == 1.0:
                x = x.abs()
            else:
                x = x.abs().pow(self.config.power)
        if 'freq_masking' in self.config.augmentations:
            x = self.freq_masking(x)
        # Compute log melspectrogram
        x = self.melscale(x)
        x = torch.log(x + EPS)
        
        # --- NEW: Image-Based Artifact-Free Time Stretch ---
        if ts_rate > 0:
            original_len = x.shape[-1]
            target_len = int(original_len / ts_rate)
            
            # Interpolate (resize) safely along the time dimension
            x_stretched = F.interpolate(x, size=target_len, mode='linear', align_corners=False)
            
            # THE MAGIC FIX: Random crop to EXACTLY 1361 frames (equivalent to 600,000 samples)
            # Since dataloader passed a super-sampled 780,000 samples, target_len is ALWAYS >= 1361.
            window_size = 1361
            if target_len > window_size:
                # Randomly slide the window. This ensures no pad/cut artifacts exist.
                # Also creates Translation-Invariance since x_i and x_j will get slightly different rhythmic phases!
                start_max = target_len - window_size
                start_idx = torch.randint(0, start_max + 1, (1,)).item()
                x = x_stretched[..., start_idx : start_idx + window_size]
            else:
                # Safety fallback just in case someone sets rate_max > 1.3
                pad_len = window_size - target_len
                pad_val = math.log(EPS)
                x = F.pad(x_stretched, (0, pad_len), "constant", pad_val)
                
        return x, y, ts_rate


class FrontEndNoAug(nn.Module):
    '''Main module class'''
    def __init__(self, config):
        super(FrontEndNoAug, self).__init__()
        self.config = config
        # audio Pre-processing
        self.spectrogram_cx = ta.transforms.Spectrogram(
            n_fft=self.config.n_fft, 
            hop_length=self.config.hop_length,
            window_fn=torch.hann_window, 
            power=None, 
            return_complex=True
            )
        self.melscale = ta.transforms.MelScale(
            sample_rate=self.config.sr, 
            n_stft=self.config.n_fft // 2 + 1,
            f_min=self.config.f_min,
            f_max=self.config.f_max,
            n_mels=self.config.n_mels
            )

    def forward(self, x, y):
        x = self.spectrogram_cx(x)
        # Get magnitude / power spectrogram
        if self.config.power is not None:
            if self.config.power == 1.0:
                x = x.abs()
            else:
                x = x.abs().pow(self.config.power)
        # Compute log melspectrogram
        x = self.melscale(x)
        x = torch.log(x + EPS)
        # Mock ts_rate, for consistency with FrontEndAug class.
        ts_rate = -1
        return x, y, ts_rate
