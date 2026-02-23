# Imports
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from IPython.display import Audio, display
from torch.utils.data import DataLoader, Dataset, Subset
from torchaudio import transforms


# Noise Dataset
class NoiseDataset(Dataset):
    def __init__(self, noise_dir, target_sample_rate = 16e3):
        self.files = [
            os.path.join(noise_dir, f)
            for f in os.listdir(noise_dir)
            if f.endswith(".wav")
        ]
        self.target_sample_rate = target_sample_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        waveform, sr = torchaudio.load(self.files[idx])

        if sr == self.target_sample_rate:
          return waveform

        # Else we need to resample it
        resampler = transforms.Resample(sr, self.target_sample_rate)
        waveform = resampler(waveform)

        return waveform
    

# Data Preprocessing

noise_dataset = NoiseDataset("./noise-dataset/ESC-50-master/audio")

speech_dataset = torchaudio.datasets.SPEECHCOMMANDS("./", download=True)

# Manual mixing of speech and noise, for example:
# 10% = 90% speech + 10% noise
# TODO: check if this is the correct way to doit
def mix_percent(speech, noise, percent):
    alpha = percent / 100.0

    # match length
    if noise.size(1) < speech.size(1):
        noise = noise.repeat(1, speech.size(1)//noise.size(1)+1)
    noise = noise[:, :speech.size(1)]

    mixed = (1 - alpha) * speech + alpha * noise

    # avoid clipping
    mixed = mixed / mixed.abs().max().clamp(min=1.0)

    return mixed

def test_mix():
  speech = torchaudio.datasets.SPEECHCOMMANDS
#   noise =
  test = mix_percent(speech[1], noise[1], 10)

def noiseify_speech_dataset(speech_dataset, noise_dataset, noise_level = 0):
  pass