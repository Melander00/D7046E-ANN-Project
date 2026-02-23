# Imports
import os
import random

import sounddevice as sd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset
from torchaudio import transforms

# torchaudio.set_audio_backend("soundfile")

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
noise_dataset = NoiseDataset("./noise_dataset")
speech_dataset = torchaudio.datasets.SPEECHCOMMANDS("./", download=True)

def merge_speech_noise(speech_clip: Tensor, noise_clip: Tensor, alpha: float, random_start = False):
    """Merges the speech clip with the noise based on the alpha value [0,1]"""

    # Ensure Mono
    if speech_clip.shape[0] > 1:
        speech_clip = torch.mean(speech_clip, dim=0, keepdim=True)
    if noise_clip.shape[0] > 1:
        noise_clip = torch.mean(noise_clip, dim=0, keepdim=True)

    # Clip Lengths
    speech_len = speech_clip.shape[1]
    noise_len = noise_clip.shape[1]

    # Crop/Pad Noise
    if noise_len > speech_len:
        random_start = random.randint(0, noise_len - speech_len) if random_start else 0
        noise_clip = noise_clip[:, random_start:random_start + speech_len]
    elif noise_len < speech_len:
        noise_clip = F.pad(noise_clip, (0, speech_len - noise_len))


    # RMS
    speech_rms = torch.sqrt(torch.mean(speech_clip ** 2))
    noise_rms = torch.sqrt(torch.mean(noise_clip ** 2))

    # Avoid zero division
    if noise_rms == 0:
        return speech_clip
    
    desired_rms = alpha * speech_rms
    noise_clip = noise_clip * (desired_rms / noise_rms)

    merged = speech_clip + noise_clip

    return merged


   

# # Manual mixing of speech and noise, for example:
# # 10% = 90% speech + 10% noise
# # TODO: check if this is the correct way to doit
# def mix_percent(speech, noise, percent):
#     alpha = percent / 100.0

#     # match length
#     if noise.size(1) < speech.size(1):
#         noise = noise.repeat(1, speech.size(1)//noise.size(1)+1)
#     noise = noise[:, :speech.size(1)]

#     mixed = (1 - alpha) * speech + alpha * noise

#     # avoid clipping
#     mixed = mixed / mixed.abs().max().clamp(min=1.0)

#     return mixed

# def test_mix():
#     speech = torchaudio.datasets.SPEECHCOMMANDS
#     #   noise =
#     test = mix_percent(speech[1], noise[1], 10)

# def noiseify_speech_dataset(speech_dataset, noise_dataset, noise_level = 0):
#     pass

def play_sound(clip: Tensor, sample_rate = 16000):
    if clip.dim() == 2:
        clip = clip.squeeze(0)
    sd.play(clip.numpy(), sample_rate)
    sd.wait()


if __name__ == "__main__":
    speech = speech_dataset[1][0]
    noise = noise_dataset[1]

    mixed = merge_speech_noise(speech, noise, 0)

    print("playing sound")
    play_sound(mixed)
    print("done!")

