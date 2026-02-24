# Imports
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset
from torchaudio import transforms


# Noise Dataset. Helper class to allow torch to load the noise audio files as tensors and resample.
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
    

# Noisify Transform. Helper class to allow the speech dataset to be mixed with the noise based on an alpha value.
class RandomNoise(torch.nn.Module):
    def __init__(self, noise_dataset, alpha, random_start = False, p = 1):
        """
        noise_dataset: Dataset to sample noise from
        alpha: noise RMS relative to speech RMS
        random_start: random crop position in noise clip
        p: probability of applying noise (for augmentation)
        """

        super().__init__()
        self.noise_dataset = noise_dataset
        self.alpha = alpha
        self.random_start = random_start
        self.p = p

    def forward(self, speech_clip):
        # Probability check
        if torch.rand(1).item() > self.p:
            return speech_clip

        # Get random noise
        noise_clip = self.noise_dataset[
            random.randint(0, len(self.noise_dataset) - 1)
        ]

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
            random_start = random.randint(0, noise_len - speech_len) if self.random_start else 0
            noise_clip = noise_clip[:, random_start:random_start + speech_len]
        elif noise_len < speech_len:
            noise_clip = F.pad(noise_clip, (0, speech_len - noise_len))


        # RMS
        speech_rms = torch.sqrt(torch.mean(speech_clip ** 2))
        noise_rms = torch.sqrt(torch.mean(noise_clip ** 2))

        # Avoid zero division
        if noise_rms == 0:
            return speech_clip
        
        # Compute the noise power.
        desired_rms = self.alpha * speech_rms
        noise_clip = noise_clip * (desired_rms / noise_rms)

        # Mix speech and noise
        merged = speech_clip + noise_clip

        return merged

# Speech Subset. Helper class that applies transforms on the speech commands subsets.
class SpeechCommandsWithTransform(Dataset):
    def __init__(self, base_dataset, indices, transform=None):
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]

        waveform, sample_rate, label, speaker_id, utterance = \
            self.base_dataset[real_idx]

        if self.transform:
            waveform = self.transform(waveform)

        return waveform, label


noise_dataset = NoiseDataset("./noise_dataset")
speech_dataset = torchaudio.datasets.SPEECHCOMMANDS("./", download=True)

# Splits the speech dataset into train, validation and test subsets.
def prepare_plain_datasets(splits = [0.7, 0.15, 0.15]):
    generator = torch.Generator().manual_seed(1)
    
    train_subset, val_subset, test_subset = torch.utils.data.random_split(
        speech_dataset,
        splits,
        generator=generator
    )

    return train_subset, val_subset, test_subset

# Creates dataloaders with transforms from subsets.
def prepare_loaders(
        subsets, 
        train_transforms = nn.Sequential(), 
        val_test_transforms = nn.Sequential(), 
        batch_size = 1000
):
    train_subset, val_subset, test_subset = subsets

    train_set = SpeechCommandsWithTransform(
        speech_dataset,
        train_subset.indices,
        transform=train_transforms
    )
    
    val_set = SpeechCommandsWithTransform(
        speech_dataset,
        val_subset.indices,
        transform=val_test_transforms
    )
    
    test_set = SpeechCommandsWithTransform(
        speech_dataset,
        test_subset.indices,
        transform=val_test_transforms
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# Applies Mel-Spectragram on the train subset and returns loaders.
def prepare_spectragram_loaders(
        noise_alpha = 0, 
        batch_size = 1000, 
        n_fft = 1024,
        hop_length = 512,
        n_mels = 64, 
):
    """
    Prepares loaders with a Mel-Spectragram transform.
    """
    subsets = prepare_plain_datasets()

    train_transforms = nn.Sequential(
        RandomNoise(noise_dataset, noise_alpha),
        transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=n_fft,
            n_mels=n_mels,
            hop_length=hop_length,
        )
    )

    val_test_transforms = nn.Sequential(
        transforms.MelSpectrogram()
    )

    return prepare_loaders(
        subsets, 
        train_transforms=train_transforms, 
        val_test_transforms=val_test_transforms,
        batch_size=batch_size
    )


# Returns loaders with raw audio waveforms.
def prepare_raw_loaders(noise_alpha = 0, batch_size = 1000):
    """
    Prepares loaders with raw audio waveforms.
    """
    subsets = prepare_plain_datasets()

    train_transforms = nn.Sequential(
        RandomNoise(noise_dataset, noise_alpha)
    )

    val_test_transforms = nn.Sequential(
    )

    return prepare_loaders(
        subsets, 
        train_transforms=train_transforms, 
        val_test_transforms=val_test_transforms,
        batch_size=batch_size
    )