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

        # Extract labels from folder names (safe method)
        labels = sorted(
            list(set(
                path.split(os.sep)[-2]
                for path in base_dataset._walker
            ))
        )

        self.label_to_index = {label: i for i, label in enumerate(labels)}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]

        waveform, sample_rate, label, speaker_id, utterance = \
            self.base_dataset[real_idx]

        if self.transform:
            waveform = self.transform(waveform)

        label = self.label_to_index[label]
        label = torch.tensor(label, dtype=torch.long)

        return waveform, label

# Force each entry to be exactly 16000 samples in length.
class FixAudioLength(nn.Module):
    def __init__(self, target_length=16000):
        super().__init__()
        self.target_length = target_length

    def forward(self, waveform):
        length = waveform.shape[1]

        if length > self.target_length:
            waveform = waveform[:, :self.target_length]
        elif length < self.target_length:
            pad_amount = self.target_length - length
            waveform = F.pad(waveform, (0, pad_amount))

        return waveform

class PrecomputedMelDataset(Dataset):
    """Dataset that loads precomputed mel-spectrograms and labels from disk"""
    def __init__(self, mel_file, label_file):
        self.mels = torch.load(mel_file)
        self.labels = torch.load(label_file)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.mels[idx], self.labels[idx]


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
# def prepare_spectragram_loaders(
#         noise_alpha = 0, 
#         batch_size = 1000, 
#         n_fft = 1024,
#         hop_length = 512,
#         n_mels = 64, 
# ):
#     """
#     Prepares loaders with a Mel-Spectragram transform.
#     """
#     subsets = prepare_plain_datasets()

#     train_transforms = nn.Sequential(
#         RandomNoise(noise_dataset, noise_alpha),
#         FixAudioLength(16000),
#         transforms.MelSpectrogram(
#             sample_rate=16000,
#             n_fft=n_fft,
#             n_mels=n_mels,
#             hop_length=hop_length,
#         )
#     )

#     val_test_transforms = nn.Sequential(
#         FixAudioLength(16000),
#         transforms.MelSpectrogram(
#             sample_rate=16000,
#             n_fft=n_fft,
#             n_mels=n_mels,
#             hop_length=hop_length,
#         )
#     )

#     return prepare_loaders(
#         subsets, 
#         train_transforms=train_transforms, 
#         val_test_transforms=val_test_transforms,
#         batch_size=batch_size
#     )

def prepare_spectrogram_loaders(
        batch_size=100,
        n_fft=1024,
        hop_length=512,
        n_mels=64,
        noise_alpha=0,
        precompute=False,
        precompute_dir="./precompute"
    ):
    """
    Prepare train/val/test loaders with optional precompute.
    If precompute=True, Mel-spectrograms are computed once and saved to disk.
    """
    os.makedirs(precompute_dir, exist_ok=True)
    train_file = os.path.join(precompute_dir, f"train_mels_{noise_alpha}.pt")
    val_file = os.path.join(precompute_dir, f"val_mels_{noise_alpha}.pt")
    test_file = os.path.join(precompute_dir, f"test_mels_{noise_alpha}.pt")
    train_labels_file = os.path.join(precompute_dir, f"train_labels_{noise_alpha}.pt")
    val_labels_file = os.path.join(precompute_dir, f"val_labels_{noise_alpha}.pt")
    test_labels_file = os.path.join(precompute_dir, f"test_labels_{noise_alpha}.pt")

    if precompute and all(os.path.exists(f) for f in [
        train_file, val_file, test_file,
        train_labels_file, val_labels_file, test_labels_file
    ]):
        print("Loading precomputed Mel-spectrograms from disk...")
        train_dataset = PrecomputedMelDataset(train_file, train_labels_file)
        val_dataset = PrecomputedMelDataset(val_file, val_labels_file)
        test_dataset = PrecomputedMelDataset(test_file, test_labels_file)
    else:
        print("Computing Mel-spectrograms (this may take a while)...")
        
        train_subset, val_subset, test_subset = prepare_plain_datasets()
        
        noise_transform = RandomNoise(noise_dataset, noise_alpha)
        fix_length_transform = FixAudioLength(16000)
        mel_transform = transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=n_fft,
            n_mels=n_mels,
            hop_length=hop_length,
        )

        all_labels = sorted(
            list(set(
                path.split(os.sep)[-2]
                for path in speech_dataset._walker
            ))
        )

        label2idx = {label: i for i, label in enumerate(all_labels)}

        def compute_mels(subset, apply_noise):
            mels, labels = [], []
            nr = 1
            for waveform, sample_rate, label, *_ in (speech_dataset[i] for i in subset.indices):
                if nr % 100 == 0:
                    print(f"\rComputing {nr}/{len(subset.indices)}", end="")
                # Apply optional noise
                if apply_noise:
                    waveform = noise_transform(waveform)
                waveform = fix_length_transform(waveform)
                mel = mel_transform(waveform)
                mels.append(mel)
                labels.append(label2idx[label])
                nr += 1
            print("")
            return torch.stack(mels), torch.tensor(labels, dtype=torch.long)

        print("Computing Training Mels")
        train_mels, train_labels = compute_mels(train_subset, apply_noise=True)
        print("Computing Validation Mels")
        val_mels, val_labels = compute_mels(val_subset, apply_noise=False)
        print("Computing Test Mels")
        test_mels, test_labels = compute_mels(test_subset, apply_noise=False)

        if precompute:
            torch.save(train_mels, train_file)
            torch.save(train_labels, train_labels_file)
            torch.save(val_mels, val_file)
            torch.save(val_labels, val_labels_file)
            torch.save(test_mels, test_file)
            torch.save(test_labels, test_labels_file)
            print(f"Saved precomputed datasets to {precompute_dir}")

        train_dataset = torch.utils.data.TensorDataset(train_mels, train_labels)
        val_dataset = torch.utils.data.TensorDataset(val_mels, val_labels)
        test_dataset = torch.utils.data.TensorDataset(test_mels, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# Returns loaders with raw audio waveforms.
def prepare_raw_loaders(noise_alpha = 0, batch_size = 1000):
    """
    Prepares loaders with raw audio waveforms.
    """
    subsets = prepare_plain_datasets()

    train_transforms = nn.Sequential(
        RandomNoise(noise_dataset, noise_alpha),
        FixAudioLength(16000),
    )

    val_test_transforms = nn.Sequential(
        FixAudioLength(16000),
    )

    return prepare_loaders(
        subsets, 
        train_transforms=train_transforms, 
        val_test_transforms=val_test_transforms,
        batch_size=batch_size
    )