# Imports
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset, Subset
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
    def __init__(self, mel_file, label_file, device="cpu"):
        self.mels = torch.load(mel_file, map_location=device)
        self.labels = torch.load(label_file, map_location=device)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.mels[idx], self.labels[idx]




# Splits the speech dataset into train, validation and test subsets.
def prepare_plain_datasets(
    splits = [0.7, 0.15, 0.15], 
    target_labels = None, 
    noise_dataset_path = "./noise_dataset/audio", 
    device="cpu"
):
    noise_dataset = NoiseDataset(noise_dataset_path) #NOTE: filepath is changed for my pc so edit it for urs if u wanna run the model
    speech_dataset = torchaudio.datasets.SPEECHCOMMANDS("./", download=True)

    if target_labels is not None:
        print("Filtering...")
        indices = [
            i for i, path in enumerate(speech_dataset._walker)
            if os.path.normpath(path).split(os.sep)[-2] in target_labels
        ]

        speech_dataset = Subset(speech_dataset, indices)
        print("Filtering complete")

    generator = torch.Generator(device=device).manual_seed(1)
    
    train_subset, val_subset, test_subset = torch.utils.data.random_split(
        speech_dataset,
        splits,
        generator=generator
    )

    return speech_dataset, noise_dataset, train_subset, val_subset, test_subset

# Creates dataloaders with transforms from subsets.
def prepare_loaders(
        subsets, 
        train_transforms = nn.Sequential(), 
        val_test_transforms = nn.Sequential(), 
        batch_size = 1000
):
    speech_dataset, noise_dataset, train_subset, val_subset, test_subset = subsets

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

def prepare_spectrogram_loaders(
        batch_size=100,
        n_fft=1024,
        hop_length=512,
        n_mels=64,
        noise_alpha=0,
        precompute=False,
        precompute_dir="./precompute",
        target_labels=None,
        dataset_split=[0.7,0.15,0.15],
        noise_dataset_path="./noise_dataset/audio",
        apply_noise_to_val_test=True,
        device="cpu",
        f_min=20,
        f_max=8000,
    ):
    """
    Prepare train/val/test loaders with optional precompute.
    """
    os.makedirs(precompute_dir, exist_ok=True)

    label_append = "all" if target_labels is None else len(target_labels)
    name_append = f"{float(noise_alpha)}_{label_append}"

    train_file = os.path.join(precompute_dir, f"train_mels_{name_append}.pt")
    val_file = os.path.join(precompute_dir, f"val_mels_{name_append}.pt")
    test_file = os.path.join(precompute_dir, f"test_mels_{name_append}.pt")
    train_labels_file = os.path.join(precompute_dir, f"train_labels_{name_append}.pt")
    val_labels_file = os.path.join(precompute_dir, f"val_labels_{name_append}.pt")
    test_labels_file = os.path.join(precompute_dir, f"test_labels_{name_append}.pt")

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
        
        speech_dataset, noise_dataset, train_subset, val_subset, test_subset = prepare_plain_datasets(
            target_labels=target_labels,
            noise_dataset_path=noise_dataset_path,
            splits=dataset_split,
            device=device
        )
        
        noise_transform = RandomNoise(noise_dataset, noise_alpha).to(device)
        fix_length_transform = FixAudioLength(16000).to(device)
        mel_transform = transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=n_fft,
            #f_min=f_min,
            #f_max=f_max,
            n_mels=n_mels,
            hop_length=hop_length,
            power=2.0,
        ).to(device)

        db_transforms = transforms.AmplitudeToDB(stype="power").to(device)

        all_labels = target_labels if target_labels is not None else sorted(
            list(set(
                path.split(os.sep)[-2]
                for path in speech_dataset._walker
            ))
        )

        label2idx = {label: i for i, label in enumerate(all_labels)}

        # --- UPDATED SECTION START ---
        # Explicitly define the 10 target keywords used in the CNN model
        # target_labels = ['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes']
        # label2idx = {label: i for i, label in enumerate(target_labels)}
        # --- UPDATED SECTION END ---

        def compute_mels(subset, apply_noise):
            mels, labels = [], []
            nr = 1
            for waveform, sample_rate, label, *_ in (speech_dataset[i] for i in subset.indices):
                # --- ADDED FILTERING START ---
                # Only process if the word is in our 10-target list
                # if label not in target_labels:
                #     continue
                # --- ADDED FILTERING END ---

                if nr % 100 == 0:
                    print(f"\rComputing {nr}/{len(subset.indices)}", end="")
                
                if apply_noise:
                    waveform = noise_transform(waveform).to(device)
                waveform = fix_length_transform(waveform).to(device)
                mel = mel_transform(waveform).to(device)
                mel = db_transforms(mel)
                mel = (mel - mel.mean()) / (mel.std() + 1e-6)
                mels.append(mel)
                labels.append(label2idx[label])
                nr += 1
            print("")
            return torch.stack(mels), torch.tensor(labels, dtype=torch.long)

        print("Computing Training Mels")
        train_mels, train_labels = compute_mels(train_subset, apply_noise=True)
        print("Computing Validation Mels")
        val_mels, val_labels = compute_mels(val_subset, apply_noise=apply_noise_to_val_test)
        print("Computing Test Mels")
        test_mels, test_labels = compute_mels(test_subset, apply_noise=apply_noise_to_val_test)

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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader


# Returns loaders with raw audio waveforms.
def prepare_raw_loaders(noise_alpha = 0, batch_size = 1000):
    """
    Prepares loaders with raw audio waveforms.
    """
    subsets = prepare_plain_datasets()

    train_transforms = nn.Sequential(
        RandomNoise(subsets[0], noise_alpha),
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