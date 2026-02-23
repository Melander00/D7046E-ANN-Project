import argparse
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset

# -------------------------
# Repro / utils
# -------------------------
EPS = 1e-8

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def set_worker_seed(worker_id: int):
    # called by DataLoader workers
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)

def fix_length(waveform: torch.Tensor, length: int) -> torch.Tensor:
    """Ensure waveform is (1, length)."""
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    T = waveform.shape[-1]
    if T == length:
        return waveform
    if T > length:
        return waveform[..., :length]
    return F.pad(waveform, (0, length - T))

def rms(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(x**2) + EPS)

def mix_with_rms(speech: torch.Tensor, noise: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    alpha = desired noise RMS fraction of speech RMS
    e.g., 0.10 => noise rms = 10% of speech rms
          0.50 => noise rms = 50% of speech rms
    """
    if alpha == 0.0:
        return speech
    rs = rms(speech)
    rn = rms(noise)
    noise_scaled = noise * (alpha * rs / (rn + EPS))
    mixed = speech + noise_scaled
    return torch.clamp(mixed, -1.0, 1.0)

def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# -------------------------
# ESC-50 Noise Sampler
# -------------------------
class ESC50NoiseSampler:
    def __init__(self, esc50_root: str, target_sr: int, clip_samples: int):
        csv_path = os.path.join(esc50_root, "meta", "esc50.csv")
        audio_dir = os.path.join(esc50_root, "audio")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"ESC-50 CSV not found: {csv_path}")
        if not os.path.exists(audio_dir):
            raise FileNotFoundError(f"ESC-50 audio dir not found: {audio_dir}")

        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.audio_dir = audio_dir
        self.target_sr = target_sr
        self.clip_samples = clip_samples
        self._resamplers: Dict[int, torchaudio.transforms.Resample] = {}

    def _resampler(self, sr: int):
        if sr == self.target_sr:
            return None
        if sr not in self._resamplers:
            self._resamplers[sr] = torchaudio.transforms.Resample(sr, self.target_sr)
        return self._resamplers[sr]

    def sample(self) -> torch.Tensor:
        """Return mono noise waveform shaped (1, clip_samples) at target_sr."""
        row = self.df.iloc[random.randint(0, len(self.df) - 1)]
        path = os.path.join(self.audio_dir, row["filename"])
        wav, sr = torchaudio.load(path)  # (C,T)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        r = self._resampler(sr)
        if r is not None:
            wav = r(wav)

        wav = wav.squeeze(0)  # (T,)
        if wav.numel() >= self.clip_samples:
            start = random.randint(0, wav.numel() - self.clip_samples)
            seg = wav[start:start + self.clip_samples]
        else:
            seg = fix_length(wav, self.clip_samples).squeeze(0)

        return torch.clamp(seg.unsqueeze(0), -1.0, 1.0)


# -------------------------
# Speech Commands dataset wrapper
# -------------------------
from torchaudio.datasets import SPEECHCOMMANDS


class SpeechCommandsWrapped(Dataset):
    def __init__(
        self,
        root: str,
        subset: str,
        target_sr: int,
        clip_samples: int,
        allowed_labels: Optional[List[str]] = None,
        noise_sampler: Optional[ESC50NoiseSampler] = None,
        noise_alpha: float = 0.0,
    ):
        self.ds = SPEECHCOMMANDS(root, download=True, subset=subset)
        self.target_sr = target_sr
        self.clip_samples = clip_samples
        self.allowed_labels = allowed_labels
        self.noise_sampler = noise_sampler
        self.noise_alpha = noise_alpha
        self._resamplers: Dict[int, torchaudio.transforms.Resample] = {}

        # indices filtered by label subset
        if allowed_labels is None:
            self.idxs = list(range(len(self.ds)))
        else:
            idxs = []
            for i in range(len(self.ds)):
                _, _, label, *_ = self.ds[i]
                if label in allowed_labels:
                    idxs.append(i)
            self.idxs = idxs

        # Build consistent label mapping from this subset
        labels = sorted(list(set(self.ds[i][2] for i in self.idxs)))
        self.label_to_idx = {l: j for j, l in enumerate(labels)}
        self.idx_to_label = {j: l for l, j in self.label_to_idx.items()}

    def _resampler(self, sr: int):
        if sr == self.target_sr:
            return None
        if sr not in self._resamplers:
            self._resamplers[sr] = torchaudio.transforms.Resample(sr, self.target_sr)
        return self._resamplers[sr]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, int]:
        real_i = self.idxs[i]
        wav, sr, label, *_ = self.ds[real_i]
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        r = self._resampler(sr)
        if r is not None:
            wav = r(wav)

        wav = fix_length(wav, self.clip_samples)

        if self.noise_sampler is not None and self.noise_alpha > 0.0:
            noise = self.noise_sampler.sample()
            wav = mix_with_rms(wav, noise, self.noise_alpha)

        y = self.label_to_idx[label]
        return wav, y


# -------------------------
# Features: log-mel
# -------------------------
class LogMelExtractor(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        n_fft: int,
        hop_length: int,
        n_mels: int,
        f_min: float,
        f_max: float,
    ):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=2.0,
        )
        self.db = torchaudio.transforms.AmplitudeToDB(stype="power")

    @torch.no_grad()
    def forward(self, wav_batch: torch.Tensor) -> torch.Tensor:
        """
        wav_batch: (B,1,T) -> (B,1,n_mels,time)
        Includes per-sample normalization for stability.
        """
        m = self.mel(wav_batch)           # power mel
        mdb = self.db(m + EPS)
        mean = mdb.mean(dim=(2, 3), keepdim=True)
        std = mdb.std(dim=(2, 3), keepdim=True).clamp_min(1e-4)
        return (mdb - mean) / std


# -------------------------
# CNN model
# -------------------------
class CNNCommand(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(96, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x).squeeze(-1).squeeze(-1)
        return self.fc(z)


# -------------------------
# Train / eval
# -------------------------
@torch.no_grad()
def evaluate(model: nn.Module, feat: nn.Module, loader: DataLoader, device: str):
    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0
    ys = []
    preds = []

    for wav, y in loader:
        wav = wav.to(device)
        y = y.to(device)

        x = feat(wav)
        logits = model(x)
        loss = F.cross_entropy(logits, y)

        total_loss += loss.item() * y.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        n += y.size(0)

        ys.append(y.detach().cpu().numpy())
        preds.append(pred.detach().cpu().numpy())

    ys = np.concatenate(ys)
    preds = np.concatenate(preds)
    return total_loss / n, correct / n, ys, preds

def train(model: nn.Module, feat: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
          epochs: int, lr: float, device: str):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_val = -1.0
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        n = 0

        for wav, y in train_loader:
            wav = wav.to(device)
            y = y.to(device)

            x = feat(wav)
            logits = model(x)
            loss = F.cross_entropy(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * y.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            n += y.size(0)

        train_loss = total_loss / n
        train_acc = correct / n

        val_loss, val_acc, _, _ = evaluate(model, feat, val_loader, device)

        print(f"Epoch {ep:02d}/{epochs} | "
              f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--esc50_root",
        type=str,
        default="/content/noise-dataset/ESC-50-master",
        required=False,
        help="Path to ESC-50 root folder (contains meta/ and audio/)."
    )
    parser.add_argument("--data_root", type=str, default="./data", help="Where SpeechCommands will be stored.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1337)

    parser.add_argument("--all_labels", action="store_true",
                        help="Use all labels in SpeechCommands subset (slower). Default uses 10-command subset.")
    args = parser.parse_args()

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    target_sr = 16000
    clip_samples = 16000

    # Recommended 10 commands (fast + usually >=75% clean accuracy)
    allowed_labels = None
    if not args.all_labels:
        allowed_labels = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]

    noise_sampler = ESC50NoiseSampler(args.esc50_root, target_sr=target_sr, clip_samples=clip_samples)

    train_clean = SpeechCommandsWrapped(args.data_root, "training", target_sr, clip_samples,
                                       allowed_labels=allowed_labels, noise_sampler=None, noise_alpha=0.0)
    val_clean = SpeechCommandsWrapped(args.data_root, "validation", target_sr, clip_samples,
                                     allowed_labels=allowed_labels, noise_sampler=None, noise_alpha=0.0)
    test_clean = SpeechCommandsWrapped(args.data_root, "testing", target_sr, clip_samples,
                                      allowed_labels=allowed_labels, noise_sampler=None, noise_alpha=0.0)

    # Ensure label mapping consistent across splits
    if train_clean.label_to_idx != val_clean.label_to_idx or train_clean.label_to_idx != test_clean.label_to_idx:
        raise RuntimeError("Label mappings differ across splits; ensure allowed_labels is consistent.")

    num_classes = len(train_clean.label_to_idx)
    idx_to_label = train_clean.idx_to_label
    print("Num classes:", num_classes)
    print("Labels:", [idx_to_label[i] for i in range(num_classes)])

    def make_loader(ds: Dataset, shuffle: bool):
        return DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=2,
            pin_memory=True,
            worker_init_fn=set_worker_seed,
        )

    train_loader = make_loader(train_clean, shuffle=True)
    val_loader = make_loader(val_clean, shuffle=False)

    # Feature extractor + model
    feat = LogMelExtractor(
        sample_rate=target_sr,
        n_fft=400,
        hop_length=160,
        n_mels=64,
        f_min=20.0,
        f_max=8000.0,
    ).to(device)

    model = CNNCommand(num_classes).to(device)
    print("CNN trainable params:", count_trainable_params(model))

    # Train on clean
    model = train(model, feat, train_loader, val_loader, epochs=args.epochs, lr=args.lr, device=device)

    # Evaluate on noise levels
    noise_levels = {
        "clean_0": 0.0,
        "noise_10": 0.10,
        "noise_50": 0.50,
    }

    for name, alpha in noise_levels.items():
        test_ds = SpeechCommandsWrapped(
            args.data_root,
            "testing",
            target_sr,
            clip_samples,
            allowed_labels=allowed_labels,
            noise_sampler=(noise_sampler if alpha > 0 else None),
            noise_alpha=alpha,
        )
        if test_ds.label_to_idx != train_clean.label_to_idx:
            raise RuntimeError("Test label mapping mismatch.")

        test_loader = make_loader(test_ds, shuffle=False)

        loss, acc, ys, preds = evaluate(model, feat, test_loader, device)
        cm = confusion_matrix(ys, preds, labels=list(range(num_classes)))

        print("\n==============================")
        print(f"CNN / {name} (alpha={alpha})")
        print(f"Loss: {loss:.4f} | Accuracy: {acc:.4f}")
        print("Confusion matrix (rows=true, cols=pred):")
        print(cm)

    # Requirement reminder
    print("\nRequirement check:")
    print("- If clean_0 accuracy < 0.75, try: --epochs 30-40, or keep 10-command subset (default).")

if __name__ == "__main__":
    main()