import os
import shutil

import torch
import torch.nn as nn
import torch.optim as optim

from data_preprocessing import prepare_spectrogram_loaders
from model_trainer import test_model, train_model


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

def main_multiple_training():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    torch.set_default_device(device)

    # Train on multiple noise levels
    noise_levels = [0.0, 0.1, 0.5]
    for alpha in noise_levels:
        print(f"\n\n === Noise Level {alpha} ===")

        model = CNNCommand(num_classes=10).to(device)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"RNN trainable params: {params}")

        train_loader, val_loader, test_loader = prepare_spectrogram_loaders(
            batch_size=64, 
            n_mels=64, 
            noise_alpha=alpha,
            target_labels = ['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes'],
            noise_dataset_path="./noise_dataset",
            precompute=True
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        best_model, losses, accuracies = train_model(model, criterion, optimizer, train_loader, val_loader, 20)

        test_accuracy, cm = test_model(best_model, test_loader)
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print("Confusion Matrix:")
        print(cm)
    pass


if __name__ == "__main__":
    # main()
    main_multiple_training()