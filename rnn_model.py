import os
import shutil

import torch
import torch.nn as nn
import torch.optim as optim

from data_preprocessing import prepare_spectrogram_loaders
from model_trainer import test_model, train_model


# 1. Define the LSTM Architecture
class RNNCommand(nn.Module):
    def __init__(self, input_size=64, hidden_size=68, num_layers=2, num_classes=10):
        super(RNNCommand, self).__init__()
        # 68 hidden size = 74,674 params (Fair match for CNN's 75,562)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # [batch, 1, 64, 32] -> [batch, 32, 64] (time, freq)
        x = x.squeeze(1).transpose(1, 2) 
        out, _ = self.lstm(x)
        # Many-to-One: Use the last hidden state of the sequence
        return self.fc(out[:, -1, :])

def main():
    device = torch.device("cpu")
    print(f"Device: {device}")

    model = RNNCommand(input_size=64, hidden_size=68, num_layers=2, num_classes=10).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"RNN trainable params: {params}")

    # 2. Train on Clean Data
    # Wipe cache once to ensure the new 10-label filter from data_preprocessing is applied
    if os.path.exists("precompute"):
        shutil.rmtree("precompute")
    
    print("\n--- Initializing Training Data (10 Labels) ---")
    train_loader, val_loader, _ = prepare_spectrogram_loaders(batch_size=64, n_mels=64, noise_alpha=0)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("\n--- Training RNN ---")
    train_model(model, criterion, optimizer, train_loader, val_loader, 20)

    # 3. Robustness Sweep (Evaluating 0%, 10%, and 50% Noise)
    noise_levels = [0.0, 0.1, 0.5]
    for alpha in noise_levels:
        print(f"\n" + "="*40)
        print(f"RNN TEST / alpha={alpha} ({int(alpha*100)}% Noise)")
        
        # Wipe precompute folder so it doesn't reuse clean data for noisy tests
        if os.path.exists("precompute"):
            shutil.rmtree("precompute")
        
        _, _, test_loader = prepare_spectrogram_loaders(batch_size=64, n_mels=64, noise_alpha=alpha)
        
        acc, cm = test_model(model, test_loader)
        print(f"Test Accuracy: {acc:.4f}")
        print("Confusion Matrix:")
        print(cm)

    torch.save(model.state_dict(), "rnn_speech_commands_final.pth")
    print("\nFinal model saved.")




def main_multiple_training():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    torch.set_default_device(device)

    # Train on multiple noise levels
    noise_levels = [0.0, 0.1, 0.5]
    for alpha in noise_levels:
        print(f"\n\n === Noise Level {alpha} ===")

        model = RNNCommand(input_size=64, hidden_size=68, num_layers=2, num_classes=10).to(device)
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