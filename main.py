# Imports
import os
import shutil

import torch
import torch.nn as nn
import torch.optim as optim

from cnn_model import CNNCommand
from data_preprocessing import prepare_spectrogram_loaders
from graph_handler import plot_confusion_matrix
from model_trainer import test_model, train_model
from rnn_model import RNNCommand


def train_and_test_model(
    model_fn,
    target_labels = None,
    noise_dataset_path = "./noise_dataset",
    learning_rate=1e-3,
    num_epochs=20,
    batch_size=64
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    torch.set_default_device(device)

    outputs = []

    noise_levels = [0.0, 0.1, 0.5]
    for alpha in noise_levels:
        print(f"\n\n === Noise Level {alpha} ===")

        # model = RNNCommand(input_size=64, hidden_size=68, num_layers=2, num_classes=10).to(device)
        model = model_fn().to(device)

        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"RNN trainable params: {params}")

        train_loader, val_loader, test_loader = prepare_spectrogram_loaders(
            batch_size=batch_size, 
            n_mels=64, 
            noise_alpha=alpha,
            target_labels = target_labels,
            noise_dataset_path=noise_dataset_path,
            precompute=True
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        best_model, losses, accuracies = train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs)

        test_accuracy, cm = test_model(best_model, test_loader)
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print("Confusion Matrix:")
        print(cm)
        outputs.append((best_model, losses, accuracies, test_accuracy, cm))
    return outputs



def create_rnn_fn(
    input_size=64,
    hidden_size=68,
    num_layers=2,
    num_classes=10
):
    def create_fn():
        return RNNCommand(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=num_classes
        )
    return create_fn

def create_cnn_fn(num_classes=10):
    def create_fn():
        return CNNCommand(
            num_classes=num_classes
        )
    return create_fn
    

def main():
    # === Arguments ===
    noise_dataset_path="./noise_dataset"


    target_labels = ['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes']
    # create_fn = create_rnn_fn(num_classes=len(target_labels))    
    create_fn = create_cnn_fn(num_classes=len(target_labels))    
    # === End Arguments ===

    outputs = train_and_test_model(
        create_fn,
        target_labels=target_labels,
        noise_dataset_path=noise_dataset_path
    )

    plot_confusion_matrix(outputs[0][4], target_labels)



if __name__ == "__main__":
    main()