
# Imports
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from IPython.display import Audio, display
from torch.utils.data import DataLoader, Dataset, Subset
from torchaudio import transforms


# Train and validation
def train_epoch(model, criterion, optimizer, train_loader, epoch_number):
  model.train()
  pass


def validate_epoch(model, criterion, val_loader, epoch_number):
  model.eval()
  pass


def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs):
  for epoch in range(num_epochs):
    train_epoch(model, criterion, optimizer, train_loader, epoch)
    validate_epoch(model, criterion, val_loader, epoch)
  pass



# Testing
def test_model():

  pass





def main():
  pass


if __name__ == "__main__":
  main()