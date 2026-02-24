from copy import deepcopy

import torch


# Train and validation
def train_epoch(model, criterion, optimizer, train_loader, epoch_number):
    model.train()

    total_loss=0
    correct=0
    guesses=0

    for batch_nr, (inputs, labels) in enumerate(train_loader):
        prediction = model(inputs)

        loss = criterion(prediction, labels)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        guess = torch.argmax(prediction, dim=1)
        correct += (guess == labels).sum().item()
        guesses += labels.size(0)
        
        if batch_nr % 10 == 0:
            print("\rBatch [{}/{}]".format(
                batch_nr+1, len(train_loader)
            ), end="")

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / guesses
    return avg_loss, accuracy


def validate_epoch(model, criterion, val_loader, epoch_number):
    model.eval()

    total_loss=0
    correct=0
    guesses=0

    with torch.no_grad():
        for batch_nr, (inputs, labels) in enumerate(val_loader):
            prediction = model(inputs)

            loss = criterion(prediction, labels)
            total_loss += loss.item()

            guess = torch.argmax(prediction, dim=1)
            correct += (guess == labels).sum().item()
            guesses += labels.size(0)
    
    avg_loss = total_loss / len(val_loader)
    accuracy = correct / guesses
    return avg_loss, accuracy


def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs):

    best_val_accuracy = 0
    best_model = None

    train_losses = []
    val_losses = []

    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        train_loss, train_accuracy = train_epoch(model, criterion, optimizer, train_loader, epoch)
        val_loss, val_accuracy = validate_epoch(model, criterion, val_loader, epoch)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        
        print(f"\rEpoch [{epoch+1}/{num_epochs}]                                            \n \tLoss: Train={round(train_loss * 1000)/1000}; Val={round(val_loss * 1000) / 1000}\n \tAcc: Train={round(train_accuracy * 10000) / 100}%; Val={round(val_accuracy * 10000) / 100}%")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model = deepcopy(model)
    
    losses = (train_losses, val_losses)
    accuracies = (train_accuracies, val_accuracies)

    return best_model, losses, accuracies


# Testing
def test_model(model, test_loader):
    model.eval()
    
    correct=0
    guesses=0

    c_predictions = []
    c_labels = []

    with torch.no_grad():
        for batch_nr, (inputs, labels) in enumerate(test_loader):
            prediction = model(inputs)
            guess = torch.argmax(prediction, dim=1)
            correct += (guess == labels).sum().item()
            guesses += labels.size(0)

            c_predictions.append(guess)
            c_labels.append(labels)

    accuracy = correct / guesses if guesses > 0 else 0

    # Flatten mini-batches into one big set.
    c_predictions = torch.cat(c_predictions)
    c_labels = torch.cat(c_labels)

    # Find the number of classes
    num_classes = int(torch.max(torch.cat([c_predictions, c_labels])).item()) + 1
    confusion_matrix = torch.zeros(num_classes, num_classes)

    for label, prediction in zip(c_labels, c_predictions):
        confusion_matrix[label, prediction] += 1

    return accuracy, confusion_matrix