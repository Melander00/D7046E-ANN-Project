import json
import os

import matplotlib.pyplot as plt
import numpy as np


def plot_graph(
    train_values = [],
    train_label = "Train Values",
    val_values = [],
    val_label = "Validation Values",
    vertical_lines = [],
    horizontal_lines = [],
    xlabel = "Epochs",
    ylabel = "Loss",
    title = "Losses per epoch",
    show = True,
    save = False,
    save_dir = "./img",
    name = "plot.png"
):
    fig, ax = plt.subplots()
    fig.set_size_inches(6,6)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)

    plt.plot(range(1, len(train_values)+1), train_values, label=train_label)
    plt.plot(range(1, len(val_values)+1), val_values, label=val_label)

    for line, c, label in vertical_lines:
        plt.vlines(line, 0, max(max(train_values), max(val_values)), linestyles="dashed", colors=c, label=label)

    for line, c, label in horizontal_lines:
        plt.hlines(line, 1, max(len(train_values), len(val_values)), linestyles="dashed", colors=c, label=label)

    plt.legend()
    if show:
        plt.show()
    if save:
        plt.savefig(os.path.join(save_dir, name))
        print(f"Saved {name} to {save_dir}")

def load_model_json(filename, file_dir = "./output"):
    with open(os.path.join(file_dir, filename), "r") as file:
        data = json.load(file)
    return data

def plot_losses(data, save_dir="", save=False):
    losses = data["losses"]
    train_loss = losses[0]
    val_loss = losses[1]
    model = data["model"]
    alpha = data["alpha"]

    name = f"{model}_loss_{alpha}.png"

    plot_graph(
        train_values=train_loss,
        val_values=val_loss,
        show=False,
        train_label="Train Losses",
        val_label="Validation Losses",
        title=f"{model.upper()} losses for alpha={alpha}",
        xlabel="Epochs",
        ylabel="Loss",
        save=save,
        save_dir=save_dir,
        name=name,
    ) 

def plot_accuracies(data, save_dir="", save=False):
    accs = data["accuracies"]
    train_acc = accs[0]
    val_acc = accs[1]
    model = data["model"]
    alpha = data["alpha"]
    test_acc = data["test_accuracy"]

    name = f"{model}_acc_{alpha}.png"

    plot_graph(
        train_values=train_acc,
        val_values=val_acc,
        show=False,
        train_label="Train Accuracy",
        val_label="Validation Accuracy",
        title=f"{model.upper()} accuracies for alpha={alpha}",
        horizontal_lines=[(test_acc, "r", "Test Accuracy")],
        xlabel="Epochs",
        ylabel="Accuracy [0;1]",
        save=save,
        save_dir=save_dir,
        name=name,
    )

def plot_all_files_in_dir(dir, save_dir="", save=False):
    everything = os.listdir(dir)
    for file_path in everything:
        if os.path.isfile(os.path.join(dir, file_path)) and file_path.endswith(".json"):
            data = load_model_json(file_path, dir)
            plot_losses(data, save_dir=save_dir, save=save)
            plot_accuracies(data, save_dir=save_dir, save=save)


def main():
    img_dir = "./img"
    data_dir = "./output"

    os.makedirs(img_dir, exist_ok=True)

    plot_all_files_in_dir(data_dir, save_dir=img_dir, save=True)

    # plt.show()
    
if __name__ == "__main__":
    main()