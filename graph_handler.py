import os

import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(
    cm, 
    class_names,
    save=True,
    save_dir="./output/",
    save_name="confusion_matrix.png",
    show=False,
):
    fig, ax = plt.subplots()
    num_classes = max(3, len(class_names) * 1)
    fig.set_size_inches(num_classes, num_classes)

    cm_normalized = cm.float() / cm.sum(dim=1, keepdim=True)

    im = ax.imshow(cm_normalized)

    plt.colorbar(im)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    # Show numbers inside cells
    for i in range(cm_normalized.shape[0]):
        for j in range(cm_normalized.shape[1]):
            ax.text(j, i, f'{round(cm_normalized[i, j].item() * 100)}%',
                    ha="center", va="center", color=("black" if cm_normalized[i, j].item() > 0.2 else "white"))

    plt.tight_layout()
    if show:
        plt.show()
    if save:
        plt.savefig(os.path.join(save_dir, save_name))
