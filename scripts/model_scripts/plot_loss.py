import os

import matplotlib.pyplot as plt
from load_safe_metrics import load_metrics


def plot_loss(destination_path, filename="metrics.pt"):
    """
    Plots the loss progression.

    Args:
        destination_path (str): path where the plot should be stored
        filename (str): filename where the plot should be stored, default: metrics.pt
    """

    train_loss_list, valid_loss_list, global_steps_list = load_metrics(
        os.path.join(destination_path, filename)
    )
    plt.plot(global_steps_list, train_loss_list, label="Train")
    plt.plot(global_steps_list, valid_loss_list, label="Valid")
    plt.xlabel("Global Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(destination_path, "loss.png"))
    plt.close()
