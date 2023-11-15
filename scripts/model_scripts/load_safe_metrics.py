import logging

import torch

# Cuda
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list):
    """
    Saves train_losses, valid_losses and global steps in an dictionary.

    Args:
        save_path (str): path to stores losses and steps
        train_loss_list (list): list containing train losses
        valid_loss_list (list): list containing valid losses
        global_steps_list (list): list containig global steps
    """

    if save_path is None:
        return

    state_dict = {
        "train_loss_list": train_loss_list,
        "valid_loss_list": valid_loss_list,
        "global_steps_list": global_steps_list,
    }

    torch.save(state_dict, save_path)
    logging.info(f"Metrics saved to ==> {save_path}")


def load_metrics(load_path):
    """
    Loads the state_dict containing train_losses, valid_losses and global steps.

    Args:
        load_path (str): path where the state_dict is stored

    Returns:
        lists: return a list of train_losses, a list of valid_losses and a global_steps_list
    """

    if load_path is None:
        return

    state_dict = torch.load(load_path, map_location=device)
    logging.info(f"Metrics loaded from <== {load_path}")

    return (
        state_dict["train_loss_list"],
        state_dict["valid_loss_list"],
        state_dict["global_steps_list"],
    )
