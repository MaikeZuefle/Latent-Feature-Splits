import logging

import torch

# Cuda
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Save and Load Functions
def save_checkpoint(save_path, model, valid_loss):
    """Saves a checkpoint of the model and the corresponding validation loss.

    Args:
        save_path: String defining the path where to save the model
        model: the model to save
        valid_loss: the validation loss to save
    """

    if save_path is None:
        return

    state_dict = {"model_state_dict": model.state_dict(), "valid_loss": valid_loss}

    torch.save(state_dict, save_path)
    logging.info(f"Model saved to ==> {save_path}")


def load_checkpoint(load_path, model):
    """
        Checks if the model already has checkpoints and if that's the case loads the checkpoints.

        Args:
            load_path (string): path where potential checkpoints are stored
            model (LanguageModel): model where the checkpoint should be used

        Returns:
            float: returns last valid loss
        """

    if load_path is None:
        return

    state_dict = torch.load(load_path, map_location=device)

    logging.info(f"Model loaded from <== {load_path}")

    model.load_state_dict(state_dict["model_state_dict"])
    return state_dict["valid_loss"]
