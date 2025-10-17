import torch

def save_checkpoint(state: dict, filename: str):
    """Saves the model state to a file.

    Args:
        state (dict): State dictionary containing model state and other info.
        filename (str): Path to the file where the state will be saved.
    """
    torch.save(state, filename)

def load_checkpoint(checkpoint, model: torch.nn.Module):
    """Loads the model state from a checkpoint.

    Args:
        checkpoint (str or dict): Path to the checkpoint file or the checkpoint dictionary.
        model (torch.nn.Module): Model to load the state into.
        optimizer (torch.optim.Optimizer, optional): Optimizer to load the state into. Defaults to None.
    """
    model.load_state_dict(checkpoint['state_dict'])
    return model