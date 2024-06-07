import torch


def set_default_device() -> None:
    """Sets the GPU as standard device if there is one available"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_default_device(device)

