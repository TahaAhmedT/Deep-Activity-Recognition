from src.baseline1.dataset import B1Dataset
from src.baseline1.extended_model import ExtendedModel
from src.utils.train_test.train_step import train_step
from src.utils.train_test.test_step import test_step
from src.utils.config_utils.load_config import load_config
from src.utils.stream_utils.stream_utils import log_stream

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader

def get_data_loaders(config, verbose=False):
    """Creates train and test data loaders.

    Args:
        config (dict): Configuration dictionary.
        verbose (bool): If True, prints info logs.

    Returns:
        tuple: (trainloader, testloader)
    """
    if verbose:
        print("[INFO] Creating data loaders...")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    batch_size = config["TRAINING_PARAMS"]["batch_size"]
    train_dataset = B1Dataset(
        videos_root=config["PATH"]["videos_root"],
        target_videos=config["TARGET_VIDEOS"]["train_ids"],
        transform=transform
    )
    test_dataset = B1Dataset(
        videos_root=config["PATH"]["videos_root"],
        target_videos=config["TARGET_VIDEOS"]["val_ids"],
        transform=transform
    )
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    if verbose:
        print("[INFO] Data loaders created.")
    return trainloader, testloader

def get_model(verbose=False):
    """Initializes the ResNet50 model and truncates the last layer.

    Args:
        verbose (bool): If True, prints info logs.

    Returns:
        nn.Module: Modified ResNet50 model wrapped in ExtendedModel.
    """
    if verbose:
        print("[INFO] Initializing ResNet50 model...")
    original_model = resnet50(weights=ResNet50_Weights.DEFAULT, progress=verbose)
    layers = list(original_model.children())[:-1]
    truncated_model = nn.Sequential(*layers)
    model = ExtendedModel(truncated_model)
    if verbose:
        print("[INFO] Model initialized and truncated.")
    return model

def get_optimizer(model, config, verbose=False):
    """Creates the optimizer for training.

    Args:
        model (nn.Module): Model to optimize.
        config (dict): Configuration dictionary.
        verbose (bool): If True, prints info logs.

    Returns:
        torch.optim.Optimizer: AdamW optimizer.
    """
    if verbose:
        print("[INFO] Creating optimizer...")
    lr = config["TRAINING_PARAMS"]["lr"]
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    if verbose:
        print(f"[INFO] Optimizer created with learning rate {lr}.")
    return optimizer

def main(verbose=True):
    """Main function to run training and testing loop.

    Args:
        verbose (bool): If True, prints info logs.
    """
    CONFIG = load_config()
    log_stream(log_file="finetune_resnet50_logs", prog="baselines_logs/baseline1_logs", verbose=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"[INFO] Using device: {device}")
    trainloader, testloader = get_data_loaders(CONFIG, verbose=verbose)
    model = get_model(verbose=verbose)
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, CONFIG, verbose=verbose)
    num_epochs = CONFIG["TRAINING_PARAMS"]["num_epochs"]

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}\n-----------------------")
        # Training step
        if verbose:
            print("[INFO] Starting training step...")
        train_step(
            data_loader=trainloader,
            model=model,
            loss_fn=criterion,
            optimizer=optimizer,
            device=device
        )
        if verbose:
            print("[INFO] Training step completed.")
            print("[INFO] Starting testing step...")
        # Testing step
        test_step(
            data_loader=testloader,
            model=model,
            loss_fn=criterion,
            device=device
        )
        if verbose:
            print("[INFO] Testing step completed.")

if __name__ == "__main__":
    main()