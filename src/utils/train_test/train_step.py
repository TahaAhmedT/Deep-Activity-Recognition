from src.utils.logging_utils.logging_utils import setup_logger

import torch
from torchmetrics.classification import Accuracy

def train_step(model: torch.nn.Module, 
               data_loader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               scheduler: torch.optim.lr_scheduler._LRScheduler,
               device: torch.device,
               verbose: bool = True):
    """Runs a training step for one epoch.

    Args:
        model (torch.nn.Module): Model to train.
        data_loader (torch.utils.data.DataLoader): DataLoader for training data.
        loss_fn (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        device (torch.device): Device to run training on.
        verbose (bool, optional): If True, prints info logs. Defaults to False.

    Returns:
        tuple: (epoch_loss, epoch_acc)
    """
    logger = setup_logger(
            log_file=__file__,
            log_dir="logs/baselines_logs/baseline1_logs",
            log_to_console=verbose,
            use_tqdm=True,
        )
    train_loss = 0.0
    model.to(device)
    model.train()

    # TorchMetrics Accuracy (for multiclass classification)
    metric_acc = Accuracy(task="multiclass", num_classes=8).to(device)
    
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)

        # 1. Forward pass
        y_pred = model(data)

        # 2. Calculate loss
        loss = loss_fn(y_pred, target)
        train_loss += loss.item()

        # 3. Update accuracy metric
        metric_acc.update(y_pred, target)

        # 4. Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (batch_idx + 1) % 100 == 0:
            print(f"batch #{batch_idx+1}/{len(data_loader)} Loss: {loss}")

    # Compute final metrics
    epoch_loss = train_loss / len(data_loader)
    epoch_acc = metric_acc.compute().item() * 100  # convert to %
    
    logger.info(f"Train Loss: {epoch_loss:.5f} | Train Accuracy: {epoch_acc:.2f}%")
    
    return model, epoch_loss, epoch_acc