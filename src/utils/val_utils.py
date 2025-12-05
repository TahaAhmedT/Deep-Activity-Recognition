from .logging_utils import setup_logger

import numpy as np
import torch
from torcheval.metrics.functional import multiclass_f1_score
from torchmetrics.classification import Accuracy


def val_step(data_loader: torch.utils.data.DataLoader, 
              model: torch.nn.Module, 
              loss_fn: torch.nn.Module, 
              device: torch.device,
              logs_path,
              num_classes,
              verbose: bool = True):
    """Runs a validation step for one epoch.

    Args:
        data_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        model (torch.nn.Module): Model to evaluate.
        loss_fn (torch.nn.Module): Loss function.
        device (torch.device): Device to run evaluation on.
        verbose (bool, optional): If True, prints info logs. Defaults to False.

    Returns:
        tuple: (epoch_loss, epoch_acc)
    """
    logger = setup_logger(
            log_file=__file__,
            log_dir=logs_path,
            log_to_console=verbose,
            use_tqdm=True,
        )
    
    val_loss = 0.0
    y_true = []
    y_pred = []
    model.to(device)
    model.eval()

    # TorchMetrics Accuracy (multiclass)
    metric_acc = Accuracy(task="multiclass", num_classes=num_classes).to(device)
    
    with torch.inference_mode():
        for batch_idx, (data, target) in enumerate(data_loader):
            if isinstance(data, list):
                # Frist convert to numpy array
                data, target = np.array(data), np.array(target)
            data, target = torch.tensor(data, dtype=torch.float32), torch.tensor(target, dtype=torch.long) # to ensure data type is float32 not 64
            data, target = data.to(device), target.to(device)

            # 1. Forward pass
            val_pred = model(data)

            # 2. Loss
            loss = loss_fn(val_pred, target)
            val_loss += loss.item()

            y_true.extend(target.cpu().numpy())
            y_pred.extend(torch.argmax(val_pred, dim=1).cpu().numpy())

            # 3. Update accuracy
            metric_acc.update(val_pred, target)
            # if (batch_idx + 1) % 100 == 0:
            #     logger.info(f"batch #{batch_idx+1}/{len(data_loader)} Loss: {loss}")

    # Compute final metrics
    y_true = torch.tensor(y_true)
    y_pred = torch.tensor(y_pred)
    epoch_f1_score = multiclass_f1_score(y_pred, y_true, num_classes=num_classes, average="wheighted")
    epoch_loss = val_loss / len(data_loader)
    epoch_acc = metric_acc.compute().item() * 100  # %
    
    logger.info(f"Validation Loss: {epoch_loss:.5f} | Validation Accuracy: {epoch_acc:.2f}% | Validation F1-score: {epoch_f1_score}\n")

    return epoch_f1_score, epoch_loss, epoch_acc, y_true.numpy(), y_pred.numpy()
