from src.utils.logging_utils import setup_logger
from src.utils.config_utils import load_config

import torch
from torcheval.metrics.functional import multiclass_f1_score
from torchmetrics.classification import Accuracy

CONFIG = load_config()

def test_step(data_loader: torch.utils.data.DataLoader, 
              model: torch.nn.Module, 
              loss_fn: torch.nn.Module, 
              device: torch.device,
              logs_path,
              verbose: bool = True):
    """Runs a test/validation step for one epoch.

    Args:
        data_loader (torch.utils.data.DataLoader): DataLoader for test/validation data.
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
    test_loss = 0.0
    y_true = []
    y_pred = []
    model.to(device)
    model.eval()

    # TorchMetrics Accuracy (multiclass)
    metric_acc = Accuracy(task="multiclass", num_classes=CONFIG["NUM_CLASSES"]).to(device)
    
    with torch.inference_mode():
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)

            # 1. Forward pass
            test_pred = model(data)

            # 2. Loss
            loss = loss_fn(test_pred, target)
            test_loss += loss.item()

            y_true.extend(target.cpu().numpy())
            y_pred.extend(torch.argmax(test_pred, dim=1).cpu().numpy())

            # 3. Update accuracy
            metric_acc.update(test_pred, target)
            if (batch_idx + 1) % 100 == 0:
                logger.info(f"batch #{batch_idx+1}/{len(data_loader)} Loss: {loss}")

    # Compute final metrics
    y_true = torch.tensor(y_true)
    y_pred = torch.tensor(y_pred)
    epoch_f1_score = multiclass_f1_score(y_pred, y_true, num_classes=8)
    epoch_loss = test_loss / len(data_loader)
    epoch_acc = metric_acc.compute().item() * 100  # %
    
    logger.info(f"Test Loss: {epoch_loss:.5f} | Test Accuracy: {epoch_acc:.2f}% | Test F1-score: {epoch_f1_score}\n")

    return epoch_f1_score, epoch_loss, epoch_acc, y_true.numpy(), y_pred.numpy()