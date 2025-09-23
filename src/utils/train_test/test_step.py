import torch
from torchmetrics.classification import Accuracy

def test_step(data_loader: torch.utils.data.DataLoader, 
              model: torch.nn.Module, 
              loss_fn: torch.nn.Module, 
              device: torch.device,
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
    test_loss = 0.0
    model.to(device)
    model.eval()

    # TorchMetrics Accuracy (multiclass)
    metric_acc = Accuracy(task="multiclass", num_classes=8).to(device)
    
    with torch.inference_mode():
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            if verbose:
                print(f"[INFO] Testing batch {batch_idx+1}/{len(data_loader)}")

            # 1. Forward pass
            test_pred = model(data)

            # 2. Loss
            loss = loss_fn(test_pred, target)
            test_loss += loss.item()

            # 3. Update accuracy
            metric_acc.update(test_pred, target)
            if verbose:
                print(f"batch #{batch_idx+1} Loss: {loss}")

    # Compute final metrics
    epoch_loss = test_loss / len(data_loader)
    epoch_acc = metric_acc.compute().item() * 100  # %
    
    print(f"Test Loss: {epoch_loss:.5f} | Test Accuracy: {epoch_acc:.2f}%\n")
    if verbose:
        print(f"[INFO] Final Test Loss: {epoch_loss:.5f}, Test Accuracy: {epoch_acc:.2f}%")
    return epoch_loss, epoch_acc