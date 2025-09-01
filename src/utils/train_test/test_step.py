import torch
from torchmetrics.classification import Accuracy

def test_step(data_loader: torch.utils.data.DataLoader, 
              model: torch.nn.Module, 
              loss_fn: torch.nn.Module, 
              device: torch.device):
    
    test_loss = 0.0
    model.to(device)
    model.eval()
    
    # TorchMetrics Accuracy (multiclass)
    metric_acc = Accuracy(task="multiclass", num_classes=model.fc.out_features).to(device)
    
    with torch.inference_mode():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)

            # 1. Forward pass
            test_pred = model(data)

            # 2. Loss
            loss = loss_fn(test_pred, target)
            test_loss += loss.item()

            # 3. Update accuracy
            metric_acc.update(test_pred, target)

    # Compute final metrics
    epoch_loss = test_loss / len(data_loader)
    epoch_acc = metric_acc.compute().item() * 100  # %
    
    print(f"Test Loss: {epoch_loss:.5f} | Test Accuracy: {epoch_acc:.2f}%\n")
