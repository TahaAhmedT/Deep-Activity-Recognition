import torch
from torchmetrics.classification import Accuracy

def train_step(model: torch.nn.Module, 
               data_loader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer, 
               device: torch.device):
    
    train_loss = 0.0
    model.to(device)
    model.train()
    
    # TorchMetrics Accuracy (for multiclass classification)
    metric_acc = Accuracy(task="multiclass", num_classes=model.fc.out_features).to(device)
    
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

    # Compute final metrics
    epoch_loss = train_loss / len(data_loader)
    epoch_acc = metric_acc.compute().item() * 100  # convert to %
    
    print(f"Train Loss: {epoch_loss:.5f} | Train Accuracy: {epoch_acc:.2f}%")
