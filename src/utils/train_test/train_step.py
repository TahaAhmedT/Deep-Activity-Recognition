import torch

def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device):
    train_loss, train_acc = 0, 0
    model.to(device)
    for batch_idx, (data, target) in enumerate(data_loader):
        # Send data to GPU
        data, target = data.to(device), target.to(device)

        # 1. Forward pass
        y_pred = model(data)

        # 2. Calculate loss
        loss = loss_fn(y_pred, target)
        train_loss += loss
        train_acc += accuracy_fn(y_true=target,
                                 y_pred=y_pred.argmax(dim=1))
        
        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()
    
    # Calculate loss and accuracy per epoch and print out what's happening
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train Loss: {train_loss:.5f} | Train Accuracy: {train_acc:.2f}%")
    