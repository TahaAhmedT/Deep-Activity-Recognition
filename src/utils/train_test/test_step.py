import torch

def test_step(data_loader: torch.utils.data.DataLoader,
              model: torch.nn.Module,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device):
    test_loss, test_acc = 0, 0
    model.to(device)
    model.eval() # Put model in eval mode

    # Turn on inference context manager
    with torch.inference_mode():
        for data, target in data_loader:
            # Send data to GPU
            data, target = data.to(device), target.to(device)

            # 1. Forward pass
            test_pred = model(data)

            # 2. Calculate loss and accuracy
            test_loss += loss_fn(test_pred, target)
            test_acc += accuracy_fn(y_true=target,
                                    y_pred=test_pred.argmax(dim=1))
            
        # Adjust metrics and print out
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"Test Loss: {test_loss:.5f} | Test Accuracy: {test_acc:.2f}%\n")