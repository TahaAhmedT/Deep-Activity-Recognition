from stores import ModelProvidersFactory, DatasetProvidersFactory
from utils.logging_utils import setup_logger
from utils.checkpoints_utils import load_checkpoint
from utils.visualize_utils import plot_confusion_matrix

import numpy as np
import torch
import torch.nn as nn
from torcheval.metrics.functional import multiclass_f1_score
from torchmetrics.classification import Accuracy


def test_step(data_loader: torch.utils.data.DataLoader, 
              model: torch.nn.Module, 
              loss_fn: torch.nn.Module, 
              device: torch.device,
              num_classes,
              verbose: bool = True):
    """
    Runs a test step for full dataset evaluation.

    Args:
        data_loader (torch.utils.data.DataLoader): DataLoader for test data.
        model (torch.nn.Module): Model to evaluate.
        loss_fn (torch.nn.Module): Loss function.
        device (torch.device): Device to run evaluation on.
        num_classes (int): Number of classes.
        verbose (bool): Kept for API compatibility (not used).

    Returns:
        tuple: (epoch_f1, epoch_loss, epoch_acc, y_true, y_pred)
    """

    test_loss = 0.0
    y_true = []
    y_pred = []

    model.to(device)
    model.eval()

    # Accuracy metric
    metric_acc = Accuracy(task="multiclass", num_classes=num_classes).to(device)

    with torch.inference_mode():
        for data, target in data_loader:

            if isinstance(data, list):
                data, target = np.array(data), np.array(target)

            data = torch.tensor(data, dtype=torch.float32).to(device)
            target = torch.tensor(target, dtype=torch.long).to(device)

            # Forward
            logits = model(data)

            # Loss
            loss = loss_fn(logits, target)
            test_loss += loss.item()

            # Predictions
            preds = torch.argmax(logits, dim=1)

            y_true.extend(target.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

            # Update accuracy
            metric_acc.update(logits, target)

    # Final metrics
    y_true_tensor = torch.tensor(y_true)
    y_pred_tensor = torch.tensor(y_pred)

    final_f1 = multiclass_f1_score(
        y_pred_tensor,
        y_true_tensor,
        num_classes=num_classes,
        average="weighted"
    )

    final_loss = test_loss / len(data_loader)
    final_acc = metric_acc.compute().item() * 100

    return final_f1, final_loss, final_acc, y_true, y_pred

def test(log_dir: str,
        batch_size: int,
        videos_root: str,
        annot_root: str,
        train_ids: list[int],
        test_ids: list[int],
        features: bool,
        model_name: str,
        checkpoint: str,
        num_classes: int,
        actions_dict: dict,
        save_path: str,
        figname: str,
        image_level: bool = None,
        crop: bool = None,
        output_file: str = None,
        input_size: int = None,
        hidden_size1: int = None,
        hidden_size2: int = None,
        num_layers: int = None,
        sequence: bool = None,
        verbose: bool = False):

    logger = setup_logger(
            log_file=__file__,
            log_dir=log_dir,
            log_to_console=verbose,
            use_tqdm=True
        )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using Device: {device} for Testing...")

    dataset_factory = DatasetProvidersFactory()
    logger.info("Preparing Test Dataloader...")
    _, testloader = dataset_factory.get_data_loaders(
                                                batch_size,
                                                videos_root,
                                                annot_root,
                                                train_ids,
                                                test_ids,
                                                features,
                                                log_dir,
                                                actions_dict,
                                                output_file,
                                                image_level,
                                                crop,
                                                sequence,
                                                verbose
                                            )
    logger.info("Test Dataloader Loaded Successfully!")
    
    models_factory = ModelProvidersFactory()
    model = models_factory.create(model_name=model_name, num_classes=num_classes,
                                         input_size=input_size, hidden_size1=hidden_size1,
                                         hidden_size2=hidden_size2, num_layers=num_layers,
                                         log_dir=log_dir, verbose=verbose)
    
    # Load a checkpoint saved during training
    logger.info("Loading the Model's Checkpoint...")
    checkpoint = torch.load(checkpoint)

    # Load trained weights into the model
    model = load_checkpoint(checkpoint=checkpoint, model=model)
    
    criterion = nn.CrossEntropyLoss()

    logger.info("Starting Testing...")
    f1_score, loss, acc, y_true, y_pred = test_step(testloader, model, criterion, device, num_classes, verbose)
    logger.info(f"Testing Finished with Accuracy = {acc} | Loss = {loss} | F1-score = {f1_score}")
    plot_confusion_matrix(y_true=y_true, y_pred=y_pred, save_path=save_path, figname=figname)
    logger.info(f"{figname} Plotted Successfully!")
    
