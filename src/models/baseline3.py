"""
This module defines a simple feed-forward artificial neural network (ANN) used in baseline3.
The ANN accepts extracted features as input and outputs class logits for volleyball activity recognition.
It is intended to be used as a lightweight classifier on top of pretrained feature extractors.
"""
from ..utils import setup_logger

from torch import nn


class Group_Activity_Classifier(nn.Module):
    """Feed-forward Artificial Neural Network for classification.

    The network consists of a simple fully connected stack that maps input features
    to class logits. It is designed to take feature vectors (e.g., pooled CNN features)
    and produce predictions for activity recognition.

    Attributes:
        input_size (int): Dimensionality of the input feature vector.
        fc_layer (nn.Sequential): Fully connected layers producing output logits.
    """

    def __init__(self, input_size, n_classes, log_dir, verbose):
        """Initializes the ANN model.

        Args:
            input_size (int): Number of input features.
            n_classes (int): Number of output classes.

        The network architecture:
            Linear(input_size -> 1000) -> Linear(1000 -> n_classes)
        """
        super().__init__()
        self.logger = setup_logger(
            log_file=__file__,
            log_dir=log_dir,
            log_to_console=verbose,
            use_tqdm=True,
        )
        self.logger.info("Initializing Our ANN Model...")
        self.input_size = input_size
        self.fc_layer = nn.Sequential(
            nn.Linear(input_size, 1000),
            nn.Linear(1000, n_classes)
        )
        self.logger.info(f"ANN Model Initialized with Input Size = {self.input_size} and Output Size = {n_classes}.")
    

    def forward(self, x):
        """Performs a forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, n_classes).
        """
        x = self.fc_layer(x)
        return x
