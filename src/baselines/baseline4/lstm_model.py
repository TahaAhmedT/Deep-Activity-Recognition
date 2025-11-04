"""
Temporal classifier based on LSTM for group activity recognition in volleyball.

This module defines Group_Activity_Temporal_Classifier which consumes a sequence of
frame-level feature vectors (e.g., extracted fine-tuned ResNet50) for each clip
and produces clip-level activity logits. The model first encodes temporal information
with an LSTM and then combines the LSTM output with the original per-frame features
before applying a fully-connected classification head.

Expected input shape:
    (batch_size, seq_len, feature_dim)
"""
from src.utils.logging_utils import setup_logger
from src.utils.config_utils import load_config

import torch
import torch.nn as nn


CONFIG = load_config()
logger = setup_logger(
            log_file=__file__,
            log_dir=CONFIG["baseline4_logs"],
            log_to_console=CONFIG["verbose"],
            use_tqdm=True,
        )


class Group_Activity_Temporal_Classifier(nn.Module):
    """LSTM-based temporal classifier for group activity recognition.

    The model encodes temporal dynamics with an LSTM and combines the LSTM's sequence
    output with the original input features before classification. This design lets
    the classifier leverage both the per-frame CNN features and the temporal context
    captured by the LSTM.

    Attributes:
        lstm (nn.LSTM): LSTM module that processes the input sequence.
        fc (nn.Sequential): Fully-connected classification head that maps the
            concatenated features to class logits.
    """

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        """Initializes the temporal classifier.

        Args:
            num_classes (int): Number of target activity classes.
            input_size (int): Dimensionality of input feature vectors per time step.
            hidden_size (int): Hidden size of the LSTM layer(s).
            num_layers (int): Number of stacked LSTM layers.
        """
        super(Group_Activity_Temporal_Classifier, self).__init__()

        logger.info("Initializing Group Activity Temporal Classifier...")
        self.lstm = nn.LSTM(
                            input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True
                        )
        
        self.fc = nn.Sequential(
            nn.Linear(input_size+hidden_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

        logger.info("Model Initialized Successfully!")
    

    def forward(self, x):
        """Forward pass of the temporal classifier.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes).
        """
        # x: (batch, seq_len, input_size)
        xx, (h, c) = self.lstm(x)  # xx: (batch, seq_len, hidden_size)

        # Concatenate original input and LSTM representations along feature dimension
        x = torch.cat([x, xx], dim=2)  # (batch, seq_len, input_size + hidden_size)

        # Use the last time-step representation for classification
        x = x[:, -1, :]  # (batch, input_size + hidden_size)
        x = self.fc(x)

        return x