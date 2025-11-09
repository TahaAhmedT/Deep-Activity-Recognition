"""
LSTM-based temporal classifier that processes per-player features sequences before max pooling.

This module defines Pooled_Players_Activity_Temporal_Classifier which:
1. Takes sequences of per-player feature vectors (e.g., from ResNet-50) 
2. Processes each player's sequence independently through an LSTM
3. Max pools across players to get clip-level features
4. Applies a classification head to predict group activity

Expected input shape:
    (batch_size, sequence_length, num_players, feature_dim)
"""

from src.utils.logging_utils import setup_logger
from src.utils.config_utils import load_config

import torch
import torch.nn as nn

class Pooled_Players_Activity_Temporal_Classifier(nn.Module):
    """LSTM-based classifier that processes per-player sequences before pooling.

    The model processes each player's feature sequence independently through an LSTM,
    pools across players, and then classifies the pooled representation. This architecture
    lets the model learn player-specific temporal patterns before combining information
    across the team.
    """

    def __init__(self, num_classes: int, input_size: int, hidden_size: int, num_layers: int, log_dir: str, verbose: bool):
        """Initialize the temporal classifier.

        Args:
            num_classes (int): Number of activity classes to predict.
            input_size (int): Dimension of input feature vectors per player per frame.
            hidden_size (int): Hidden size of the LSTM layer(s).
            num_layers (int): Number of stacked LSTM layers.
            log_dir (str): Directory for logger output.
            verbose (bool): If True, enable console logging.

        Network structure:
            - LSTM(input_size -> hidden_size, num_layers) per player
            - Max pool across players
            - Classification head: Linear(hidden_size -> 512) -> BN -> ReLU -> Dropout
                               -> Linear(512 -> 128) -> ReLU -> Dropout
                               -> Linear(128 -> num_classes)
        """
        super(Pooled_Players_Activity_Temporal_Classifier, self).__init__()

        self.logger = setup_logger(
            log_file=__file__,
            log_dir=log_dir,
            log_to_console=verbose,
            use_tqdm=True
        )
        self.logger.info("Initializing Pooled Players Activity Temporal Classifier...")
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        self.logger.info("Model Initialized Successfully!")
    

    def forward(self, x):
        """Forward pass of the temporal classifier.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, num_players, features_dim).
                Contains feature sequences for each player in the clip.

        Returns:
            torch.Tensor: Output activity logits of shape (batch, num_classes).

        The forward pass:
            1. Reshape input to process each player sequence independently
            2. Run LSTM on each player sequence
            3. Take final timestep representation for each player
            4. Max pool across players
            5. Apply classification head
        """
        # x: (batch, seq, num_players, features_dim)
        batch, seq_len, num_players, features_dim = x.shape
        x = x.view(batch * num_players, seq_len, features_dim) 

        x, _ = self.lstm(x) # x: (batch * num_players, seq_len, hidden_size)

        # Use the last time-step representation for classification
        x = x[:, -1, :] # x: (batch * num_players, hidden_size)

        x = x.view(batch, num_players, -1) # x: (batch, num_players, hidden_size)

        # Max pool over num_players
        x = torch.max(x, 1)[0] # x: (batch, hidden_size)

        x = self.fc(x) # x: (batch, num_classes)
        return x