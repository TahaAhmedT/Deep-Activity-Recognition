"""
Two-stage temporal classifier for group activity recognition (baseline7).

This module implements a Two_Stage_Activity_Temporal_Classifier that models player- and
frame-level temporal dynamics in two stages:

1. Per-player LSTM: processes each player's feature sequence independently to capture
   player-specific temporal patterns and produces a per-player summary.
2. Player pooling + Frame LSTM: max-pools per-player summaries to get frame-level
   representations, then applies a second LSTM over the sequence of frame representations
   to capture clip-level temporal context before final classification.

Expected input shape:
    (batch_size, seq_len, num_players, feature_dim)

The classifier is intended to be used as a clip-level predictor for volleyball group activities.
"""
from src.utils.logging_utils import setup_logger

import torch
import torch.nn as nn


class Two_Stage_Activity_Temporal_Classifier(nn.Module):
    """Two-stage LSTM temporal classifier.

    The architecture:
        - player_lstm: LSTM(input_size -> hidden_size1) applied per player sequence
        - max-pool across players to obtain a frame representation
        - frame_lstm: LSTM(hidden_size1 -> hidden_size2) applied over frame representations
        - fc: classification head mapping final frame-level representation to class logits

    Attributes:
        player_lstm (nn.LSTM): LSTM that models per-player temporal dynamics.
        frame_lstm (nn.LSTM): LSTM that models temporal dynamics across frames.
        fc (nn.Sequential): Fully-connected classification head.
        logger (logging.Logger): Logger for informational messages.
    """

    def __init__(self, num_classes: int, input_size: int, hidden_size1: int, hidden_size2: int, num_layers: int, log_dir: str, verbose: bool):
        """Initialize the two-stage temporal classifier.

        Args:
            num_classes (int): Number of activity classes for prediction.
            input_size (int): Dimensionality of input feature vectors per player per frame.
            hidden_size1 (int): Hidden size of the per-player LSTM.
            hidden_size2 (int): Hidden size of the frame-level LSTM.
            num_layers (int): Number of layers for both LSTMs.
            log_dir (str): Directory path used by the logger.
            verbose (bool): If True, enable console logging.

        """
        super().__init__()
        
        self.logger = setup_logger(
            log_file=__file__,
            log_dir=log_dir,
            log_to_console=verbose,
            use_tqdm=True
        )
        self.logger.info("Initializing Two Stage Activity Temporal Classifier...")
        self.player_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size1,
            num_layers=num_layers,
            batch_first=True
        )

        self.adaptive_max_pool = nn.AdaptiveMaxPool1d(1)

        self.frame_lstm = nn.LSTM(
            input_size=hidden_size1,
            hidden_size=hidden_size2,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size2, hidden_size2),
            nn.BatchNorm1d(hidden_size2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        self.logger.info("Model Initialized Successfully!")

    def forward(self, x):
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor with shape (batch, seq_len, num_players, features_dim).

        Returns:
            torch.Tensor: Logits tensor of shape (batch, num_classes).
        """
        # x: (batch, seq, num_players, features_dim)
        batch, seq_len, num_players, features_dim = x.shape

        # -----------------------------------------------------
        # 1. Per-player LSTM (temporal modeling per player)
        # -----------------------------------------------------
        # reshape: players become separate sequences
        x = x.view(batch * num_players, seq_len, features_dim)

        player_out, _ = self.player_lstm(x) # player_out: (batch * num_players, seq_len, hidden_size1)

        player_out = player_out.view(batch, num_players, seq_len, -1)

        # ----------------------------------------------------
        # 2. Max-pool players -> frame representation
        # ----------------------------------------------------
        player_out = player_out.permute(0, 2, 3, 1) # (batch, seq_len, hidden_size1, num_players)
        player_out = player_out.contiguous()
        player_out = player_out.view(batch * seq_len, -1, num_players)

        player_out = self.adaptive_max_pool(player_out) # (batch*seq_len, hidden_size1, 1)
        player_out = player_out.squeeze() # (batch*seq_len, hidden_size1)
        player_out = player_out.view(batch, seq_len, -1)

        # ----------------------------------------------------
        # 3. Second LSTM: sequence over frames (clip-level temporal modeling)
        # ----------------------------------------------------
        clip_out, _ = self.frame_lstm(player_out) # (batch, seq_len, hidden_size2)

        # last frame output -> clip representation
        clip_rep = clip_out[:, -1, :] # (batch, hidden_size2)

        # ---------------------------------------------------
        # 4. Final classification layer
        # ---------------------------------------------------
        out = self.fc(clip_rep)

        return out
