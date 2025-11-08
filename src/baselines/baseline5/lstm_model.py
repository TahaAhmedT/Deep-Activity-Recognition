from src.utils.logging_utils import setup_logger
from src.utils.config_utils import load_config

import torch
import torch.nn as nn

class Pooled_Players_Activity_Temporal_Classifier(nn.Module):
    def __init__(self, num_classes: int, input_size: int, hidden_size: int, num_layers: int, log_dir: str, verbose: bool):
        super(Pooled_Players_Activity_Temporal_Classifier, self).__init__()

        self.logger = setup_logger(
            log_dir=__file__,
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