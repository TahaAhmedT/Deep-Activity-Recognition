from utils.logging_utils import setup_logger

import torch.nn as nn


class Activity_Classifier(nn.Module):
    """Extended model that adds a custom classification layer to a backbone.

    Attributes:
        backbone (nn.Module): Feature extractor (e.g., truncated ResNet).
        fc_layer (nn.Sequential): Fully connected layer for classification.
        verbose (bool): If True, prints info logs.
    """

    def __init__(self, backbone, n_classes: int, log_dir: str, verbose=False):
        """
        Args:
            backbone (nn.Module): Feature extractor model.
            verbose (bool, optional): If True, prints info logs. Defaults to False.
        """
        super(Activity_Classifier, self).__init__()

        self.logger = setup_logger(
            log_file=__file__,
            log_dir=log_dir,
            log_to_console=verbose,
            use_tqdm=True,
        )

        self.logger.info("Initializing Activity Classifier...")        
        self.backbone = backbone
        self.fc_layer = nn.Sequential(
            nn.Linear(2048, n_classes)
        )
        
        self.logger.info("Activity Classifier Initialized Successfully!")

    def forward(self, x):
        """Forward pass through the backbone and classification layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output logits for classification.
        """
        
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)

        return x
    