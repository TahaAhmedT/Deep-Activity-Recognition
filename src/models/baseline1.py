import torch.nn as nn

class Activity_Classifier(nn.Module):
    """Extended model that adds a custom classification layer to a backbone.

    Attributes:
        backbone (nn.Module): Feature extractor (e.g., truncated ResNet).
        fc_layer (nn.Sequential): Fully connected layer for classification.
        verbose (bool): If True, prints info logs.
    """

    def __init__(self, backbone, n_classes: int, verbose=False):
        """
        Args:
            backbone (nn.Module): Feature extractor model.
            verbose (bool, optional): If True, prints info logs. Defaults to False.
        """
        super(Activity_Classifier, self).__init__()
        self.backbone = backbone
        self.fc_layer = nn.Sequential(
            nn.Linear(2048, n_classes)
        )
        self.verbose = verbose
        if self.verbose:
            print("[INFO] ExtendedModel initialized with backbone and FC layer.")

    def forward(self, x):
        """Forward pass through the backbone and classification layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output logits for classification.
        """
        if self.verbose:
            print(f"[INFO] Forward pass with input shape: {x.shape}")
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        if self.verbose:
            print(f"[INFO] Output shape after FC layer: {x.shape}")
        return x
    