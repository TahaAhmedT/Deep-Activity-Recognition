import torch.nn as nn

# Extend the truncated model with custom layers
class ExtendedModel(nn.Module):
    def __init__(self, backbone):
        super(ExtendedModel, self).__init__()
        self.backbone = backbone
        self.fc_layer = nn.Sequential(
            nn.Linear(2048, 8)  # we have 8 classes in our dataset
        )

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x