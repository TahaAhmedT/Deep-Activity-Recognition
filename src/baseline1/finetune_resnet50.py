import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.models import resnet50
from torch.utils.data import DataLoader
from src.baseline1.dataset import B1Dataset
from src.baseline1.extended_model import ExtendedModel
from src.utils.train_test.train_step import train_step
from src.utils.train_test.test_step import test_step

from src.utils.config_utils import load_config
CONFIG = load_config()

# Data transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

batch_size = CONFIG["TRAINING_PARAMS"]["batch_size"]
train_dataset = B1Dataset(videos_root=CONFIG["PATH"]["videos_root"], target_videos=CONFIG["TARGET_VIDEOS"]["train_ids"], transform=transform)
test_dataset = B1Dataset(videos_root=CONFIG["PATH"]["videos_root"], target_videos=CONFIG["TARGET_VIDEOS"]["val_ids"], transform=transform)

trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"

original_model = resnet50(pretrained=True)

# Remove the last 4 main blocks (layer3, layer4, avgpool, fc)
layers = list(original_model.children())[:-4]
# By debugging: our last layer's shape: 512 * 28 * 28

# Create a new model with the modified layers
truncated_model = nn.Sequential(*layers)

model = ExtendedModel(truncated_model)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=CONFIG["TRAINING_PARAMS"]["lr"]) # Only optimize unfrozen params

# Training  and Testing Loop
NUM_EPOCHS = CONFIG["TRAINING_PARAMS"]["num_epochs"]
for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch+1}\n-----------------------")
    # Training step
    train_step(data_loader=trainloader,
               model=model,
               loss_fn=criterion,
               optimizer=optimizer,
               device=device)
    
    # Testing Step
    test_step(data_loader=testloader,
              model=model,
              loss_fn=criterion,
              device=device)

