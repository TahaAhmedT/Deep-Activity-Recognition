import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
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

batch_size = 32
train_dataset = B1Dataset(videos_root='path/to/videos', target_videos=[0, 1, 2], transform=transform)
test_dataset = B1Dataset(videos_root='path/to/videos', target_videos=[3, 4, 5], transform=transform)

trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

original_model = resnet50(pretrained=True)

# Remove the last 4 main blocks (layer3, layer4, avgpool, fc)
layers = list(original_model.children())[:-4]
# By debugging: our last layer's shape: 512 * 28 * 28

# Create a new model with the modified layers
truncated_model = nn.Sequential(*layers)

model = ExtendedModel(truncated_model)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001) # Only optimize unfrozen params

# Training  and Testing Loop
for epoch in range(5):
    print(f"Epoch {epoch+1}\n-----------------------")
    # Training step
    train_step(data_loader=trainloader,
               model=model,
               loss_fn=criterion,
               optimizer=optimizer,
               device='cuda' if torch.cuda.is_available() else 'cpu')
    # for batch_idx, (data, target) in enumerate(trainloader):
    #     optimizer.zero_grad()
    #     output = model(data)
    #     loss = criterion(output, target)
    #     loss.backward()
    #     optimizer.step()

    #     if batch_idx % 10 == 0:
    #         print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}')
    # Testing Step
    test_step(data_loader=testloader,
              model=model,
              loss_fn=criterion,
              device='cuda' if torch.cuda.is_available() else 'cpu')

# Model Evaluation
# model.eval()
# correct = 0
# total = 0
# with torch.no_grad():
#     for data, target in testloader:
#         output = model(data)
#         _, predicted = torch.max(output, 1)
#         total += target.size(0)
#         correct += (predicted == target).sum().item()
# print(f"Accuracy: {(correct / total) * 100}%")
