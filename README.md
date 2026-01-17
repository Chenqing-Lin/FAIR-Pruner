# FAIR-Pruner (PyTorch)

**FAIR-Pruner** is a PyTorch pruning toolkit that performs **flexible, automatic layer-wise pruning** using the principle of **Tolerance of Difference (ToD)**. The workflow is designed to be straightforward and reproducible: you (1) compute pruning statistics on a calibration/analysis loader, (2) derive layer-wise pruning ratios from a ToD level, (3) generate a pruned “skeleton” model, then (4) instantiate the final pruned model by transferring weights and emitting a pruning report.

This repository includes a usage notebook organized into two parts:
- **Part A:** pruning a standard model (e.g., `torchvision.models.vgg16`)
- **Part B:** pruning a user-defined model (custom PyTorch modules)

---

## Installation

The notebook installs the package via pip as:

```bash
pip install network-pruner
```

Then imports the pruning API as:

```python
from Network_Pruner import FAIR_Pruner as fp
```

---

## How it works?

FAIR-Pruner is used through four main calls:

1. **Compute pruning metrics**
```python
results = fp.get_metrics(
    model,
    dataloader,
    the_samplesize_for_compute_distance=32
)
```

2. **Convert metrics → layer-wise pruning ratios (controlled by ToD)**
```python
ratios = fp.get_ratios(model, results, ToD_level=0.015)
```

3. **Build a pruned model “skeleton” (architecture with reduced channels/units)**
```python
pruned_skeleton = fp.get_skeleton(
    model=model,
    ratios=ratios,
    example_inputs=example_inputs
)
```

4. **Create the final pruned model + report**
```python
pruned_model, report = fp.prune(
    pruned_skeleton,
    model,          # original model (source for weight transfer)
    results,
    ratios,
    example_inputs=example_inputs
)
```

---

## Quick Start (Pruning A Standard Model: VGG16)

This snippet mirrors the notebook’s Part A, including CIFAR-10 setup and the calibration subset.

```python
from torch.utils.data import DataLoader, Subset
from Network_Pruner import FAIR_Pruner as fp
from torchvision.models import VGG16_Weights
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision
import torch

# 1) Load a standard model
model = models.vgg16(weights=VGG16_Weights.DEFAULT)

# 2) Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 3) Build a small analysis loader (calibration set) + test loader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
idx = torch.randperm(len(trainset), generator=torch.Generator().manual_seed(0))[:32]
subset = Subset(trainset, idx)
analysis_ds_loader = DataLoader(subset, batch_size=32, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

example_inputs = next(iter(analysis_ds_loader))[0]

# 4) Compute pruning statistics
results = fp.get_metrics(
    model,
    analysis_ds_loader,
    the_samplesize_for_compute_distance=2
)

# 5) Derive pruning ratios from ToD
ratios = fp.get_ratios(model, results, ToD_level=0.015)

# 6) Build skeleton + prune
pruned_model_skeleton = fp.get_skeleton(model=model, ratios=ratios, example_inputs=example_inputs)
pruned_model, report = fp.prune(
    pruned_model_skeleton,
    model,
    results,
    ratios,
    example_inputs=example_inputs
)

print(report)
```

Notes:
- `ToD_level` is the main knob controlling pruning aggressiveness.
- `the_samplesize_for_compute_distance` trades off speed vs. stability of the distance estimation (the notebook uses `2`).

---

## Example (User-Defined Model)

This part trains a simple custom fully-connected network on CIFAR-10, then prunes it using the same pipeline.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from Network_Pruner import FAIR_Pruner as fp

class CustomFCModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CustomFCModel, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(input_size, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64),  nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc_layers(x)

# Hyperparameters
input_size = 3 * 32 * 32
num_classes = 10
batch_size = 64
learning_rate = 0.001
epochs = 5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Reuse the CIFAR-10 trainset from the standard-model example
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

model = CustomFCModel(input_size, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train (short demo run)
model.train()
for epoch in range(epochs):
    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, pred = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/total:.4f}, Train Acc: {100*correct/total:.2f}%")

# Prune
print("Pruning CustomFCModel...")

custom_results = fp.get_metrics(
    model,
    train_loader,
    loss_function=criterion,
    device=device,
    the_samplesize_for_compute_distance=2
)

custom_ratios = fp.get_ratios(model, custom_results, ToD_level=0.05)

example_input = next(iter(train_loader))[0]
pruned_skeleton = fp.get_skeleton(
    model=model,
    ratios=custom_ratios,
    example_inputs=example_input,
    verbose=True
)

pruned_model, pruning_report = fp.prune(
    pruned_skeleton,
    model,
    custom_results,
    custom_ratios,
    example_inputs=example_input,
    device=device
)

print(pruning_report)
print(pruned_model)
```

---

## Citation

If you use FAIR-Pruner in your research, please cite:

```bibtex
@article{lin2025fair,
  title={FAIR-Pruner: Leveraging Tolerance of Difference for Flexible Automatic Layer-Wise Neural Network Pruning},
  author={Lin, Chenqing and Hussien, Mostafa and Yu, Chengyao and Jing, Bingyi and Cheriet, Mohamed and Abdelrahman, Osama and Ming, Ruixing},
  journal={arXiv preprint arXiv:2508.02291},
  year={2025}
}
```

---
