# FAIR-Pruner: ToD-based Automatic Identification and Removal

## Project Overview

**FAIR-Pruner** is a statistically grounded, search-free framework for structured neural network pruning. It automatically determines layer-wise sparsity through a novel Tolerance of Difference (ToD) mechanism, eliminating the need for expensive architecture search or manual hyperparameter tuning. This repo is the Python implementation of the proposed pruning method. 

# Network-Pruner

[![PyPI](https://img.shields.io/pypi/v/network-pruner)](https://pypi.org/project/network-pruner/)

**Welcome to try our method!**  
Install the package in one line and start pruning immediately:
```bash
pip install network-pruner
```
## Key Features
- **Search-Free & Efficient:** Decouples importance estimation from sparsity allocation. Once scores are computed, you can generate models at any compression rate instantly by adjusting the ToD parameter $\alpha$, with zero additional retraining or search cost.
- **Statistically Grounded:** Built on a rigorous theoretical framework. We prove that the U-score is uniformly consistent and that ToD-based pruning recovers population-optimal pruning sets with vanishing error probability.
- **Automatic Layer-wise Allocation:** Automatically identifies heterogeneous redundancy patterns. It aggressively prunes redundant layers while protecting task-critical ones without manual per-layer budgets.

## Requirements

- Python 3.7.7
- PyTorch 1.13.1
- torchvision 0.14.1+cu117
- scipy 1.10.1
- numpy
- pickle

# Preparatory work

## Install our Network_Pruner library
For example, install Network_Pruner version 1.2. 
```{bash}
pip install network-pruner==1.2
```
## Install Dataset and Model
To facilitate the demonstration of the pruning process, I provide `Mini_CIFAR10_640case.pkl` a small dataset with 640 pieces belonging to the CIFAR10 dataset and `CIFAR10_vgg16.pht` a VGG16 model trained on the CIFAR10 dataset.

# Pruning Example


## Import our method
```
from Network_Pruner import FAIR_Pruner as fp
```
## Preset the necessary parameters
```
import torch
import torch.nn as nn

model_path = r'../CIFAR10_vgg16.pht'
data_path =  r'../cifar10_prune_dataset.pkl'
with open(data_path, 'rb') as f:
    prune_datasetloader = pickle.load(f)
model = torch.load(model_path)
tiny_model_save_path = r'../test_tiny_model.pht'
the_list_of_layers_to_prune = [2,4,7,9,12,14,16,19,21,23,26,28,30,35,38,41]
the_list_of_layers_to_compute_Distance = [3,5,8,10,13,15,17,20,22,24,27,29,31,36,39]
loss_function = nn.CrossEntropyLoss() # The loss function used when training the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class_num = 10
```
## Start pruning

### Calculate the Reconstruction Error and Distance
```
results = fp.FAIR_Pruner_get_results(model, prune_datasetloader, results_save_path,the_list_of_layers_to_prune,
            the_list_of_layers_to_compute_Distance, loss_function, device,class_num,the_samplesize_for_compute_distance=16,class_num_for_distance=None,num_iterations=1)
k_list = get_k_list(results,   the_list_of_layers_to_prune,0.05)
```
### Determine the number of neurons that should be prune off in each layer based on the ToD level
```
k_list = fp.get_k_list(results, the_list_of_layers_to_prune, ToD_level = 0.05)
```
### Define a pruned Tiny network class
```
class Tiny_model_class(nn.Module):
    def __init__(self):
        super(Tiny_model_class, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64 - k_list[0], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64 - k_list[0], 64 - k_list[1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64 - k_list[1], 128 - k_list[2], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128 - k_list[2], 128 - k_list[3], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128 - k_list[3], 256 - k_list[4], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256 - k_list[4], 256 - k_list[5], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256 - k_list[5], 256 - k_list[6], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256 - k_list[6], 512 - k_list[7], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512 - k_list[7], 512 - k_list[8], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512 - k_list[8], 512 - k_list[9], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512 - k_list[9], 512 - k_list[10], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512 - k_list[10], 512 - k_list[11], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512 - k_list[11], 512 - k_list[12], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.classifier = nn.Sequential(
            nn.Linear((512 - k_list[12]) * 7 * 7, 4096 - k_list[13]),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096 - k_list[13], 4096 - k_list[14]),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096 - k_list[14], 1000 - k_list[15])
        )
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

tiny_model = Tiny_model_class()
```
### Copy the parameters of the original model to the small network model and save tiny_model
```
tiny_model = fp.Generate_model_after_pruning(tiny_model,model_path,
                             tiny_model_save_path,
                             results,k_list,
                             the_list_of_layers_to_prune)
```

# Final Thoughts
- The model saved after pruning is the final pruned version that can be used for further training or evaluation.
- You can experiment with different ToD levels to see how pruning affects the model’s performance.
- Make sure to adjust the layer pruning configurations (the_list_of_layers_to_prune, the_list_of_layers_to_compute_Distance) according to your model architecture.
