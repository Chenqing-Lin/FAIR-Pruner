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
- **Automatic Layer-wise Pruning:** Automatically identifies heterogeneous redundancy patterns. It aggressively prunes redundant layers while protecting task-critical ones without manual per-layer budgets.
- **One-Shot Performance:** Grounded in a comprehensive evaluation of model health, achieving competitive accuracy immediately after pruning—often making post-pruning fine-tuning unnecessary.
- **Framework Native:** Fully integrated with PyTorch, supporting a wide range of modules including CNNs, MLPs, and LSTMs.
- **Automatic Layer-wise Pruning:** Eliminates manual hyperparameter tuning by adaptively determining the sparsity level of each layer.
- **Flexible Deployment:** Decouples importance estimation from threshold determination, allowing users to generate models at varying pruning ratios effortlessly.

## Quick Start (Demo Example)
```
from Network_Pruner import FAIR_Pruner as fp
```
## Preset the basic necessary parameters
```
import torch
import torch.nn as nn
import pickle
from torchvision.models import VGG16_Weights
import torchvision.models as models

analysis_data_path =  r'cifar10_prune_dataset.pkl'                                                  # used only for statistics collection 
model = models.vgg16(weights=VGG16_Weights.DEFAULT)                                                 # already initialized / loaded / trained ()                                        
layer2prune = [2,4,7,9,12,14,16,19,21,23,26,28,30,35,38,41]                         # The index of the layer where the units to be pruned are located. It is also used when calculating statistics.
analysis_layer = [3,5,8,10,13,15,17,20,22,24,27,29,31,36,39]                # used only to compute the node statistics(Often, it is the activation layer between the current unit and the next layer of units.)
loss_function = nn.CrossEntropyLoss()                                                               # The loss function used when training the model            
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')                               # The device we use                                                                              

finetune_pruned = False				                                                       # Optional fine-tuning settings (disabled by default)
finetune_epochs = 20  			                                                           # used only when finetune_pruned == True
val_data_path = r'cifar10_val_dataset.pkl'                                              # used only when finetune_pruned == True, used only if fine-tuning is enabled
finetune_data_path = r'Cifar10_train_dataset.pkl'                                          # used only when finetune_pruned == True, used only if fine-tuning is enabled
```
## Start pruning

### Calculate statistics
```
with open(analysis_data_path, 'rb') as f:                 
    analysis_datasetloader = pickle.load(f)
with open(val_data_path, 'rb') as f:
    val_datasetloader = pickle.load(f)
with open(finetune_data_path, 'rb') as f:
    finetune_datasetloader = pickle.load(f)
results = fp.get_metrics(model, analysis_ds_loader, layer2prune,
                analysis_layer, loss_function=loss_function,the_samplesize_for_compute_distance=2)
```
### Determine the number of neurons that should be prune off in each layer based on the ToD level
```
ratios = fp.get_ratios(results,layer2prune,0.05)
```
### Define a pruned Tiny network class
```
example_inputs = next(iter(analysis_ds_loader))[0]
pruned_model_skeleton = fp.get_skeleton(model=model,named_modules_indices=the_list_of_layers_to_prune,ratios=ratios,example_inputs=example_inputs,align_residual_add=True,verbose=True,strict_forward_check=True,)
```
### Copy the parameters of the original model to the small network model and save tiny_model
```
pruned_model,report = fp.prune(pruned_model_skeleton,model,
               results,ratios,
               layer2prune,example_inputs=example_inputs,finetune_pruned=False,finetune_epochs=10,finetunedata=analysis_ds_loader,valdata=analysis_ds_loader,device=device)
```
###  Print / log key outputs
```
print("Pruning ratio:", report['pruning rate'])
print("Num. of pruned parameters:", report['parameters number'])
```

## Requirements

- Python 3.7.7
- PyTorch 1.13.1
- torchvision 0.14.1+cu117
- scipy 1.10.1
- numpy
- pickle
## Compatibility
- Currently, this library is exclusively designed for PyTorch.

# Final Thoughts
- The model saved after pruning is the final pruned version that can be used for further training or evaluation.
- You can experiment with different ToD levels to see how pruning affects the model’s performance.
- Make sure to adjust the layer pruning configurations (the_list_of_layers_to_prune, the_list_of_layers_to_compute_Distance) according to your model architecture.
- Our code only provides the pruned backbone of the VGG16 model as an example, in order to facilitate users' understanding and learning. The framework is designed to be simple and scalable, which means our approach can be applied to other model architectures as well.
